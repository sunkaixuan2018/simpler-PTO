#!/usr/bin/env python3
"""
Per-rank Python worker for distributed (multi-card) kernel execution.

Replaces the monolithic C++ distributed_worker binary.  Each rank runs
as a separate process, using the comm_* C API (via ctypes bindings) for
HCCL / sim communication and the existing PTO runtime C API for kernel
execution.

Spawned by DistributedCodeRunner — not intended for direct invocation.
"""

import argparse
import ctypes
import struct
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "python"))
sys.path.insert(0, str(script_dir))


DTYPE_FORMAT = {
    "float32": ("f", 4),
    "float64": ("d", 8),
    "int32": ("i", 4),
    "int64": ("q", 8),
    "uint32": ("I", 4),
    "uint64": ("Q", 8),
    "float16": ("e", 2),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int8": ("b", 1),
    "uint8": ("B", 1),
}


ARG_DIRECTION_SCALAR = 0
CALLABLE_ALIGN = 64
CORE_MAX_TENSOR_ARGS = 16
CHIP_MAX_TENSOR_ARGS = 64
CHIP_MAX_CHILDREN = 32
CALLABLE_FUNC_NAME_MAX = 64
CORE_BINARY_OFFSET = 128
CHIP_HEADER_SIZE = 660
CONTINUOUS_TENSOR_SIZE = 40
CHIP_MAX_SCALAR_ARGS = 128
CHIP_STORAGE_TENSOR_BYTES = CHIP_MAX_TENSOR_ARGS * CONTINUOUS_TENSOR_SIZE
CHIP_STORAGE_SCALAR_BYTES = CHIP_MAX_SCALAR_ARGS * 8
CHIP_STORAGE_ARGS_SIZE = CHIP_STORAGE_TENSOR_BYTES + CHIP_STORAGE_SCALAR_BYTES + 8


def align_up(value, alignment=CALLABLE_ALIGN):
    return (value + alignment - 1) & ~(alignment - 1)


def build_core_callable(binary):
    buf = bytearray(CORE_BINARY_OFFSET + len(binary))
    struct.pack_into("<i", buf, CORE_MAX_TENSOR_ARGS * 4, 0)
    struct.pack_into("<I", buf, CORE_MAX_TENSOR_ARGS * 4 + 4, len(binary))
    struct.pack_into("<Q", buf, CORE_MAX_TENSOR_ARGS * 4 + 8, 0)
    buf[CORE_BINARY_OFFSET:] = binary
    return bytes(buf)


def _write_c_string(buf, offset, len_offset, value):
    data = value.encode("utf-8")[:CALLABLE_FUNC_NAME_MAX - 1]
    buf[offset:offset + len(data)] = data
    struct.pack_into("<I", buf, len_offset, len(data))


def build_chip_callable(func_name, binary, children, arg_count):
    if arg_count > CHIP_MAX_TENSOR_ARGS:
        raise ValueError(f"Too many callable args: {arg_count}")
    if len(children) > CHIP_MAX_CHILDREN:
        raise ValueError(f"Too many child kernels: {len(children)}")

    child_offsets = []
    storage_size = len(binary)
    for _, child in children:
        storage_size = align_up(storage_size)
        child_offsets.append(storage_size)
        storage_size += len(child)

    buf = bytearray(CHIP_HEADER_SIZE + storage_size)
    for i in range(arg_count):
        struct.pack_into("<i", buf, i * 4, ARG_DIRECTION_SCALAR)
    struct.pack_into("<i", buf, CHIP_MAX_TENSOR_ARGS * 4, arg_count)
    struct.pack_into("<I", buf, CHIP_MAX_TENSOR_ARGS * 4 + 4, len(binary))
    _write_c_string(buf, 264, 328, func_name)

    for i, (func_id, _) in enumerate(children):
        struct.pack_into("<i", buf, 332 + i * 4, func_id)
    for i, child_offset in enumerate(child_offsets):
        struct.pack_into("<I", buf, 460 + i * 4, child_offset)
    struct.pack_into("<i", buf, 588, len(children))

    storage_base = CHIP_HEADER_SIZE
    buf[storage_base:storage_base + len(binary)] = binary
    for child_offset, (_, child) in zip(child_offsets, children):
        start = storage_base + child_offset
        buf[start:start + len(child)] = child

    return bytes(buf)


def build_scalar_chip_args(values):
    if len(values) > CHIP_MAX_SCALAR_ARGS:
        raise ValueError(f"Too many scalar args: {len(values)}")

    buf = bytearray(CHIP_STORAGE_ARGS_SIZE)
    for i, value in enumerate(values):
        struct.pack_into("<Q", buf, CHIP_STORAGE_TENSOR_BYTES + i * 8,
                         int(value) & 0xFFFFFFFFFFFFFFFF)
    struct.pack_into("<i", buf, CHIP_STORAGE_TENSOR_BYTES + CHIP_STORAGE_SCALAR_BYTES, 0)
    struct.pack_into("<i", buf, CHIP_STORAGE_TENSOR_BYTES + CHIP_STORAGE_SCALAR_BYTES + 4,
                     len(values))
    return bytes(buf)


class HostRuntimeApi:
    def __init__(self, lib_path, aicpu_path, aicore_path):
        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        self.lib = ctypes.CDLL(str(lib_path), mode=mode)
        self.aicpu_binary = Path(aicpu_path).read_bytes()
        self.aicore_binary = Path(aicore_path).read_bytes()
        self.comm_streams = {}
        self.closed = False
        self._setup_functions()
        self.ctx = self.lib.create_device_context()
        if not self.ctx:
            raise RuntimeError("create_device_context failed")

    def _setup_functions(self):
        lib = self.lib
        lib.create_device_context.argtypes = []
        lib.create_device_context.restype = ctypes.c_void_p
        lib.destroy_device_context.argtypes = [ctypes.c_void_p]
        lib.destroy_device_context.restype = None
        lib.get_runtime_size.argtypes = []
        lib.get_runtime_size.restype = ctypes.c_size_t
        lib.set_device.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.set_device.restype = ctypes.c_int
        lib.device_malloc_ctx.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.device_malloc_ctx.restype = ctypes.c_void_p
        lib.device_free_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.device_free_ctx.restype = None
        lib.copy_to_device_ctx.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.copy_to_device_ctx.restype = ctypes.c_int
        lib.copy_from_device_ctx.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.copy_from_device_ctx.restype = ctypes.c_int
        lib.run_runtime.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int,
        ]
        lib.run_runtime.restype = ctypes.c_int
        lib.finalize_device.argtypes = [ctypes.c_void_p]
        lib.finalize_device.restype = ctypes.c_int
        lib.ensure_acl_ready_ctx.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.ensure_acl_ready_ctx.restype = ctypes.c_int
        lib.create_comm_stream_ctx.argtypes = [ctypes.c_void_p]
        lib.create_comm_stream_ctx.restype = ctypes.c_void_p
        lib.destroy_comm_stream_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.destroy_comm_stream_ctx.restype = ctypes.c_int
        lib.comm_init.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p]
        lib.comm_init.restype = ctypes.c_void_p
        lib.comm_alloc_windows.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
        lib.comm_alloc_windows.restype = ctypes.c_int
        lib.comm_get_local_window_base.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        lib.comm_get_local_window_base.restype = ctypes.c_int
        lib.comm_barrier.argtypes = [ctypes.c_void_p]
        lib.comm_barrier.restype = ctypes.c_int
        lib.comm_destroy.argtypes = [ctypes.c_void_p]
        lib.comm_destroy.restype = ctypes.c_int

    def set_device(self, device_id):
        rc = self.lib.set_device(self.ctx, device_id)
        if rc != 0:
            raise RuntimeError(f"set_device failed: {rc}")

    def malloc(self, nbytes):
        ptr = self.lib.device_malloc_ctx(self.ctx, nbytes)
        if not ptr:
            raise RuntimeError(f"device_malloc_ctx failed for {nbytes} bytes")
        return ptr

    def free(self, ptr):
        if ptr:
            self.lib.device_free_ctx(self.ctx, ctypes.c_void_p(ptr))

    def copy_to(self, dev_ptr, host_ptr, nbytes):
        rc = self.lib.copy_to_device_ctx(
            self.ctx, ctypes.c_void_p(dev_ptr), ctypes.c_void_p(host_ptr), nbytes)
        if rc != 0:
            raise RuntimeError(f"copy_to_device_ctx failed: {rc}")

    def copy_from(self, host_ptr, dev_ptr, nbytes):
        rc = self.lib.copy_from_device_ctx(
            self.ctx, ctypes.c_void_p(host_ptr), ctypes.c_void_p(dev_ptr), nbytes)
        if rc != 0:
            raise RuntimeError(f"copy_from_device_ctx failed: {rc}")

    def comm_init(self, rank, nranks, device_id, rootinfo_path):
        rc = self.lib.ensure_acl_ready_ctx(self.ctx, device_id)
        if rc != 0:
            raise RuntimeError(f"ensure_acl_ready_ctx failed: {rc}")

        stream = self.lib.create_comm_stream_ctx(self.ctx)
        if not stream:
            raise RuntimeError("create_comm_stream_ctx failed")

        handle = self.lib.comm_init(
            rank, nranks, ctypes.c_void_p(stream),
            str(rootinfo_path).encode("utf-8"))
        if not handle:
            self.lib.destroy_comm_stream_ctx(self.ctx, ctypes.c_void_p(stream))
            raise RuntimeError(f"comm_init failed for rank {rank}")
        self.comm_streams[int(handle)] = int(stream)
        return int(handle)

    def comm_alloc_windows(self, handle, win_size):
        device_ctx = ctypes.c_uint64(0)
        rc = self.lib.comm_alloc_windows(
            ctypes.c_void_p(handle), win_size, ctypes.byref(device_ctx))
        if rc != 0:
            raise RuntimeError(f"comm_alloc_windows failed: {rc}")
        return device_ctx.value

    def comm_get_local_window_base(self, handle):
        base = ctypes.c_uint64(0)
        rc = self.lib.comm_get_local_window_base(
            ctypes.c_void_p(handle), ctypes.byref(base))
        if rc != 0:
            raise RuntimeError(f"comm_get_local_window_base failed: {rc}")
        return base.value

    def comm_barrier(self, handle):
        rc = self.lib.comm_barrier(ctypes.c_void_p(handle))
        if rc != 0:
            raise RuntimeError(f"comm_barrier failed: {rc}")

    def comm_destroy(self, handle):
        rc = self.lib.comm_destroy(ctypes.c_void_p(handle))
        stream = self.comm_streams.pop(int(handle), 0)
        if stream:
            destroy_rc = self.lib.destroy_comm_stream_ctx(
                self.ctx, ctypes.c_void_p(stream))
            if rc == 0:
                rc = destroy_rc
        if rc != 0:
            raise RuntimeError(f"comm_destroy failed: {rc}")

    def run(self, callable_blob, args_blob, block_dim, aicpu_thread_num, device_id):
        runtime = ctypes.create_string_buffer(self.lib.get_runtime_size())
        callable_buf = ctypes.create_string_buffer(callable_blob)
        args_buf = ctypes.create_string_buffer(args_blob)
        aicpu_buf = (ctypes.c_uint8 * len(self.aicpu_binary)).from_buffer_copy(
            self.aicpu_binary)
        aicore_buf = (ctypes.c_uint8 * len(self.aicore_binary)).from_buffer_copy(
            self.aicore_binary)
        rc = self.lib.run_runtime(
            self.ctx,
            ctypes.cast(runtime, ctypes.c_void_p),
            ctypes.cast(callable_buf, ctypes.c_void_p),
            ctypes.cast(args_buf, ctypes.c_void_p),
            block_dim,
            aicpu_thread_num,
            device_id,
            aicpu_buf,
            len(self.aicpu_binary),
            aicore_buf,
            len(self.aicore_binary),
            0,
            0,
        )
        if rc != 0:
            raise RuntimeError(f"run_runtime failed: {rc}")

    def close(self):
        if self.closed:
            return
        for handle in list(self.comm_streams):
            try:
                self.comm_destroy(handle)
            except Exception:
                pass
        self.lib.finalize_device(self.ctx)
        self.lib.destroy_device_context(self.ctx)
        self.closed = True


def parse_buffer_spec(spec):
    parts = spec.split(":")
    return {"name": parts[0], "dtype": parts[1], "count": int(parts[2])}


def parse_kernel_spec(spec):
    p = spec.index(":")
    return {"func_id": int(spec[:p]), "filename": spec[p + 1:]}


def main():
    parser = argparse.ArgumentParser(description="Distributed per-rank worker")
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--nranks", type=int, required=True)
    parser.add_argument("--root", type=int, default=0)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--rootinfo-file", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--orch-file", required=True)
    parser.add_argument("--orch-func", required=True)
    parser.add_argument("--win-sync-prefix", type=int, default=0)
    parser.add_argument("--aicpu-thread-num", type=int, default=1)
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--orch-thread-num", type=int, default=0)
    parser.add_argument("--win-buffer", action="append", default=[])
    parser.add_argument("--dev-buffer", action="append", default=[])
    parser.add_argument("--load", action="append", default=[], dest="loads")
    parser.add_argument("--save", action="append", default=[], dest="saves")
    parser.add_argument("--arg", action="append", default=[], dest="args")
    parser.add_argument("--kernel-bin", action="append", default=[])
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    data_dir = Path(args.data_dir) if args.data_dir else artifact_dir / f"rank_{args.rank}"

    buffers = []
    for spec in args.win_buffer:
        b = parse_buffer_spec(spec)
        b["placement"] = "window"
        buffers.append(b)
    for spec in args.dev_buffer:
        b = parse_buffer_spec(spec)
        b["placement"] = "device"
        buffers.append(b)

    kernel_bins = [parse_kernel_spec(s) for s in args.kernel_bin]

    buf_by_name = {b["name"]: b for b in buffers}

    def elem_size(dtype):
        return DTYPE_FORMAT.get(dtype, ("f", 4))[1]

    def buf_bytes(b):
        return b["count"] * elem_size(b["dtype"])

    # ----------------------------------------------------------------
    # 1. Load library
    # ----------------------------------------------------------------
    lib_path = artifact_dir / "libhost_runtime.so"
    aicpu_path = artifact_dir / "libaicpu_kernel.so"
    aicore_path = artifact_dir / "aicore_kernel.o"

    runtime_api = HostRuntimeApi(lib_path, aicpu_path, aicore_path)
    sys.stderr.write(f"[rank {args.rank}] Library loaded\n")

    runtime_api.set_device(args.device_id)
    sys.stderr.write(f"[rank {args.rank}] Device {args.device_id} set for runtime\n")

    # ----------------------------------------------------------------
    # 2. Comm init + alloc windows
    # ----------------------------------------------------------------
    comm = runtime_api.comm_init(args.rank, args.nranks, args.device_id, args.rootinfo_file)

    total_win = args.win_sync_prefix
    for b in buffers:
        if b["placement"] == "window":
            total_win += buf_bytes(b)

    device_ctx_ptr = runtime_api.comm_alloc_windows(comm, total_win)
    local_base = runtime_api.comm_get_local_window_base(comm)

    sys.stderr.write(f"[rank {args.rank}] Comm initialized, local_base=0x{local_base:x}\n")

    # ----------------------------------------------------------------
    # 3. Allocate buffers
    # ----------------------------------------------------------------
    win_offset = args.win_sync_prefix

    for b in buffers:
        nbytes = buf_bytes(b)
        if b["placement"] == "window":
            b["dev_ptr"] = local_base + win_offset
            win_offset += nbytes
        else:
            ptr = runtime_api.malloc(nbytes)
            if not ptr:
                sys.stderr.write(f"[rank {args.rank}] device_malloc failed for '{b['name']}'\n")
                return 3
            b["dev_ptr"] = ptr
        sys.stderr.write(
            f"[rank {args.rank}] Buffer '{b['name']}': {b['placement']} "
            f"{b['count']}x{b['dtype']}={nbytes}B @ 0x{b['dev_ptr']:x}\n"
        )

    # ----------------------------------------------------------------
    # 4. Load inputs
    # ----------------------------------------------------------------
    for name in args.loads:
        b = buf_by_name.get(name)
        if not b:
            sys.stderr.write(f"[rank {args.rank}] --load: buffer '{name}' not found\n")
            return 1
        path = data_dir / f"{name}.bin"
        host_data = path.read_bytes()
        if len(host_data) != buf_bytes(b):
            sys.stderr.write(
                f"[rank {args.rank}] Size mismatch for '{name}': "
                f"file={len(host_data)}, expected={buf_bytes(b)}\n"
            )
            return 2
        host_buf = (ctypes.c_uint8 * len(host_data)).from_buffer_copy(host_data)
        runtime_api.copy_to(b["dev_ptr"], ctypes.addressof(host_buf), len(host_data))

    # ----------------------------------------------------------------
    # 5. Barrier before kernel execution
    # ----------------------------------------------------------------
    runtime_api.comm_barrier(comm)

    # ----------------------------------------------------------------
    # 6. Run simpler runtime
    # ----------------------------------------------------------------
    orch_binary = (artifact_dir / args.orch_file).read_bytes()
    children = []
    for k in kernel_bins:
        data = (artifact_dir / k["filename"]).read_bytes()
        children.append((k["func_id"], build_core_callable(data)))

    func_args = []
    for tok in args.args:
        if tok == "nranks":
            func_args.append(args.nranks)
        elif tok == "root":
            func_args.append(args.root)
        elif tok == "deviceCtx":
            func_args.append(device_ctx_ptr)
        else:
            b = buf_by_name.get(tok)
            if not b:
                sys.stderr.write(f"[rank {args.rank}] --arg: unknown token '{tok}'\n")
                return 1
            func_args.append(b["dev_ptr"])

    sys.stderr.write(
        f"[rank {args.rank}] Launching kernel: {len(func_args)} args, "
        f"{len(children)} kernels\n"
    )

    chip_callable = build_chip_callable(args.orch_func, orch_binary, children, len(func_args))
    chip_args = build_scalar_chip_args(func_args)

    runtime_api.run(
        chip_callable, chip_args, args.block_dim, args.aicpu_thread_num, args.device_id)
    sys.stderr.write(f"[rank {args.rank}] Kernel execution complete\n")

    # ----------------------------------------------------------------
    # 7. Barrier + save outputs
    # ----------------------------------------------------------------
    runtime_api.comm_barrier(comm)

    for name in args.saves:
        b = buf_by_name.get(name)
        if not b:
            sys.stderr.write(f"[rank {args.rank}] --save: buffer '{name}' not found\n")
            continue
        nbytes = buf_bytes(b)
        host_buf = (ctypes.c_uint8 * nbytes)()
        runtime_api.copy_from(ctypes.addressof(host_buf), b["dev_ptr"], nbytes)
        path = data_dir / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(host_buf))
        sys.stderr.write(f"[rank {args.rank}] Saved '{name}' to {path} ({nbytes}B)\n")

    # ----------------------------------------------------------------
    # 8. Cleanup
    # ----------------------------------------------------------------
    for b in buffers:
        if b["placement"] == "device" and b.get("dev_ptr"):
            runtime_api.free(b["dev_ptr"])

    runtime_api.comm_destroy(comm)
    runtime_api.close()
    sys.stderr.write(f"[rank {args.rank}] Done\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

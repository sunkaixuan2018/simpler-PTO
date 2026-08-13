#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate explicit VMM export/import across independently exec'd processes."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    _l3_child_onboard_region_close,
    _l3_child_onboard_region_create,
    _memory_wmb_for_test,
    _worker_host_mapped_counter_notify,
    _worker_host_mapped_counter_test,
    _worker_host_mapped_payload_read,
    _worker_host_mapped_payload_write,
    _worker_host_mapped_region_close,
    _worker_host_mapped_region_device_addr_for_test,
    _worker_host_mapped_region_import_onboard,
)
from simpler.task_interface import ArgDirection, CallConfig, ChipCallable, ChipStorageTaskArgs, ChipWorker, CoreCallable
from simpler.worker_chip_orch_comm import NotifyOp, WaitCmp

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.runtime_builder import RuntimeBuilder

_REGION_BYTES = 192
_PAYLOAD_OFFSET = 0
_TAIL_OFFSET = 64
_COMPLETION_OFFSET = 128
_EXPECTED_PAYLOAD = 0x70616D74736F6850
_EXPECTED_TAIL = 1
_TAIL_TIMEOUT = 0xFFFFFFFE
_PAYLOAD_MISMATCH = 0xFFFFFFFD
_DEV_SVM_MAP_HOST = 2
_OBSERVER_SOURCE = Path(__file__).with_name("pre_fork_svm_observer.cpp")
_COMPLETION_SOURCE = Path(__file__).with_name("pre_fork_svm_completion.cpp")
_CASES = ("acl-copy-import-control", "host-register-import")
_ROLES = ("coordinator", "l2", "l3")


def _phase(name: str, **details: Any) -> None:
    fields = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[cross-process-vmm] phase={name}{' ' + fields if fields else ''}", flush=True)


def _send_message(control: socket.socket, message: dict[str, Any]) -> None:
    control.sendall(json.dumps(message, sort_keys=True).encode("utf-8") + b"\n")


def _receive_message(control: socket.socket, timeout_s: float) -> dict[str, Any]:
    control.settimeout(timeout_s)
    chunks: list[bytes] = []
    while True:
        chunk = control.recv(4096)
        if not chunk:
            raise RuntimeError("control channel closed before a complete message arrived")
        if b"\n" in chunk:
            head, _separator, tail = chunk.partition(b"\n")
            if tail:
                raise RuntimeError("control channel received more than one message at once")
            chunks.append(head)
            return json.loads(b"".join(chunks).decode("utf-8"))
        chunks.append(chunk)


class _HalHostMapping:
    def __init__(self, device_addr: int, nbytes: int, device_id: int):
        self._device_addr = int(device_addr)
        self._device_id = int(device_id)
        self._lib = ctypes.CDLL("libascend_hal.so")
        register = self._lib.halHostRegister
        register.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        register.restype = ctypes.c_int
        self._unregister = self._lib.halHostUnregister
        self._unregister.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._unregister.restype = ctypes.c_int
        host_ptr = ctypes.c_void_p()
        rc = int(
            register(
                ctypes.c_void_p(self._device_addr),
                ctypes.c_size_t(nbytes),
                ctypes.c_uint(_DEV_SVM_MAP_HOST),
                ctypes.c_int(self._device_id),
                ctypes.byref(host_ptr),
            )
        )
        if rc != 0 or not host_ptr.value:
            raise RuntimeError(f"halHostRegister failed rc={rc} host_va={host_ptr.value}")
        self.host_addr = int(host_ptr.value)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        rc = int(self._unregister(ctypes.c_void_p(self._device_addr), ctypes.c_int(self._device_id)))
        if rc != 0:
            raise RuntimeError(f"halHostUnregister failed rc={rc}")


def _build_observer(platform: str, runtime: str) -> ChipCallable:
    compiler = KernelCompiler(platform=platform)
    include_dirs = compiler.get_orchestration_include_dirs(runtime)
    completion_binary = compiler.compile_incore(
        source_path=str(_COMPLETION_SOURCE),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=include_dirs,
    )
    if not platform.endswith("sim"):
        completion_binary = extract_text_section(completion_binary)
    completion = CoreCallable.build(signature=[ArgDirection.OUT], binary=completion_binary)
    binary = compiler.compile_orchestration(
        runtime_name=runtime,
        source_path=str(_OBSERVER_SOURCE),
        extra_include_dirs=[str(compiler.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[],
        func_name="pre_fork_svm_observer",
        config_name="aicpu_orchestration_config",
        binary=binary,
        children=[(0, completion)],
    )


def _observer_args(device_addr: int) -> ChipStorageTaskArgs:
    args = ChipStorageTaskArgs()
    args.add_scalar(1)
    args.add_scalar(int(device_addr))
    args.add_scalar(_EXPECTED_PAYLOAD)
    args.add_scalar(_EXPECTED_TAIL)
    return args


def _completion_result(value: int) -> dict[str, Any]:
    if value == _EXPECTED_TAIL:
        return {"status": "pass", "reason": "AICPU observed the L3 publication and wrote completion"}
    if value == _TAIL_TIMEOUT:
        return {"status": "fail", "reason": "AICPU did not observe the L3 tail publication"}
    if value == _PAYLOAD_MISMATCH:
        return {"status": "fail", "reason": "AICPU observed the tail but not the L3 payload"}
    return {"status": "fail", "reason": f"unexpected completion value {value}"}


def _l2_role(args: argparse.Namespace) -> int:
    control = socket.socket(fileno=int(args.control_fd))
    worker = ChipWorker()
    callable_handle = None
    registry_handle = 0
    try:
        _phase("l2-build-observer-start")
        observer = _build_observer(args.platform, args.runtime)
        binaries = RuntimeBuilder(args.platform).get_binaries(args.runtime)
        callable_handle = worker.register_callable(observer)
        worker.init(int(args.device), binaries)
        _phase("l2-worker-ready", device=args.device)
        region = _l3_child_onboard_region_create(_REGION_BYTES)
        registry_handle = int(region.registry_handle)
        zeros = ctypes.create_string_buffer(_REGION_BYTES)
        worker.copy_to(int(region.device_addr), ctypes.addressof(zeros), _REGION_BYTES)
        _send_message(
            control,
            {
                "status": "ready",
                "device_addr": int(region.device_addr),
                "mapping_bytes": int(region.mapping_bytes),
                "shareable_handle": int(region.shareable_handle),
            },
        )
        command = _receive_message(control, float(args.timeout))
        if command.get("command") != "run":
            raise RuntimeError(f"unexpected L2 command {command}")
        config = CallConfig()
        config.aicpu_thread_num = 2
        worker.run(callable_handle, _observer_args(int(region.device_addr)), config)
        completion_buffer = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint32))
        worker.copy_from(
            ctypes.addressof(completion_buffer),
            int(region.device_addr) + _COMPLETION_OFFSET,
            ctypes.sizeof(ctypes.c_uint32),
        )
        completion = int(ctypes.c_uint32.from_buffer(completion_buffer).value)
        result = _completion_result(completion)
        result.update({"role": "l2", "completion": completion, "device_addr": int(region.device_addr)})
        _send_message(control, result)
        return 0 if result["status"] == "pass" else 1
    except BaseException as exc:  # noqa: BLE001
        with contextlib.suppress(OSError):
            _send_message(control, {"status": "fail", "role": "l2", "reason": f"{type(exc).__name__}: {exc}"})
        return 1
    finally:
        if registry_handle:
            with contextlib.suppress(BaseException):
                _l3_child_onboard_region_close(registry_handle)
        if callable_handle is not None:
            with contextlib.suppress(BaseException):
                worker.unregister_callable(callable_handle)
        with contextlib.suppress(BaseException):
            worker.finalize()
        control.close()


def _acl_copy_publish(handle: int) -> dict[str, Any]:
    payload = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint64))
    ctypes.c_uint64.from_buffer(payload).value = _EXPECTED_PAYLOAD
    _worker_host_mapped_payload_write(handle, _PAYLOAD_OFFSET, ctypes.addressof(payload), len(payload))
    _worker_host_mapped_counter_notify(handle, _TAIL_OFFSET, _EXPECTED_TAIL, int(NotifyOp.Set))
    readback = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint64))
    _worker_host_mapped_payload_read(handle, _PAYLOAD_OFFSET, ctypes.addressof(readback), len(readback))
    payload_value = int(ctypes.c_uint64.from_buffer(readback).value)
    tail_matched, tail_value = _worker_host_mapped_counter_test(handle, _TAIL_OFFSET, _EXPECTED_TAIL, int(WaitCmp.EQ))
    return {
        "host_self_payload": payload_value,
        "host_self_tail": int(tail_value),
        "host_self_visible": payload_value == _EXPECTED_PAYLOAD and bool(tail_matched),
    }


def _host_register_publish(
    device_addr: int, mapping_bytes: int, device_id: int
) -> tuple[_HalHostMapping, dict[str, Any]]:
    mapping = _HalHostMapping(device_addr, mapping_bytes, device_id)
    ctypes.c_uint64.from_address(mapping.host_addr + _PAYLOAD_OFFSET).value = _EXPECTED_PAYLOAD
    _memory_wmb_for_test()
    ctypes.c_uint32.from_address(mapping.host_addr + _TAIL_OFFSET).value = _EXPECTED_TAIL
    _memory_wmb_for_test()
    payload_value = int(ctypes.c_uint64.from_address(mapping.host_addr + _PAYLOAD_OFFSET).value)
    tail_value = int(ctypes.c_uint32.from_address(mapping.host_addr + _TAIL_OFFSET).value)
    return mapping, {
        "host_addr": mapping.host_addr,
        "host_self_payload": payload_value,
        "host_self_tail": tail_value,
        "host_self_visible": payload_value == _EXPECTED_PAYLOAD and tail_value == _EXPECTED_TAIL,
    }


def _l3_role(args: argparse.Namespace) -> int:
    control = socket.socket(fileno=int(args.control_fd))
    owner = None
    mapping: _HalHostMapping | None = None
    handle = 0
    try:
        owner = _worker_host_mapped_region_import_onboard(
            int(args.device), int(args.shareable_handle), int(args.mapping_bytes), f"cross-process-{os.getpid()}"
        )
        handle = int(owner)
        imported_addr = int(_worker_host_mapped_region_device_addr_for_test(handle))
        _phase("l3-import-ready", imported_addr=imported_addr, case=args.case)
        if args.case == "acl-copy-import-control":
            publication = _acl_copy_publish(handle)
        else:
            mapping, publication = _host_register_publish(imported_addr, int(args.mapping_bytes), int(args.device))
        if not publication["host_self_visible"]:
            raise RuntimeError(f"L3 self-read failed after publication: {publication}")
        _send_message(
            control,
            {
                "status": "published",
                "role": "l3",
                "imported_device_addr": imported_addr,
                **publication,
            },
        )
        command = _receive_message(control, float(args.timeout))
        if command.get("command") != "read-completion":
            raise RuntimeError(f"unexpected L3 command {command}")
        if mapping is None:
            matched, completion = _worker_host_mapped_counter_test(
                handle, _COMPLETION_OFFSET, _EXPECTED_TAIL, int(WaitCmp.EQ)
            )
            completion = int(completion)
        else:
            completion = int(ctypes.c_uint32.from_address(mapping.host_addr + _COMPLETION_OFFSET).value)
            matched = completion == _EXPECTED_TAIL
        result = _completion_result(completion)
        result.update(
            {
                "role": "l3",
                "completion": completion,
                "completion_visible": bool(matched),
                "imported_device_addr": imported_addr,
            }
        )
        _send_message(control, result)
        return 0 if result["status"] == "pass" else 1
    except BaseException as exc:  # noqa: BLE001
        with contextlib.suppress(OSError):
            _send_message(control, {"status": "fail", "role": "l3", "reason": f"{type(exc).__name__}: {exc}"})
        return 1
    finally:
        if mapping is not None:
            with contextlib.suppress(BaseException):
                mapping.close()
        if handle:
            with contextlib.suppress(BaseException):
                _worker_host_mapped_region_close(handle)
        owner = None
        control.close()


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5.0)
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5.0)


def _child_command(args: argparse.Namespace, role: str, control_fd: int, **extra: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--role",
        role,
        "--control-fd",
        str(control_fd),
        "--platform",
        args.platform,
        "--runtime",
        args.runtime,
        "--device",
        str(args.device),
        "--case",
        args.case,
        "--timeout",
        str(args.timeout),
    ]
    for key, value in extra.items():
        command.extend((f"--{key.replace('_', '-')}", str(value)))
    return command


def _coordinator_role(args: argparse.Namespace) -> dict[str, Any]:
    l2_parent, l2_child = socket.socketpair()
    l2_process = subprocess.Popen(
        _child_command(args, "l2", l2_child.fileno()),
        pass_fds=(l2_child.fileno(),),
    )
    l2_child.close()
    l3_parent: socket.socket | None = None
    l3_process: subprocess.Popen[bytes] | None = None
    try:
        export = _receive_message(l2_parent, float(args.timeout))
        if export.get("status") != "ready":
            return {"status": "fail", "reason": "L2 failed before VMM export", "l2_export": export}
        _phase("coordinator-received-export", shareable_handle=export["shareable_handle"])
        l3_parent, l3_child = socket.socketpair()
        l3_process = subprocess.Popen(
            _child_command(
                args,
                "l3",
                l3_child.fileno(),
                shareable_handle=int(export["shareable_handle"]),
                mapping_bytes=int(export["mapping_bytes"]),
            ),
            pass_fds=(l3_child.fileno(),),
        )
        l3_child.close()
        publication = _receive_message(l3_parent, float(args.timeout))
        if publication.get("status") != "published":
            _send_message(l2_parent, {"command": "stop"})
            return {
                "status": "fail",
                "reason": "L3 failed before publishing through the imported mapping",
                "l2_export": export,
                "l3_publication": publication,
            }
        _send_message(l2_parent, {"command": "run"})
        l2_result = _receive_message(l2_parent, float(args.timeout))
        _send_message(l3_parent, {"command": "read-completion"})
        l3_result = _receive_message(l3_parent, float(args.timeout))
        passed = l2_result.get("status") == "pass" and l3_result.get("status") == "pass"
        return {
            "status": "pass" if passed else "fail",
            "reason": (
                "explicit VMM import preserved bidirectional L3/AICPU visibility"
                if passed
                else "explicit VMM import did not preserve bidirectional L3/AICPU visibility"
            ),
            "case": args.case,
            "l2_pid": l2_process.pid,
            "l3_pid": l3_process.pid,
            "independent_exec_processes": True,
            "l2_export": export,
            "l3_publication": publication,
            "l2_result": l2_result,
            "l3_result": l3_result,
        }
    except BaseException as exc:  # noqa: BLE001
        return {"status": "fail", "case": args.case, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        l2_parent.close()
        if l3_parent is not None:
            l3_parent.close()
        _stop_process(l2_process)
        if l3_process is not None:
            _stop_process(l3_process)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="a2a3", choices=("a2a3",))
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--case", required=True, choices=_CASES)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output")
    parser.add_argument("--role", choices=_ROLES, default="coordinator", help=argparse.SUPPRESS)
    parser.add_argument("--control-fd", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--shareable-handle", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--mapping-bytes", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.role == "l2":
        return _l2_role(args)
    if args.role == "l3":
        return _l3_role(args)

    started = time.time()
    result = _coordinator_role(args)
    result.update(
        {
            "platform": args.platform,
            "runtime": args.runtime,
            "device_id": int(args.device),
            "elapsed_s": time.time() - started,
        }
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

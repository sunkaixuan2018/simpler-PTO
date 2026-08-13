#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate ordinary-allocation SVM visibility with and without a process fork."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import select
import signal
import time
from pathlib import Path
from typing import Any

from _task_interface import _memory_wmb_for_test  # pyright: ignore[reportMissingImports]
from simpler.task_interface import CallConfig, ChipCallable, ChipStorageTaskArgs, ChipWorker

from simpler_setup.kernel_compiler import KernelCompiler
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
    binary = compiler.compile_orchestration(
        runtime_name=runtime,
        source_path=str(_OBSERVER_SOURCE),
        extra_include_dirs=[str(compiler.project_root / "src" / "common")],
    )
    return ChipCallable.build(
        signature=[],
        func_name="pre_fork_svm_observer",
        binary=binary,
        children=[],
    )


def _observer_args(device_addr: int) -> ChipStorageTaskArgs:
    args = ChipStorageTaskArgs()
    args.add_scalar(int(device_addr))
    args.add_scalar(_EXPECTED_PAYLOAD)
    args.add_scalar(_EXPECTED_TAIL)
    return args


def _write_publication(host_addr: int) -> None:
    ctypes.c_uint64.from_address(host_addr + _PAYLOAD_OFFSET).value = _EXPECTED_PAYLOAD
    _memory_wmb_for_test()
    ctypes.c_uint32.from_address(host_addr + _TAIL_OFFSET).value = _EXPECTED_TAIL
    _memory_wmb_for_test()


def _completion(host_addr: int) -> int:
    return int(ctypes.c_uint32.from_address(host_addr + _COMPLETION_OFFSET).value)


def _completion_result(value: int) -> dict[str, Any]:
    if value == _EXPECTED_TAIL:
        return {"status": "pass", "reason": "AICPU observed the host publication and host observed completion"}
    if value == _TAIL_TIMEOUT:
        return {"status": "fail", "reason": "AICPU timed out while observing the host-written tail"}
    if value == _PAYLOAD_MISMATCH:
        return {"status": "fail", "reason": "AICPU observed the tail but not the host-written payload"}
    return {"status": "fail", "reason": f"unexpected completion value {value}"}


def _wait_status_description(wait_status: int) -> str:
    if os.WIFSIGNALED(wait_status):
        signal_number = os.WTERMSIG(wait_status)
        with contextlib.suppress(ValueError):
            return f"signal {signal_number} ({signal.Signals(signal_number).name})"
        return f"signal {signal_number}"
    if os.WIFEXITED(wait_status):
        return f"exit code {os.WEXITSTATUS(wait_status)}"
    return f"wait status {wait_status}"


def _read_child_result(fd: int, timeout_s: float) -> dict[str, Any]:
    ready, _, _ = select.select([fd], [], [], timeout_s)
    if not ready:
        return {"status": "fail", "reason": f"forked ChipWorker did not finish within {timeout_s:.0f}s"}
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    if not chunks:
        return {"status": "fail", "reason": "forked ChipWorker exited without a result"}
    return json.loads(b"".join(chunks).decode("utf-8"))


def _run_same_process(worker: ChipWorker, handle, args: ChipStorageTaskArgs, host_addr: int) -> dict[str, Any]:
    _write_publication(host_addr)
    config = CallConfig()
    config.aicpu_thread_num = 2
    worker.run(handle, args, config)
    return _completion_result(_completion(host_addr))


def _run_fork_inherited(
    worker: ChipWorker, handle, args: ChipStorageTaskArgs, host_addr: int, timeout_s: float
) -> dict[str, Any]:
    ready_read, ready_write = os.pipe()
    launch_read, launch_write = os.pipe()
    mapping_read, mapping_write = os.pipe()
    result_read, result_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        os.close(launch_write)
        os.close(mapping_read)
        os.close(result_read)
        child_result: dict[str, Any]
        try:
            os.write(ready_write, b"R")
            os.close(ready_write)
            if os.read(launch_read, 1) != b"G":
                raise RuntimeError("parent did not publish the launch signal")
            os.close(launch_read)
            child_payload = int(ctypes.c_uint64.from_address(host_addr + _PAYLOAD_OFFSET).value)
            child_tail = int(ctypes.c_uint32.from_address(host_addr + _TAIL_OFFSET).value)
            if child_payload != _EXPECTED_PAYLOAD or child_tail != _EXPECTED_TAIL:
                os.write(mapping_write, b"M")
                raise RuntimeError(
                    "forked process did not observe the inherited host mapping publication: "
                    f"payload={child_payload:#x} tail={child_tail}"
                )
            os.write(mapping_write, b"V")
            os.close(mapping_write)
            config = CallConfig()
            config.aicpu_thread_num = 2
            worker.run(handle, args, config)
            child_result = {
                "status": "pass",
                "reason": "forked process observed the host VA and its inherited ChipWorker run completed",
                "child_host_mapping_visible": True,
            }
        except BaseException as exc:  # noqa: BLE001
            child_result = {"status": "fail", "reason": f"forked ChipWorker run failed: {type(exc).__name__}: {exc}"}
        try:
            os.write(result_write, json.dumps(child_result).encode("utf-8"))
        finally:
            os.close(result_write)
            os._exit(0 if child_result["status"] == "pass" else 1)

    os.close(ready_write)
    os.close(launch_read)
    os.close(mapping_write)
    os.close(result_write)
    try:
        ready, _, _ = select.select([ready_read], [], [], 10.0)
        if not ready or os.read(ready_read, 1) != b"R":
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            return {"status": "fail", "reason": "forked ChipWorker did not reach its launch point"}
        _write_publication(host_addr)
        os.write(launch_write, b"G")
        os.close(launch_write)
        mapping_ready, _, _ = select.select([mapping_read], [], [], 10.0)
        if not mapping_ready:
            os.kill(pid, signal.SIGKILL)
            _, wait_status = os.waitpid(pid, 0)
            return {
                "status": "fail",
                "reason": "forked process did not finish its inherited host VA read within 10s",
                "child_status": _wait_status_description(wait_status),
                "child_host_mapping_visible": False,
            }
        mapping_status = os.read(mapping_read, 1)
        if mapping_status != b"V":
            child_result = _read_child_result(result_read, 5.0)
            _, wait_status = os.waitpid(pid, 0)
            if not mapping_status:
                child_result["reason"] = (
                    f"forked process died while reading the inherited host VA: {_wait_status_description(wait_status)}"
                )
            child_result["child_status"] = _wait_status_description(wait_status)
            child_result["child_host_mapping_visible"] = False
            return child_result
        child_result = _read_child_result(result_read, timeout_s)
        if child_result["status"] != "pass":
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        _, wait_status = os.waitpid(pid, 0)
        if child_result["status"] != "pass":
            return child_result
        if not os.WIFEXITED(wait_status) or os.WEXITSTATUS(wait_status) != 0:
            return {"status": "fail", "reason": f"forked ChipWorker exit status {wait_status}"}
        result = _completion_result(_completion(host_addr))
        result["child_host_mapping_visible"] = bool(child_result["child_host_mapping_visible"])
        return result
    finally:
        os.close(ready_read)
        os.close(mapping_read)
        with contextlib.suppress(OSError):
            os.close(launch_write)
        os.close(result_read)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    observer = _build_observer(args.platform, args.runtime)
    binaries = RuntimeBuilder(args.platform).get_binaries(args.runtime)
    worker = ChipWorker()
    handle = worker.register_callable(observer)
    device_addr = 0
    mapping: _HalHostMapping | None = None
    try:
        worker.init(int(args.device), binaries)
        device_addr = worker.malloc(_REGION_BYTES)
        zeros = ctypes.create_string_buffer(_REGION_BYTES)
        worker.copy_to(device_addr, ctypes.addressof(zeros), _REGION_BYTES)
        mapping = _HalHostMapping(device_addr, _REGION_BYTES, int(args.device))
        observer_args = _observer_args(device_addr)
        if args.case == "same-process-owner":
            result = _run_same_process(worker, handle, observer_args, mapping.host_addr)
        else:
            result = _run_fork_inherited(worker, handle, observer_args, mapping.host_addr, float(args.timeout))
        result.update(
            {
                "case": args.case,
                "device_addr": device_addr,
                "host_addr": mapping.host_addr,
                "identity_mapping": device_addr == mapping.host_addr,
                "completion": _completion(mapping.host_addr),
            }
        )
        return result
    finally:
        if mapping is not None:
            mapping.close()
        if device_addr:
            worker.free(device_addr)
        worker.unregister_callable(handle)
        worker.finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="a2a3", choices=("a2a3",))
    parser.add_argument("--runtime", default="tensormap_and_ringbuffer")
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--case", required=True, choices=("same-process-owner", "fork-inherited-owner"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output")
    cli_args = parser.parse_args()

    started = time.time()
    try:
        result = _run(cli_args)
    except BaseException as exc:  # noqa: BLE001
        result = {"status": "fail", "reason": f"{type(exc).__name__}: {exc}", "case": cli_args.case}
    result.update(
        {
            "platform": cli_args.platform,
            "runtime": cli_args.runtime,
            "device_id": int(cli_args.device),
            "elapsed_s": time.time() - started,
        }
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if cli_args.output:
        output = Path(cli_args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

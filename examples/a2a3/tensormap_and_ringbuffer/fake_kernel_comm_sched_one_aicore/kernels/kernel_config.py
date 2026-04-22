"""
Single-cluster gather communication scheduling case.

This case keeps the old fake_kernel_comm_sched_one_aicore benchmark shape but
uses the current tensormap_and_ringbuffer orchestration API. Runtime scheduling
is constrained to one AICore cluster by block_dim=1; each communication task is
submitted as a single-AIV task.
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _KERNELS_ROOT.parents[4]
_DEPS_ROOT = _PROJECT_ROOT / "examples" / "scripts" / "_deps"
_COMM_ISA_ROOT = _DEPS_ROOT / "pto-comm-isa"

_platform = os.environ.get("PTO_PLATFORM", "a2a3")
if _platform != "a2a3":
    raise RuntimeError("fake_kernel_comm_sched_one_aicore requires PTO_PLATFORM=a2a3")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value, 0)


def _strategy_code() -> int:
    value = os.environ.get("GATHER_STRATEGY", "hybrid").strip().lower()
    if value in ("hybrid", "auto", "0"):
        return 0
    if value in ("mte", "sync", "tgather", "1"):
        return 1
    if value in ("sdma", "async", "tget", "2"):
        return 2
    raise ValueError(f"Unsupported GATHER_STRATEGY={value!r}")


ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "fake_kernel_comm_sched_one_aicore_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "WindowMemCopyIn", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_in.cpp"), "core_type": "aiv"},
    {"func_id": 1, "name": "GatherSync", "source": str(_KERNELS_ROOT / "aiv" / "gather_sync_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 2, "name": "GatherAsync", "source": str(_KERNELS_ROOT / "aiv" / "gather_async_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "WindowMemCopyOut", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_out.cpp"), "core_type": "aiv"},
    {"func_id": 4, "name": "CommBarrier", "source": str(_KERNELS_ROOT / "aiv" / "comm_barrier_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 5, "name": "DummyCommSync", "source": str(_KERNELS_ROOT / "aiv" / "dummy_remote_read_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 6, "name": "DummyCommAsync", "source": str(_KERNELS_ROOT / "aiv" / "dummy_remote_read_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 7, "name": "DummyWindowFill", "source": str(_KERNELS_ROOT / "aiv" / "dummy_window_fill_kernel.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 1,
    "rounds": 1,
}

RUNTIME_ENV = {
    "PTO2_ENABLE_SDMA": "1",
}

_NRANKS = _env_int("PTO_NRANKS", 4)
_ROOT = _env_int("PTO_ROOT_RANK", 0)
_GATHER_COUNT = _env_int("GATHER_COUNT", 256)
_N_ITER = _env_int("N_ITER", 200)
_STRATEGY = _strategy_code()
_SERIALIZE_DUMMY = _env_int("EXTREME_SERIALIZE_DUMMY", 0)
_DUMMY_COMM_BYTES = _env_int("DUMMY_COMM_BYTES", 16 * 1024 * 1024)
_DUMMY_SOURCE_ELEMS = _env_int("DUMMY_SOURCE_ELEMS", (1 * 1024 * 1024) // 4)
_DUMMY_BUFFER_ELEMS = _env_int("DUMMY_BUFFER_ELEMS", (2 * 1024 * 1024) // 4)

if _NRANKS <= 1:
    raise ValueError("PTO_NRANKS must be greater than 1")
if _ROOT < 0 or _ROOT >= _NRANKS:
    raise ValueError(f"PTO_ROOT_RANK must be in [0, {_NRANKS}), got {_ROOT}")
if _GATHER_COUNT <= 0:
    raise ValueError("GATHER_COUNT must be positive")
if _N_ITER <= 0:
    raise ValueError("N_ITER must be positive")

DISTRIBUTED_CONFIG = {
    "nranks": _NRANKS,
    "root": _ROOT,
    "win_sync_prefix": 256,
    "pto_isa_root": str(_COMM_ISA_ROOT),
    "buffers": [
        {"name": "src", "dtype": "float32", "count": _GATHER_COUNT, "placement": "device"},
        {"name": "out", "dtype": "float32", "count": _NRANKS * _GATHER_COUNT, "placement": "device"},
        {"name": "win_src", "dtype": "float32", "count": _GATHER_COUNT, "placement": "window"},
        {"name": "win_dst", "dtype": "float32", "count": _NRANKS * _GATHER_COUNT, "placement": "window"},
        {"name": "dummy_src", "dtype": "float32", "count": _DUMMY_SOURCE_ELEMS, "placement": "window"},
        {"name": "dummy_dst", "dtype": "float32", "count": _DUMMY_BUFFER_ELEMS, "placement": "device"},
        {"name": "debug_poll_counts", "dtype": "int32", "count": _N_ITER * _NRANKS, "placement": "device"},
        {"name": "config", "dtype": "int64", "count": 16, "placement": "device"},
        {"name": "comm_barrier", "dtype": "int32", "count": _NRANKS * (_N_ITER + 1), "placement": "window"},
    ],
    "inputs": [
        "src",
        "out",
        "win_src",
        "win_dst",
        "dummy_src",
        "debug_poll_counts",
        "config",
        "comm_barrier",
    ],
    "outputs": ["out", "debug_poll_counts"],
    "args": [
        "src",
        "out",
        "win_src",
        "win_dst",
        "dummy_src",
        "dummy_dst",
        "debug_poll_counts",
        "comm_barrier",
        "config",
        "deviceCtx",
    ],
}

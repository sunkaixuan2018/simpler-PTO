"""
fake_kernel_comm_sched_trueload:
One-AICore dual-AIV contention case with an explicit true-load foreground flow.

Design:
1) Keep the same one-AICore scheduler constraints as the existing one_aicore case.
2) Replace the main gather kernels with explicit remote copy kernels:
   - MTE path: TLOAD/TSTORE
   - SDMA path: TGET_ASYNC
3) Keep the dummy background flow unchanged to preserve contention pressure.
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_BASE_CASE_ROOT = _KERNELS_ROOT.parents[1] / "fake_kernel_comm_sched" / "kernels"
_EXTREME_AIV_ROOT = _KERNELS_ROOT.parents[1] / "fake_kernel_comm_sched_extreme" / "kernels" / "aiv"

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "fake_kernel_comm_sched_trueload_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "WindowMemCopyIn",  "source": str(_BASE_CASE_ROOT / "aiv" / "window_memcopy_in.cpp"),  "core_type": "aiv"},
    {"func_id": 1, "name": "TrueLoadSync",     "source": str(_KERNELS_ROOT / "aiv" / "trueload_sync_kernel.cpp"),   "core_type": "aiv"},
    {"func_id": 2, "name": "TrueLoadAsync",    "source": str(_KERNELS_ROOT / "aiv" / "trueload_async_kernel.cpp"),  "core_type": "aiv"},
    {"func_id": 3, "name": "WindowMemCopyOut", "source": str(_BASE_CASE_ROOT / "aiv" / "window_memcopy_out.cpp"),  "core_type": "aiv"},
    {"func_id": 4, "name": "CommBarrier",      "source": str(_BASE_CASE_ROOT / "aiv" / "comm_barrier_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 5, "name": "DummyCommSync",    "source": str(_EXTREME_AIV_ROOT / "dummy_remote_read_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 6, "name": "DummyCommAsync",   "source": str(_EXTREME_AIV_ROOT / "dummy_remote_read_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 7, "name": "DummyWindowFill",  "source": str(_EXTREME_AIV_ROOT / "dummy_window_fill_kernel.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 2,
    "block_dim": 24,
    "n_devices": int(os.environ.get("N_DEVICES", "4")),
    "first_device_id": int(os.environ.get("FIRST_DEVICE", "0")),
    "requires_comm": True,
    "requires_sdma": True,
}

RUNTIME_ENV = {
    "PTO2_EXTREME_SINGLE_AICORE_DUAL_AIV": "1",
    "PTO2_ONE_AICORE_SPLIT_VECTOR_QUEUES": "1",
}

"""
fake_kernel_comm_sched_one_aicore_nodummy:
Single-AICore-domain gather benchmark without the extra dummy communication.

Design:
1) Keep the same one-AICore runtime constraints as fake_kernel_comm_sched_one_aicore.
2) Keep the same orchestration argument interface for easy A/B comparison.
3) Remove only the dummy communication task submission from the task graph.
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_BASE_CASE_ROOT = _KERNELS_ROOT.parents[1] / "fake_kernel_comm_sched" / "kernels"

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "fake_kernel_comm_sched_one_aicore_nodummy_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "WindowMemCopyIn",  "source": str(_BASE_CASE_ROOT / "aiv" / "window_memcopy_in.cpp"),  "core_type": "aiv"},
    {"func_id": 1, "name": "GatherSync",       "source": str(_BASE_CASE_ROOT / "aiv" / "gather_sync_kernel.cpp"),  "core_type": "aiv"},
    {"func_id": 2, "name": "GatherAsync",      "source": str(_BASE_CASE_ROOT / "aiv" / "gather_async_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "WindowMemCopyOut", "source": str(_BASE_CASE_ROOT / "aiv" / "window_memcopy_out.cpp"),  "core_type": "aiv"},
    {"func_id": 4, "name": "CommBarrier",      "source": str(_BASE_CASE_ROOT / "aiv" / "comm_barrier_kernel.cpp"), "core_type": "aiv"},
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

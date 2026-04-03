"""
fake_kernel_comm_sched_one_aicore:
Baseline gather case constrained to a single AICore domain.

Design:
1) Reuse the proven fake_kernel_comm_sched kernels and orchestration.
2) Force block_dim=1 so runtime exposes one AICore group (1 AIC + 2 AIV).
3) Enable same-AICore dual-AIV filtering in AICPU executor.
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_BASE_CASE_ROOT = _KERNELS_ROOT.parents[1] / "fake_kernel_comm_sched" / "kernels"

ORCHESTRATION = {
    "source": str(_BASE_CASE_ROOT / "orchestration" / "fake_kernel_comm_sched_orch.cpp"),
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
    # Dedicated orchestrator + scheduler thread pair.
    "aicpu_thread_num": 2,
    # One AICore group only (1 AIC + 2 AIV lanes).
    "block_dim": 1,
    "n_devices": int(os.environ.get("N_DEVICES", "4")),
    "first_device_id": int(os.environ.get("FIRST_DEVICE", "0")),
    "requires_comm": True,
    "requires_sdma": True,
}

# Reuse existing executor-side same-AICore dual-AIV selection logic.
RUNTIME_ENV = {
    "PTO2_EXTREME_SINGLE_AICORE_DUAL_AIV": "1",
}


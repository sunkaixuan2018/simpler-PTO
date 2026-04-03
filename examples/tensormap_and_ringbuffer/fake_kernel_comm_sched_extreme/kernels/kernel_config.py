"""
fake_kernel_comm_sched_extreme: single-AICPU + dual-AIV-on-one-AICore stress case.

This case intentionally injects two concurrent communication tasks per iteration:
1) main gather task (used for benchmark stats)
2) dummy gather task (extra contention on the second AIV lane)
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_COMMON_AIV_ROOT = _KERNELS_ROOT.parents[1] / "fake_kernel_comm_sched" / "kernels" / "aiv"
_EXTREME_AIV_ROOT = _KERNELS_ROOT / "aiv"

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "fake_kernel_comm_sched_extreme_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "WindowMemCopyIn",   "source": str(_COMMON_AIV_ROOT / "window_memcopy_in.cpp"),   "core_type": "aiv"},
    {"func_id": 1, "name": "GatherSync",         "source": str(_COMMON_AIV_ROOT / "gather_sync_kernel.cpp"),   "core_type": "aiv"},
    {"func_id": 2, "name": "GatherAsync",        "source": str(_COMMON_AIV_ROOT / "gather_async_kernel.cpp"),  "core_type": "aiv"},
    {"func_id": 3, "name": "WindowMemCopyOut",   "source": str(_COMMON_AIV_ROOT / "window_memcopy_out.cpp"),   "core_type": "aiv"},
    {"func_id": 4, "name": "CommBarrier",        "source": str(_COMMON_AIV_ROOT / "comm_barrier_kernel.cpp"),  "core_type": "aiv"},
    # Dummy communication load: non-collective remote reads to avoid gather re-entrancy issues.
    {"func_id": 5, "name": "DummyCommSync",      "source": str(_EXTREME_AIV_ROOT / "dummy_remote_read_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 6, "name": "DummyCommAsync",     "source": str(_EXTREME_AIV_ROOT / "dummy_remote_read_kernel.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 1,
    "block_dim": 1,
    "n_devices": int(os.environ.get("N_DEVICES", "4")),
    "first_device_id": int(os.environ.get("FIRST_DEVICE", "0")),
    "requires_comm": True,
    "requires_sdma": True,
}

# Applied during compile/run by code_runner. Keep scoped to this extreme case only.
RUNTIME_ENV = {
    "PTO2_EXTREME_SINGLE_AICORE_DUAL_AIV": "1",
}

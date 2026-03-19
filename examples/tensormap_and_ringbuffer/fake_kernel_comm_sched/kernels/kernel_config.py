"""
fake_kernel_comm_sched: Scheduler-level variant selection.

Same as fake_kernel_comm but variant selection happens in the AICPU scheduler
(at dispatch time) instead of in orchestration (at submit time).
Orchestration submits a variant task; scheduler picks sync or async at dispatch.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "fake_kernel_comm_sched_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "WindowMemCopyIn",  "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_in.cpp"),  "core_type": "aiv"},
    {"func_id": 1, "name": "GatherSync",        "source": str(_KERNELS_ROOT / "aiv" / "gather_sync_kernel.cpp"),  "core_type": "aiv"},
    {"func_id": 2, "name": "GatherAsync",       "source": str(_KERNELS_ROOT / "aiv" / "gather_async_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "WindowMemCopyOut",  "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_out.cpp"),  "core_type": "aiv"},
    {"func_id": 4, "name": "CommBarrier",       "source": str(_KERNELS_ROOT / "aiv" / "comm_barrier_kernel.cpp"),  "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
    "n_devices": 2,
    "first_device_id": 0,
    "requires_comm": True,
    "requires_sdma": True,
}

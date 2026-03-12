"""
AllGather (TGATHER) for tensormap_and_ringbuffer runtime.

N sequential Gathers. Flow: WindowMemCopyIn -> for r in [0, n_ranks): Barrier_r
-> Gather(root=r) -> [if rank_id==r: WindowMemCopyOut] -> Barrier(post)
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "allgather_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "WindowMemCopyIn", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_in.cpp"), "core_type": "aiv"},
    {"func_id": 1, "name": "Gather", "source": str(_KERNELS_ROOT / "aiv" / "gather_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 2, "name": "WindowMemCopyOut", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_out.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "CommBarrierAll", "source": str(_KERNELS_ROOT / "aiv" / "comm_barrier_all_kernel.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
    "n_devices": 2,
    "first_device_id": 0,
    "requires_comm": True,
}

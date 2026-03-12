"""
Paged Attention + AllGather (TGATHER): Paged Attention → N sequential Gathers.

Flow per rank:
  QK → Softmax → PV → OnlineUpdate  (paged attention)
  → WindowMemCopyIn -> for r in [0,n_ranks): Barrier -> Gather(root=r)
  -> [if rank==r: WindowMemCopyOut] -> Barrier(post)
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "paged_attention_allgather_orch.cpp"),
    "function_name": "build_paged_attention_allgather_graph",
}

KERNELS = [
    {"func_id": 0, "name": "QK", "source": str(_KERNELS_ROOT / "aic" / "aic_qk_matmul.cpp"), "core_type": "aic"},
    {"func_id": 1, "name": "SF", "source": str(_KERNELS_ROOT / "aiv" / "aiv_softmax_prepare.cpp"), "core_type": "aiv"},
    {"func_id": 2, "name": "PV", "source": str(_KERNELS_ROOT / "aic" / "aic_pv_matmul.cpp"), "core_type": "aic"},
    {"func_id": 3, "name": "UP", "source": str(_KERNELS_ROOT / "aiv" / "aiv_online_update.cpp"), "core_type": "aiv"},
    {"func_id": 4, "name": "WindowMemCopyIn", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_in.cpp"), "core_type": "aiv"},
    {"func_id": 5, "name": "Gather", "source": str(_KERNELS_ROOT / "aiv" / "gather_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 6, "name": "WindowMemCopyOut", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_out.cpp"), "core_type": "aiv"},
    {"func_id": 7, "name": "CommBarrierAll", "source": str(_KERNELS_ROOT / "aiv" / "comm_barrier_all_kernel.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "host_build_graph",
    "n_devices": 2,
    "first_device_id": 0,
    "requires_comm": True,
    "aicpu_thread_num": 3,
    "block_dim": 3,
}

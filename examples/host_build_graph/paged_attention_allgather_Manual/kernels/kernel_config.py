"""
Paged Attention + AllGather (Manual): Paged Attention → AllGather.

Flow per rank:
  QK → Softmax → PV → OnlineUpdate  (paged attention, possibly multi-block)
  → WindowMemCopyIn → CommBarrier → AllGatherManual (direct RDMA reads)
  → WindowMemCopyOut → CommBarrier(post)

All ranks get the full allgather output (concatenation of all ranks' attn_out prefix).
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
    {"func_id": 5, "name": "AllGatherManual", "source": str(_KERNELS_ROOT / "aiv" / "allgather_manual_kernel.cpp"), "core_type": "aiv"},
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

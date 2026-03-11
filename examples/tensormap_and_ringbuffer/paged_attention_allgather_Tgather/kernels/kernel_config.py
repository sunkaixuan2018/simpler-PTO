"""
Paged Attention + AllGather (TGATHER) for tensormap_and_ringbuffer.

Flow: Paged Attention (QK->SF->PV->UP) -> WindowMemCopyIn -> for r in [0,n_ranks):
Barrier -> Gather(root=r) -> [rank r: WindowMemCopyOut] -> Barrier(post)
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_PA_ROOT = _KERNELS_ROOT.parent.parent / "paged_attention" / "kernels"
_AG_ROOT = _KERNELS_ROOT.parent.parent / "allgather_Tgather" / "kernels"

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "paged_attention_allgather_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "QK", "source": str(_PA_ROOT / "aic" / "aic_qk_matmul.cpp"), "core_type": "aic"},
    {"func_id": 1, "name": "SF", "source": str(_PA_ROOT / "aiv" / "aiv_softmax_prepare.cpp"), "core_type": "aiv"},
    {"func_id": 2, "name": "PV", "source": str(_PA_ROOT / "aic" / "aic_pv_matmul.cpp"), "core_type": "aic"},
    {"func_id": 3, "name": "UP", "source": str(_PA_ROOT / "aiv" / "aiv_online_update.cpp"), "core_type": "aiv"},
    {"func_id": 4, "name": "AIC_HUB", "source": str(_PA_ROOT / "aic" / "aic_hub.cpp"), "core_type": "aic"},
    {"func_id": 5, "name": "AIV_HUB", "source": str(_PA_ROOT / "aiv" / "aiv_hub.cpp"), "core_type": "aiv"},
    {"func_id": 6, "name": "WindowMemCopyIn", "source": str(_AG_ROOT / "aiv" / "window_memcopy_in.cpp"), "core_type": "aiv"},
    {"func_id": 7, "name": "Gather", "source": str(_AG_ROOT / "aiv" / "gather_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 8, "name": "WindowMemCopyOut", "source": str(_AG_ROOT / "aiv" / "window_memcopy_out.cpp"), "core_type": "aiv"},
    {"func_id": 9, "name": "CommBarrierAll", "source": str(_AG_ROOT / "aiv" / "comm_barrier_all_kernel.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
    "n_devices": 2,
    "first_device_id": 0,
    "requires_comm": True,
}

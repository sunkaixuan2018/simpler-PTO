"""
Gather (TGET_ASYNC) for tensormap_and_ringbuffer runtime.

Flow: WindowMemCopyIn -> CommBarrier -> GatherAsync (root only) -> WindowMemCopyOut (root only).
Uses SDMA async DMA engine for GM-to-GM direct transfer.
Requires HCCL (multi-card), PTO_COMM_ISA_ROOT for comm headers, SDMA workspace.
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "gather_async_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "WindowMemCopyIn", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_in.cpp"), "core_type": "aiv"},
    {"func_id": 1, "name": "GatherAsync", "source": str(_KERNELS_ROOT / "aiv" / "gather_async_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 2, "name": "WindowMemCopyOut", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_out.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "CommBarrier", "source": str(_KERNELS_ROOT / "aiv" / "comm_barrier_kernel.cpp"), "core_type": "aiv"},
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

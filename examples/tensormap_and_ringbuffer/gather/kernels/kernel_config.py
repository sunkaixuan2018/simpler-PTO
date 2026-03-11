"""
Gather operator for tensormap_and_ringbuffer runtime.

Multi-card TGATHER communication, no computation.
Flow: WindowMemCopyIn -> CommBarrier -> TGATHER -> WindowMemCopyOut (root only).

Adapted from host_build_graph/gather with:
- aicpu_orchestration_entry (device-side orchestration)
- AIV_HUB for tensormap_and_ringbuffer runtime
- Kernels use TensorData interface (buffer.addr)
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "gather_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "WindowMemCopyIn", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_in.cpp"), "core_type": "aiv"},
    {"func_id": 1, "name": "Gather", "source": str(_KERNELS_ROOT / "aiv" / "gather_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 2, "name": "WindowMemCopyOut", "source": str(_KERNELS_ROOT / "aiv" / "window_memcopy_out.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "CommBarrier", "source": str(_KERNELS_ROOT / "aiv" / "comm_barrier_kernel.cpp"), "core_type": "aiv"},
    {"func_id": 5, "name": "ZeroBuffer", "source": str(_KERNELS_ROOT / "aiv" / "zero_buffer.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
    "n_devices": 2,
    "first_device_id": 0,
    "requires_comm": True,
}

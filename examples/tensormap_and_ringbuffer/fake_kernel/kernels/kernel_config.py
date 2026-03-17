"""
fake_kernel: Two interchangeable add kernels with runtime selection.

Both kernels compute out = a + b with identical interfaces but different
internal implementations. The orchestration dynamically selects which
kernel to dispatch based on a variant_id argument (or random if -1).
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "fake_kernel_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "AddV1", "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_v1.cpp"), "core_type": "aiv"},
    {"func_id": 1, "name": "AddV2", "source": str(_KERNELS_ROOT / "aiv" / "kernel_add_v2.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 3,
}

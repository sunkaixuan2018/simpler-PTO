"""
comm_gather: 2-card collective TGATHER only (no compute kernel).
Uses HCCL; run with run_example.py --n-ranks 2 --first-device 0.
Executable is built by Makefile in kernels/host/ (CANN env on a2a3).
"""

from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent

# No orchestration; host executable runs ForkAndRunWithHcclRootInfo internally.
ORCHESTRATION = None

# No kernels compiled by Python; device kernel is in run_gather_kernel.cpp, built by Makefile.
KERNELS = []

RUNTIME_CONFIG = {
    "runtime": "comm_gather",
    "n_ranks": 2,
    "first_device_id": 0,
}

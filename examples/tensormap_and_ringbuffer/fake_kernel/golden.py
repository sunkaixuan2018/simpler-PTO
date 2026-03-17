"""
Golden script for fake_kernel example.

Both kernel variants compute: out = a + b
The variant_id controls which kernel is selected at runtime:
  -1 = random (default), 0 = AddV1, 1 = AddV2

Args layout: [a, b, out, size_a, size_b, size_out, SIZE, variant_id]
"""

import ctypes
import torch

__outputs__ = ["out"]

RTOL = 1e-5
ATOL = 1e-5


def generate_inputs(params: dict) -> list:
    ROWS = 128
    COLS = 128
    SIZE = ROWS * COLS

    a = torch.randn(SIZE, dtype=torch.float32)
    b = torch.randn(SIZE, dtype=torch.float32)
    out = torch.zeros(SIZE, dtype=torch.float32)

    variant_id = params.get("variant_id", -1)

    return [
        ("a", a),
        ("b", b),
        ("out", out),
        ("size_a", ctypes.c_int64(a.nbytes)),
        ("size_b", ctypes.c_int64(b.nbytes)),
        ("size_out", ctypes.c_int64(out.nbytes)),
        ("SIZE", ctypes.c_int64(SIZE)),
        ("variant_id", ctypes.c_int64(variant_id)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    tensors["out"][:] = tensors["a"] + tensors["b"]


ALL_CASES = {
    "Random": {"variant_id": -1},
    "V1": {"variant_id": 0},
    "V2": {"variant_id": 1},
}
DEFAULT_CASE = "Random"

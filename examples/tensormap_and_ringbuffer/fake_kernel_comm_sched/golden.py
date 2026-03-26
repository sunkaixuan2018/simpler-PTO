"""
Golden reference for fake_kernel_comm_sched: scheduler-level variant selection.

Variant selection is transparent — the scheduler internally picks sync or async
gather at dispatch time, or a forced strategy is passed via args[11].

Args layout: [src, out, size_src, size_out, device_ctx_ptr, win_in_base,
              win_out_base, n_ranks, root, rank_id, sdma_workspace_ptr, strategy]

Environment variables:
    GATHER_COUNT    Elements per rank (default 256). Controls tensor sizes.
    GATHER_STRATEGY Gather strategy: hybrid (default), mte, sdma.
                    Encoded as int and passed as args[11] to orchestration.
"""

import ctypes
import os

import numpy as np

GATHER_COUNT = int(os.environ.get("GATHER_COUNT", "256"))

# Strategy encoding: 0=hybrid, 1=mte/sync, 2=sdma/async
_STRATEGY_MAP = {"hybrid": 0, "mte": 1, "sdma": 2}

__outputs__ = ["out"]
RTOL = 1e-4
ATOL = 1e-4


def generate_inputs(params: dict) -> list:
    rank_id = params.get("rank_id", 0)
    n_ranks = params.get("n_ranks", 4)
    root = params.get("root", 0)

    np.random.seed(42 + rank_id)
    src = np.random.randn(GATHER_COUNT).astype(np.float32) * 0.1
    out = np.zeros((n_ranks * GATHER_COUNT,), dtype=np.float32)

    strategy_str = os.environ.get("GATHER_STRATEGY", "hybrid")
    strategy_int = _STRATEGY_MAP.get(strategy_str, 0)

    result = [
        ("src", src),
        ("out", out),
        ("size_src", ctypes.c_int64(src.nbytes)),
        ("size_out", ctypes.c_int64(out.nbytes)),
    ]

    if "device_ctx_ptr" in params and "win_in_base" in params and "win_out_base" in params:
        result.extend([
            ("device_ctx_ptr", ctypes.c_uint64(params["device_ctx_ptr"])),
            ("win_in_base", ctypes.c_uint64(params["win_in_base"])),
            ("win_out_base", ctypes.c_uint64(params["win_out_base"])),
            ("n_ranks", ctypes.c_int32(n_ranks)),
            ("root", ctypes.c_int32(root)),
            ("rank_id", ctypes.c_int32(rank_id)),
            ("sdma_workspace_ptr", ctypes.c_uint64(params.get("sdma_workspace_ptr", 0))),
            ("strategy", ctypes.c_int32(strategy_int)),
        ])

    return result


def compute_golden(tensors: dict, params: dict) -> None:
    rank_id = params.get("rank_id", 0)
    n_ranks = params.get("n_ranks", 4)
    root = params.get("root", 0)

    out = tensors["out"]

    if rank_id == root:
        out_np = out.cpu().numpy()
        for r in range(n_ranks):
            np.random.seed(42 + r)
            src_r = np.random.randn(GATHER_COUNT).astype(np.float32) * 0.1
            out_np[r * GATHER_COUNT : (r + 1) * GATHER_COUNT] = src_r[:GATHER_COUNT]

"""
Golden reference for fake_kernel_comm_sched: scheduler-level variant selection.

Variant selection is transparent — the scheduler internally picks sync or async
gather at dispatch time, or a forced strategy is passed via args[11].

Args layout: [src, out, size_src, size_out, device_ctx_ptr, win_in_base,
              win_out_base, n_ranks, root, rank_id, sdma_workspace_ptr,
              strategy, debug_poll_counts]

Environment variables:
    GATHER_COUNT    Elements per rank (default 256). Controls tensor sizes.
    GATHER_STRATEGY Gather strategy: hybrid (default), mte, sdma.
                    Encoded as int and passed as args[11] to orchestration.
"""

import ctypes
import json
import os
from pathlib import Path

import numpy as np

GATHER_COUNT = int(os.environ.get("GATHER_COUNT", "256"))
_N_ITER = int(os.environ.get("N_ITER", "100"))

# Strategy encoding: 0=hybrid, 1=mte/sync, 2=sdma/async
_STRATEGY_MAP = {"hybrid": 0, "mte": 1, "sdma": 2}

__outputs__ = ["out", "debug_poll_counts"]
RTOL = 1e-4
ATOL = 1e-4

_OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "outputs"


def generate_inputs(params: dict) -> list:
    rank_id = params.get("rank_id", 0)
    n_ranks = params.get("n_ranks", 4)
    root = params.get("root", 0)

    np.random.seed(42 + rank_id)
    src = np.random.randn(GATHER_COUNT).astype(np.float32) * 0.1
    out = np.zeros((n_ranks * GATHER_COUNT,), dtype=np.float32)

    strategy_str = os.environ.get("GATHER_STRATEGY", "hybrid")
    strategy_int = _STRATEGY_MAP.get(strategy_str, 0)

    # Debug tensor: SDMA poll counts, flat [N_ITER * n_ranks] (one row per iteration)
    debug_poll_counts = np.zeros((_N_ITER * n_ranks,), dtype=np.int32)

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
            ("debug_poll_counts", debug_poll_counts),
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

    # Keep debug output unconstrained in golden to avoid introducing
    # a fake expected value before device execution.


def post_run_collect(outputs: dict, params: dict) -> None:
    """
    Optional runtime hook called by CodeRunner after runtime.finalize().
    Writes real debug poll counts from device outputs for bench analysis.
    """
    rank_id = params.get("rank_id", 0)
    n_ranks = params.get("n_ranks", 4)
    root = params.get("root", 0)

    debug = outputs.get("debug_poll_counts")
    if debug is None or rank_id != root:
        return

    poll_np = debug.cpu().numpy().reshape(_N_ITER, n_ranks)
    strategy_str = os.environ.get("GATHER_STRATEGY", "hybrid")
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = _OUTPUTS_DIR / f"poll_counts_{strategy_str}_gc{GATHER_COUNT}_r{n_ranks}.json"
    with open(fname, "w") as f:
        json.dump({
            "strategy": strategy_str,
            "gather_count": GATHER_COUNT,
            "n_ranks": n_ranks,
            "n_iter": _N_ITER,
            "poll_counts": poll_np.tolist(),
        }, f)

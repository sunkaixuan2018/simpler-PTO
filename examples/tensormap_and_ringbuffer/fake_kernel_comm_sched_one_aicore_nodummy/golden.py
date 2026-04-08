"""
Golden reference for fake_kernel_comm_sched_one_aicore_nodummy.

This keeps the same input interface as fake_kernel_comm_sched_one_aicore so
the two cases can be benchmarked side-by-side with identical host-side
arguments. The orchestration ignores the dummy-task-related args.
"""

import ctypes
import json
import math
import os
from pathlib import Path

import numpy as np

GATHER_COUNT = int(os.environ.get("GATHER_COUNT", "256"))
_N_ITER = int(os.environ.get("N_ITER", "200"))
_SERIALIZE_DUMMY = int(os.environ.get("EXTREME_SERIALIZE_DUMMY", "0"))
_DUMMY_COMM_BYTES = int(os.environ.get("DUMMY_COMM_BYTES", str(512 * 1024)))
_DUMMY_BYTES_PER_REPEAT = max(1, GATHER_COUNT * np.dtype(np.float32).itemsize)
_DUMMY_COMM_SCALE = max(1, int(math.ceil(_DUMMY_COMM_BYTES / _DUMMY_BYTES_PER_REPEAT)))
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
    debug_poll_counts = np.zeros((_N_ITER * n_ranks,), dtype=np.int32)

    result = [
        ("src", src),
        ("out", out),
        ("size_src", ctypes.c_int64(src.nbytes)),
        ("size_out", ctypes.c_int64(out.nbytes)),
    ]

    if "device_ctx_ptr" in params and "win_in_base" in params and "win_out_base" in params:
        result.extend(
            [
                ("device_ctx_ptr", ctypes.c_uint64(params["device_ctx_ptr"])),
                ("win_in_base", ctypes.c_uint64(params["win_in_base"])),
                ("win_out_base", ctypes.c_uint64(params["win_out_base"])),
                ("n_ranks", ctypes.c_int32(n_ranks)),
                ("root", ctypes.c_int32(root)),
                ("rank_id", ctypes.c_int32(rank_id)),
                ("sdma_workspace_ptr", ctypes.c_uint64(params.get("sdma_workspace_ptr", 0))),
                ("strategy", ctypes.c_int32(strategy_int)),
                ("debug_poll_counts", debug_poll_counts),
                ("n_iter", ctypes.c_int32(_N_ITER)),
                ("serialize_dummy", ctypes.c_int32(_SERIALIZE_DUMMY)),
                ("dummy_comm_scale", ctypes.c_int32(_DUMMY_COMM_SCALE)),
            ]
        )

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


def post_run_collect(outputs: dict, params: dict) -> None:
    rank_id = params.get("rank_id", 0)
    n_ranks = params.get("n_ranks", 4)
    root = params.get("root", 0)

    debug = outputs.get("debug_poll_counts")
    if debug is None or rank_id != root:
        return

    poll_np = debug.cpu().numpy().reshape(_N_ITER, n_ranks)
    strategy_str = os.environ.get("GATHER_STRATEGY", "hybrid")
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = _OUTPUTS_DIR / f"poll_counts_one_aicore_nodummy_{strategy_str}_gc{GATHER_COUNT}_r{n_ranks}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(
            {
                "strategy": strategy_str,
                "case": "one_aicore_nodummy",
                "gather_count": GATHER_COUNT,
                "n_ranks": n_ranks,
                "n_iter": _N_ITER,
                "serialize_dummy_arg": _SERIALIZE_DUMMY,
                "dummy_comm_bytes_arg": _DUMMY_COMM_BYTES,
                "dummy_comm_scale_arg": _DUMMY_COMM_SCALE,
                "poll_counts": poll_np.tolist(),
            },
            f,
        )

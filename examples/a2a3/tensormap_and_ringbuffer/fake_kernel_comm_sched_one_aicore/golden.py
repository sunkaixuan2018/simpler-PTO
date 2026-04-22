"""
Golden data for fake_kernel_comm_sched_one_aicore.

Every rank contributes one contiguous float32 slice. The kernel writes the
all-gathered result to every rank's out buffer so the distributed verifier can
use a single expected tensor for all ranks.
"""

import os
import struct
from pathlib import Path

__outputs__ = ["out", "debug_poll_counts"]

RTOL = 1e-5
ATOL = 1e-5


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value, 0)


def _strategy_code() -> int:
    value = os.environ.get("GATHER_STRATEGY", "hybrid").strip().lower()
    if value in ("hybrid", "auto", "0"):
        return 0
    if value in ("mte", "sync", "tgather", "1"):
        return 1
    if value in ("sdma", "async", "tget", "2"):
        return 2
    raise ValueError(f"Unsupported GATHER_STRATEGY={value!r}")


def _rank_src(rank: int, count: int) -> list[float]:
    base = float(rank * 1024)
    return [base + float(i % 1024) / 1024.0 for i in range(count)]


def _read_f32(path: Path) -> list[float]:
    raw = path.read_bytes()
    count = len(raw) // 4
    return list(struct.unpack(f"<{count}f", raw))


def generate_distributed_inputs(rank: int, nranks: int, root: int, comm_ctx=None) -> list:
    del comm_ctx

    gather_count = _env_int("GATHER_COUNT", 256)
    n_iter = _env_int("N_ITER", 200)
    dummy_source_elems = _env_int("DUMMY_SOURCE_ELEMS", (1 * 1024 * 1024) // 4)
    dummy_buffer_elems = _env_int("DUMMY_BUFFER_ELEMS", (2 * 1024 * 1024) // 4)
    dummy_comm_bytes = _env_int("DUMMY_COMM_BYTES", 16 * 1024 * 1024)
    serialize_dummy = _env_int("EXTREME_SERIALIZE_DUMMY", 0)

    if nranks <= 1:
        raise ValueError(f"fake_kernel_comm_sched_one_aicore expects nranks > 1, got {nranks}")

    src = _rank_src(rank, gather_count)
    out = [0.0] * (nranks * gather_count)
    debug = [0] * (n_iter * nranks)
    config = [
        gather_count,
        n_iter,
        _strategy_code(),
        serialize_dummy,
        dummy_comm_bytes,
        dummy_source_elems,
        dummy_buffer_elems,
        root,
        nranks,
        rank,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    return [
        ("src", src),
        ("out", out),
        ("win_src", [0.0] * gather_count),
        ("win_dst", [0.0] * (nranks * gather_count)),
        ("dummy_src", [0.0] * dummy_source_elems),
        ("debug_poll_counts", debug),
        ("config", config),
        ("comm_barrier", [0] * (nranks * (n_iter + 1))),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    artifact_dir = Path(params["artifact_dir"])
    nranks = int(params["nranks"])
    if nranks <= 1:
        raise ValueError(f"fake_kernel_comm_sched_one_aicore expects nranks > 1, got {nranks}")

    gathered = []
    for rank in range(nranks):
        gathered.extend(_read_f32(artifact_dir / f"rank_{rank}" / "src.bin"))

    tensors["out"][:] = gathered
    tensors["debug_poll_counts"][:] = [1] * len(tensors["debug_poll_counts"])

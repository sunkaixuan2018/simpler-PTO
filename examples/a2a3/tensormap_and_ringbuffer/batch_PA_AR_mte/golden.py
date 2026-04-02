"""
Golden script for batch_PA_AR_distributed_v1.

Each rank builds its own batch paged attention inputs, computes a local PA
result, notifies all peers that local_out is ready, directly reads all ranks'
local_out from remote window memory, and finally writes the allreduce sum:

    out = sum(local_pa_out_rank_r for r in ranks)
"""

import struct
from pathlib import Path

import torch

from paged_attention_golden import generate_inputs as _generate_pa_inputs
from paged_attention_golden import paged_attention

__outputs__ = ["out"]

RTOL = 1e-2
ATOL = 1e-2

PA_CASE = {
    "batch": 32,
    "num_heads": 16,
    "kv_head_num": 1,
    "head_dim": 16,
    "block_size": 16,
    "context_len": 33,
    "context_lens_list": [33, 17, 29, 12, 25, 8, 31, 16],
    "max_model_len": 256,
    "dtype": "float16",
}

_DTYPE_INFO = {
    "float16": ("e", 2, torch.float16),
    "float32": ("f", 4, torch.float32),
    "int32": ("i", 4, torch.int32),
    "int64": ("q", 8, torch.int64),
}


def _to_python_list(tensor: torch.Tensor):
    return tensor.detach().cpu().reshape(-1).tolist()


def _read_binary(path: Path, dtype: str) -> torch.Tensor:
    fmt_char, elem_size, torch_dtype = _DTYPE_INFO[dtype]
    raw = path.read_bytes()
    count = len(raw) // elem_size
    values = struct.unpack(f"<{count}{fmt_char}", raw)
    return torch.tensor(values, dtype=torch_dtype)


def _build_rank_inputs(rank: int, nranks: int):
    with torch.random.fork_rng():
        torch.manual_seed(2026 + rank)
        pa_inputs = dict(_generate_pa_inputs(PA_CASE, return_all_sizes=False))

    query = pa_inputs["query"]
    key_cache = pa_inputs["key_cache"]
    value_cache = pa_inputs["value_cache"]
    block_table = pa_inputs["block_table"]
    context_lens = pa_inputs["context_lens"]

    batch = PA_CASE["batch"]
    num_heads = PA_CASE["num_heads"]
    head_dim = PA_CASE["head_dim"]
    block_size = PA_CASE["block_size"]
    kv_head_num = PA_CASE["kv_head_num"]
    max_blocks = PA_CASE["max_model_len"] // block_size
    total_blocks = key_cache.numel() // (block_size * kv_head_num * head_dim)

    config = torch.tensor(
        [
            batch,
            num_heads,
            kv_head_num,
            head_dim,
            block_size,
            max_blocks,
            struct.unpack("I", struct.pack("f", 1.0))[0],
            total_blocks,
        ],
        dtype=torch.int64,
    )

    out_elems = batch * num_heads * head_dim
    num_chunks = (batch + 16 - 1) // 16
    zeros = torch.zeros(out_elems, dtype=torch.float32)
    return [
        ("query", _to_python_list(query.to(torch.float16))),
        ("key_cache", _to_python_list(key_cache.to(torch.float16))),
        ("value_cache", _to_python_list(value_cache.to(torch.float16))),
        ("block_table", _to_python_list(block_table.to(torch.int32))),
        ("context_lens", _to_python_list(context_lens.to(torch.int32))),
        ("local_out", _to_python_list(zeros)),
        ("out", _to_python_list(zeros)),
        ("config", _to_python_list(config)),
        ("notify_counter", [0] * num_chunks),
        ("comm_barrier", [0] * nranks),
    ]


def generate_distributed_inputs(rank: int, nranks: int, root: int, comm_ctx=None) -> list:
    del root
    del comm_ctx
    if nranks <= 1:
        raise ValueError(f"batch_PA_AR_distributed_v1 expects nranks > 1, got {nranks}")
    return _build_rank_inputs(rank, nranks)


def compute_golden(tensors: dict, params: dict) -> None:
    artifact_dir = Path(params["artifact_dir"])
    nranks = int(params["nranks"])
    if nranks <= 1:
        raise ValueError(f"batch_PA_AR_distributed_v1 expects nranks > 1, got {nranks}")

    reduced = None
    for rank in range(nranks):
        rank_dir = artifact_dir / f"rank_{rank}"
        config = _read_binary(rank_dir / "config.bin", "int64")
        batch = int(config[0].item())
        num_heads = int(config[1].item())
        kv_head_num = int(config[2].item())
        head_dim = int(config[3].item())
        block_size = int(config[4].item())
        block_num = int(config[5].item())
        scale_bits = int(config[6].item())
        total_blocks = int(config[7].item())

        scale_value = struct.unpack("f", struct.pack("I", scale_bits))[0]
        query = _read_binary(rank_dir / "query.bin", "float16").reshape(batch, num_heads, head_dim)
        key_cache = _read_binary(rank_dir / "key_cache.bin", "float16").reshape(
            total_blocks, block_size, kv_head_num, head_dim
        )
        value_cache = _read_binary(rank_dir / "value_cache.bin", "float16").reshape(
            total_blocks, block_size, kv_head_num, head_dim
        )
        block_table = _read_binary(rank_dir / "block_table.bin", "int32").reshape(batch, block_num)
        context_lens = _read_binary(rank_dir / "context_lens.bin", "int32")

        local_out = paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=kv_head_num,
            num_heads=num_heads,
            scale_value=scale_value,
            block_table=block_table,
            context_lens=context_lens,
        ).reshape(-1)

        if reduced is None:
            reduced = local_out.to(torch.float32)
        else:
            reduced = reduced + local_out.to(torch.float32)

    tensors["out"][:] = reduced.tolist()
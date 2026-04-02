"""batch_PA_AR_smda — 多卡：各卡先做 PA，再按 chunk 流水通信并对 `out` 做跨卡求和（allreduce sum）。"""

import ctypes
from pathlib import Path

import torch

from paged_attention_golden import compute_golden as pa_compute_golden, generate_inputs as pa_generate_inputs, paged_attention

__outputs__ = ["out"]

RTOL = 2e-2
ATOL = 2e-2

# 与 kernel_config.py 中的 Case1 保持一致；batch > 16 用于观察 chunk 流水。
PA_CASE = {
    "batch": 32,
    "num_heads": 16,
    "kv_head_num": 1,
    "head_dim": 16,
    "block_size": 16,
    "context_len": 33,
    "context_lens_list": [33, 17, 29, 12, 31, 15, 27, 18, 25, 9, 33, 16, 21, 14, 28, 11, 30, 13, 24, 10],
    "max_model_len": 256,
    "dtype": "float16",
}
IN_CORE_BATCH = 16
NOTIFY_SLOTS = (PA_CASE["batch"] + IN_CORE_BATCH - 1) // IN_CORE_BATCH


def _tensor_to_host_list(tensor: torch.Tensor) -> list:
    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.float16:
        return [float(x) for x in t.view(-1).tolist()]
    if t.dtype in (torch.int32, torch.int64):
        return [int(x) for x in t.view(-1).tolist()]
    if t.dtype == torch.float32:
        return [float(x) for x in t.view(-1).tolist()]
    raise TypeError(f"unsupported dtype {t.dtype}")


def generate_inputs(params: dict) -> list:
    """单卡 / 仿真：与 batch_paged_attention 相同的 10 个实参布局。"""
    del params
    return pa_generate_inputs({**PA_CASE, "name": "Case1"}, return_all_sizes=False)


def generate_distributed_inputs(rank: int, nranks: int, root: int, comm_ctx=None) -> list:
    del root
    del comm_ctx

    torch.manual_seed(1000 + rank)
    items = pa_generate_inputs({**PA_CASE, "name": "Case1"}, return_all_sizes=False)
    tensors = {name: t for name, t in items if isinstance(t, torch.Tensor)}
    key_nbytes = None
    for name, v in items:
        if name == "size_key_cache" and isinstance(v, ctypes._SimpleCData):
            key_nbytes = int(v.value)
            break
    assert key_nbytes is not None

    return [
        ("query", _tensor_to_host_list(tensors["query"])),
        ("key_cache", _tensor_to_host_list(tensors["key_cache"])),
        ("value_cache", _tensor_to_host_list(tensors["value_cache"])),
        ("block_table", _tensor_to_host_list(tensors["block_table"])),
        ("context_lens", _tensor_to_host_list(tensors["context_lens"])),
        ("local_out", [0.0] * (PA_CASE["batch"] * PA_CASE["num_heads"] * PA_CASE["head_dim"])),
        ("config", _tensor_to_host_list(tensors["config"])),
        ("key_cache_size_buf", [key_nbytes]),
        ("out", [0.0] * (PA_CASE["batch"] * PA_CASE["num_heads"] * PA_CASE["head_dim"])),
        ("peer_out", [0.0] * (nranks * PA_CASE["batch"] * PA_CASE["num_heads"] * PA_CASE["head_dim"])),
        ("notify_counter", [0] * (nranks * NOTIFY_SLOTS)),
    ]


def _load_rank_tensor(path: Path, dtype: str, count: int) -> torch.Tensor:
    """与 device 侧从同一份 .bin 读入的字节一致；避免 unpack+list 再 tensor 带来的 fp16 舍入差异。"""
    raw = path.read_bytes()
    if dtype == "float16":
        el = 2
        assert len(raw) == count * el, (path, len(raw), count * el)
        return torch.frombuffer(bytearray(raw), dtype=torch.float16, count=count).clone()
    if dtype == "float32":
        el = 4
        assert len(raw) == count * el, (path, len(raw), count * el)
        return torch.frombuffer(bytearray(raw), dtype=torch.float32, count=count).clone()
    if dtype == "int32":
        el = 4
        assert len(raw) == count * el, (path, len(raw), count * el)
        return torch.frombuffer(bytearray(raw), dtype=torch.int32, count=count).clone()
    if dtype == "int64":
        el = 8
        assert len(raw) == count * el, (path, len(raw), count * el)
        return torch.frombuffer(bytearray(raw), dtype=torch.int64, count=count).clone()
    raise ValueError(dtype)


def compute_golden(tensors: dict, params: dict) -> None:
    """单卡：标准 PA。分布式校验：从 artifact 读各卡输入，期望 out = sum(PA(rank_i))。"""
    if "query" in tensors:
        p = {**PA_CASE, "name": "Case1"}
        pa_compute_golden(tensors, p)
        return

    artifact_dir = params.get("artifact_dir")
    if not artifact_dir:
        return

    root = Path(artifact_dir)
    batch = PA_CASE["batch"]
    num_heads = PA_CASE["num_heads"]
    head_dim = PA_CASE["head_dim"]
    kv = PA_CASE["kv_head_num"]
    block_size = PA_CASE["block_size"]
    max_model_len = PA_CASE["max_model_len"]
    max_num_blocks_per_req = max_model_len // block_size
    context_lens_list = PA_CASE.get("context_lens_list")
    if context_lens_list:
        max_ctx = max(context_lens_list[:batch])
    else:
        max_ctx = PA_CASE["context_len"]
    total_blocks = batch * ((max_ctx + block_size - 1) // block_size)

    def pa_for_rank(r: int) -> torch.Tensor:
        d = root / f"rank_{r}"
        query = _load_rank_tensor(d / "query.bin", "float16", batch * num_heads * head_dim).view(
            batch, num_heads, head_dim
        )
        kv_elems = total_blocks * block_size * kv * head_dim
        key_cache = _load_rank_tensor(d / "key_cache.bin", "float16", kv_elems).view(
            total_blocks, block_size, kv, head_dim
        )
        value_cache = _load_rank_tensor(d / "value_cache.bin", "float16", kv_elems).view(
            total_blocks, block_size, kv, head_dim
        )
        block_table = _load_rank_tensor(d / "block_table.bin", "int32", batch * max_num_blocks_per_req).view(
            batch, max_num_blocks_per_req
        )
        context_lens = _load_rank_tensor(d / "context_lens.bin", "int32", batch)
        return paged_attention(
            query,
            key_cache,
            value_cache,
            kv,
            num_heads,
            1.0,
            block_table,
            context_lens,
        ).reshape(-1)

    n_ranks = int(params.get("nranks", 2))
    acc = pa_for_rank(0)
    for r in range(1, n_ranks):
        acc = acc + pa_for_rank(r)

    tensors["out"][:] = acc.tolist()


if __name__ == "__main__":
    from paged_attention_golden import run_golden_test

    run_golden_test({"Case1": PA_CASE}, "Case1", generate_inputs, label="batch_PA_AR_smda (batch>16 local PA)")
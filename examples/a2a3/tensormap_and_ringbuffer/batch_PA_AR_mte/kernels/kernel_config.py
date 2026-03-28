"""
batch_PA_AR_distributed_v1 kernel configuration.

Execution flow:
1. Each rank runs local batch paged attention into a window buffer.
2. After local PA completes, a notify kernel signals every other rank that the
   local PA output is ready.
3. Once the local rank receives all peer notifications for a chunk, the final
   AIV kernel directly reads all ranks' local_out from remote window memory.
4. The final kernel writes the multi-rank sum into the allreduced out.
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).resolve().parent
_EXAMPLES_ROOT = _KERNELS_ROOT.parent.parent
_PA_KERNELS_ROOT = _EXAMPLES_ROOT / "batch_paged_attention" / "kernels"

_platform = os.environ.get("PTO_PLATFORM", "a2a3sim")
if _platform != "a2a3":
    raise RuntimeError("batch_PA_AR_distributed_v1 currently requires PTO_PLATFORM=a2a3")

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "batch_pa_ar_distributed_v1_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

KERNELS = [
    {"func_id": 0, "name": "QK", "source": str(_PA_KERNELS_ROOT / "aic" / "aic_qk_matmul.cpp"), "core_type": "aic"},
    {"func_id": 1, "name": "SF", "source": str(_PA_KERNELS_ROOT / "aiv" / "aiv_softmax_prepare.cpp"), "core_type": "aiv"},
    {"func_id": 2, "name": "PV", "source": str(_PA_KERNELS_ROOT / "aic" / "aic_pv_matmul.cpp"), "core_type": "aic"},
    {"func_id": 3, "name": "UP", "source": str(_PA_KERNELS_ROOT / "aiv" / "aiv_online_update.cpp"), "core_type": "aiv"},
    {"func_id": 4, "name": "AIC_HUB", "source": str(_PA_KERNELS_ROOT / "aic" / "aic_hub.cpp"), "core_type": "aic"},
    {"func_id": 5, "name": "AIV_HUB", "source": str(_PA_KERNELS_ROOT / "aiv" / "aiv_hub.cpp"), "core_type": "aiv"},
    {"func_id": 6, "name": "PA_NOTIFY_READY", "source": str(_KERNELS_ROOT / "aiv" / "aiv_pa_notify_ready.cpp"), "core_type": "aiv"},
    {"func_id": 7, "name": "ALLREDUCE_ADD", "source": str(_KERNELS_ROOT / "aiv" / "aiv_allreduce_add.cpp"), "core_type": "aiv"},
]

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 24,
    "rounds": 1,
}

# Fixed demo case chosen to match the 16x16 templated PA kernels.
_BATCH = 32
_NUM_HEADS = 16
_HEAD_DIM = 16
_BLOCK_SIZE = 16
_MAX_MODEL_LEN = 256
_MAX_BLOCKS_PER_REQ = _MAX_MODEL_LEN // _BLOCK_SIZE
_MAX_CONTEXT_LEN = 33
_TOTAL_BLOCKS = _BATCH * ((_MAX_CONTEXT_LEN + _BLOCK_SIZE - 1) // _BLOCK_SIZE)
_OUT_ELEMS = _BATCH * _NUM_HEADS * _HEAD_DIM
_IN_CORE_BATCH = 16
_NUM_CHUNKS = (_BATCH + _IN_CORE_BATCH - 1) // _IN_CORE_BATCH

DISTRIBUTED_CONFIG = {
    "nranks": 4,
    "root": 0,
    "win_sync_prefix": 256,
    "buffers": [
        {"name": "query", "dtype": "float16", "count": _OUT_ELEMS, "placement": "device"},
        {"name": "key_cache", "dtype": "float16", "count": _TOTAL_BLOCKS * _BLOCK_SIZE * _HEAD_DIM, "placement": "device"},
        {"name": "value_cache", "dtype": "float16", "count": _TOTAL_BLOCKS * _BLOCK_SIZE * _HEAD_DIM, "placement": "device"},
        {"name": "block_table", "dtype": "int32", "count": _BATCH * _MAX_BLOCKS_PER_REQ, "placement": "device"},
        {"name": "context_lens", "dtype": "int32", "count": _BATCH, "placement": "device"},
        {"name": "local_out", "dtype": "float32", "count": _OUT_ELEMS, "placement": "window"},
        {"name": "out", "dtype": "float32", "count": _OUT_ELEMS, "placement": "device"},
        {"name": "config", "dtype": "int64", "count": 8, "placement": "device"},
        {"name": "notify_counter", "dtype": "int32", "count": _NUM_CHUNKS, "placement": "window"},
    ],
    "inputs": [
        "query",
        "key_cache",
        "value_cache",
        "block_table",
        "context_lens",
        "local_out",
        "out",
        "config",
        "notify_counter",
    ],
    "outputs": ["out"],
    "args": [
        "query",
        "key_cache",
        "value_cache",
        "block_table",
        "context_lens",
        "local_out",
        "out",
        "config",
        "notify_counter",
        "deviceCtx",
    ],
}

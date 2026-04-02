"""
batch_PA_AR_smda — 本地 batch_paged_attention（func 0–5）+ 按 chunk 的多 peer TNOTIFY + 同步 TGET + 独立 ADD（多卡 allreduce sum）。

需 PTO_PLATFORM=a2a3；默认验证 case 使用 batch>16，以便真实覆盖至少两个 chunk 的流水通信。
"""

import os
from pathlib import Path

_KERNELS_ROOT = Path(__file__).parent
_platform = os.environ.get("PTO_PLATFORM", "a2a3sim")

ORCHESTRATION = {
    "source": str(_KERNELS_ROOT / "orchestration" / "batch_pa_ar_distributed_orch.cpp"),
    "function_name": "aicpu_orchestration_entry",
}

_KERNELS_PA = [
    {"func_id": 0, "name": "QK", "source": str(_KERNELS_ROOT / "aic" / "aic_qk_matmul.cpp"), "core_type": "aic"},
    {"func_id": 2, "name": "PV", "source": str(_KERNELS_ROOT / "aic" / "aic_pv_matmul.cpp"), "core_type": "aic"},
    {"func_id": 4, "name": "AIC_HUB", "source": str(_KERNELS_ROOT / "aic" / "aic_hub.cpp"), "core_type": "aic"},
    {"func_id": 1, "name": "SF", "source": str(_KERNELS_ROOT / "aiv" / "aiv_softmax_prepare.cpp"), "core_type": "aiv"},
    {"func_id": 3, "name": "UP", "source": str(_KERNELS_ROOT / "aiv" / "aiv_online_update.cpp"), "core_type": "aiv"},
    {"func_id": 5, "name": "AIV_HUB", "source": str(_KERNELS_ROOT / "aiv" / "aiv_hub.cpp"), "core_type": "aiv"},
    {"func_id": 9, "name": "BARRIER", "source": str(_KERNELS_ROOT / "aiv" / "comm_barrier_kernel.cpp"), "core_type": "aiv"},
]

_KERNELS_DIST = [
    {"func_id": 6, "name": "PA_NOTIFY", "source": str(_KERNELS_ROOT / "aiv" / "aiv_pa_notify_ready.cpp"), "core_type": "aiv"},
    {"func_id": 7, "name": "TGET_PEER", "source": str(_KERNELS_ROOT / "aiv" / "aiv_tget_peer_out.cpp"), "core_type": "aiv"},
    {"func_id": 8, "name": "AR_ADD", "source": str(_KERNELS_ROOT / "aiv" / "aiv_allreduce_add.cpp"), "core_type": "aiv"},
]

KERNELS = _KERNELS_PA + (_KERNELS_DIST if _platform == "a2a3" else [])

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "orch_thread_num": 1,
    "block_dim": 24,
    "rounds": 1,
}

# Case1 参数（batch > 16，覆盖两个 chunk）
_PA_BATCH = 32
_PA_NUM_HEADS = 16
_PA_HEAD_DIM = 16
_PA_KV_HEAD_NUM = 1
_PA_BLOCK_SIZE = 16
_PA_MAX_MODEL_LEN = 256
_PA_MAX_BLOCKS_PER_REQ = _PA_MAX_MODEL_LEN // _PA_BLOCK_SIZE
_PA_CONTEXT_LENS_LIST = [33, 17, 29, 12, 31, 15, 27, 18, 25, 9, 33, 16, 21, 14, 28, 11, 30, 13, 24, 10]
_PA_MAX_CONTEXT_LEN = max(_PA_CONTEXT_LENS_LIST)
_IN_CORE_BATCH = 16
_PA_TOTAL_BLOCKS = _PA_BATCH * ((_PA_MAX_CONTEXT_LEN + _PA_BLOCK_SIZE - 1) // _PA_BLOCK_SIZE)
_PA_OUT_FLOATS = _PA_BATCH * _PA_NUM_HEADS * _PA_HEAD_DIM
_PA_NOTIFY_CHUNK_SLOTS = (_PA_BATCH + _IN_CORE_BATCH - 1) // _IN_CORE_BATCH
_NRANKS = int(os.environ.get("PTO_NRANKS", "2"))
if _NRANKS <= 0:
    raise ValueError("PTO_NRANKS must be positive")
_PA_NOTIFY_SLOTS = _PA_NOTIFY_CHUNK_SLOTS * _NRANKS
_PA_PEER_OUT_FLOATS = _PA_OUT_FLOATS * _NRANKS

if _platform == "a2a3":
    RUNTIME_ENV = {
        "PTO2_ENABLE_SDMA": "1",
    }

    DISTRIBUTED_CONFIG = {
        "nranks": _NRANKS,
        "root": 0,
        "win_sync_prefix": 256,
        "buffers": [
            {"name": "query", "dtype": "float16", "count": _PA_OUT_FLOATS, "placement": "device"},
            {
                "name": "key_cache",
                "dtype": "float16",
                "count": _PA_TOTAL_BLOCKS * _PA_BLOCK_SIZE * _PA_KV_HEAD_NUM * _PA_HEAD_DIM,
                "placement": "device",
            },
            {
                "name": "value_cache",
                "dtype": "float16",
                "count": _PA_TOTAL_BLOCKS * _PA_BLOCK_SIZE * _PA_KV_HEAD_NUM * _PA_HEAD_DIM,
                "placement": "device",
            },
            {"name": "block_table", "dtype": "int32", "count": _PA_BATCH * _PA_MAX_BLOCKS_PER_REQ, "placement": "device"},
            {"name": "context_lens", "dtype": "int32", "count": _PA_BATCH, "placement": "device"},
            {"name": "local_out", "dtype": "float32", "count": _PA_OUT_FLOATS, "placement": "window"},
            {"name": "out", "dtype": "float32", "count": _PA_OUT_FLOATS, "placement": "device"},
            {"name": "config", "dtype": "int64", "count": 7, "placement": "device"},
            {"name": "key_cache_size_buf", "dtype": "int64", "count": 1, "placement": "device"},
            {"name": "peer_out", "dtype": "float32", "count": _PA_PEER_OUT_FLOATS, "placement": "device"},
            {"name": "notify_counter", "dtype": "int32", "count": _PA_NOTIFY_SLOTS, "placement": "window"},
            {"name": "comm_barrier", "dtype": "int32", "count": _NRANKS, "placement": "window"},
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
            "key_cache_size_buf",
            "notify_counter",
            "comm_barrier",
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
            "key_cache_size_buf",
            "peer_out",
            "notify_counter",
            "comm_barrier",
            "deviceCtx",
        ],
    }
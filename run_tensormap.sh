#!/usr/bin/env bash
#
# 运行 tensormap_and_ringbuffer 算子测试
#
# 用法: ./run_tensormap.sh <算子名称> [设备数] [起始卡ID]
# 示例: ./run_tensormap.sh paged_attention
# 示例: ./run_tensormap.sh gather 2 0
#
# 单卡算子（设备数默认 1）: vector_example, paged_attention, batch_paged_attention, bgemm
# 多卡算子（设备数需 >= 2）: gather
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -lt 1 ]; then
    echo "用法: $0 <算子名称> [设备数] [起始卡ID]"
    echo "示例: $0 paged_attention"
    echo "示例: $0 gather 2 0"
    echo ""
    echo "可用算子: vector_example, paged_attention, batch_paged_attention, bgemm, gather, allgather_Manual, allgather_Tgather, paged_attention_allgather_Manual, paged_attention_allgather_Tgather"
    echo "  - 单卡: vector_example, paged_attention, batch_paged_attention, bgemm (默认 1 卡)"
    echo "  - 多卡: gather, allgather_Manual, allgather_Tgather, paged_attention_allgather_Manual, paged_attention_allgather_Tgather (需指定 2 卡及以上)"
    exit 1
fi

OP_NAME="$1"
N_DEVICES="${2:-1}"
FIRST_DEVICE="${3:-0}"

KERNELS_DIR="examples/tensormap_and_ringbuffer/${OP_NAME}/kernels"
GOLDEN_FILE="examples/tensormap_and_ringbuffer/${OP_NAME}/golden.py"

if [ ! -d "$KERNELS_DIR" ]; then
    echo "错误: 算子目录不存在: $KERNELS_DIR"
    exit 1
fi

if [ ! -f "$GOLDEN_FILE" ]; then
    echo "错误: golden 文件不存在: $GOLDEN_FILE"
    exit 1
fi

exec python3 examples/scripts/multi_card_run_example.py \
    -k "$KERNELS_DIR" \
    -g "$GOLDEN_FILE" \
    --n-devices "$N_DEVICES" \
    --first-device "$FIRST_DEVICE" \
    "${@:4}"

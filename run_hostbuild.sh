#!/usr/bin/env bash
#
# 运行 host_build_graph 多卡算子测试
#
# 用法: ./run_hostbuild.sh <算子名称> <设备数> <起始卡ID>
# 示例: ./run_hostbuild.sh allgather_Tgather 2 0
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -lt 3 ]; then
    echo "用法: $0 <算子名称> <设备数> <起始卡ID>"
    echo "示例: $0 allgather_Tgather 2 0"
    echo ""
    echo "可用算子示例: allgather_Tgather, allgather_Manual, paged_attention_gather, paged_attention_allgather_Tgather, paged_attention_allgather_Manual, ..."
    exit 1
fi

OP_NAME="$1"
N_DEVICES="$2"
FIRST_DEVICE="$3"

KERNELS_DIR="examples/host_build_graph/${OP_NAME}/kernels"
GOLDEN_FILE="examples/host_build_graph/${OP_NAME}/golden.py"

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

# cpt_and_comm

多卡「先计算，再通信」示例：GEMM → WindowMemCopyIn → TGATHER → WindowMemCopyOut。

## 流程

1. **GEMM**：每卡执行 C = A @ B（64x64）
2. **WindowMemCopyIn**：将 dev_C 前 64 元素拷贝到 HCCL window
3. **TGATHER**：root 从各 rank 收集到本地 window
4. **WindowMemCopyOut**：root 将 gathered 结果拷贝到 dev_out

## 依赖

- CANN（libhccl.so、libacl.so）
- pto-comm-isa（`pto::comm::TGATHER`、`ParallelGroup`）
- a2a3 真机（不支持 sim）

## 运行

```bash
# 1. 先 source CANN 环境（与跑 pto-comm-isa comm case 相同）
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
# 若 CANN 安装路径不同，用实际路径，如 nnae/nnrt 等

# 2. 编译 C++ HCCL 辅助库（与 pto-comm-isa 同方式链接，Python 通过 ctypes 调用）
cd examples/scripts/hccl_helper && mkdir -p build && cd build && cmake .. && make && cd ../../../../..

# 3. 设置 pto-comm-isa 路径（注意用单个 =）
export PTO_COMM_ISA_ROOT=/path/to/pto-comm-isa

# 验证：应存在 include/pto/pto-inst.hpp
ls $PTO_COMM_ISA_ROOT/include/pto/pto-inst.hpp

# 4. 2 卡运行
python examples/scripts/multi_card_run_example.py \
  -k examples/host_build_graph/cpt_and_comm/kernels \
  -g examples/host_build_graph/cpt_and_comm/golden.py \
  --n-devices 2 --first-device 0
```

## 配置

- `RUNTIME_CONFIG.requires_comm`: True
- `RUNTIME_CONFIG.n_devices`: 2
- `RUNTIME_CONFIG.root`: 0

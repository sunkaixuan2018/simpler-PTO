# cpt_and_comm

多卡「先计算，再通信」示例：GEMM → WindowMemCopyIn → TGATHER → WindowMemCopyOut。

## 流程

```
  Rank 0                              Rank 1
  ──────                              ──────
  GEMM: C = A @ B                     GEMM: C = A @ B
       ↓                                   ↓
  WindowMemCopyIn:                    WindowMemCopyIn:
    dev_C[:64] → win_in[rank0]          dev_C[:64] → win_in[rank1]
       ↓                                   ↓
  ─── HcclBarrier (host sync) ────────────────────────
       ↓                                   ↓
  Gather (root):                      Gather (non-root):
    TGATHER 从各 rank 的 win_in           (skip TGATHER)
    收集到 win_dst
       ↓
  WindowMemCopyOut (root only):
    win_dst → dev_out
       ↓
  ─── HcclBarrier (host sync) ────────────────────────
       ↓                                   ↓
  比较 dev_out 与 golden               完成
```

每卡 GEMM 为 64×64 float32 矩阵乘，gather 收集每 rank 前 64 个元素，
root 最终 `dev_out` 形状为 `(n_ranks × 64,)`。

## 前置依赖

| 依赖 | 说明 |
|------|------|
| CANN | Ascend 计算平台（提供 `aclrtSetDevice`、`HcclGetRootInfo` 等） |
| pto-comm-isa | 提供 `pto::comm::TGATHER`、`ParallelGroup`、comm kernel headers |
| a2a3 真机 | 不支持 sim 模式 |

## 环境准备

```bash
# 1. source CANN 环境
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 2. 编译 C++ HCCL 辅助库（libhccl_helper.so）
cd examples/scripts/hccl_helper
mkdir -p build && cd build && cmake .. && make
cd ../../../..

# 3. 设置 pto-comm-isa 路径（单个 =，不要 ==）
export PTO_COMM_ISA_ROOT=/path/to/pto-comm-isa

# 验证：应存在 include/pto/pto-inst.hpp
ls $PTO_COMM_ISA_ROOT/include/pto/pto-inst.hpp
```

## 运行

```bash
# 2 卡运行（device 0 和 device 1）
python3 examples/scripts/multi_card_run_example.py \
  -k examples/host_build_graph/cpt_and_comm/kernels \
  -g examples/host_build_graph/cpt_and_comm/golden.py \
  --n-devices 2 --first-device 0
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--n-devices N` | 使用 N 张卡（默认取 `kernel_config.py` 中的 `n_devices`） |
| `--first-device D` | 起始设备 ID（如 `--first-device 4` + `--n-devices 4` 使用 4,5,6,7） |
| `-v` / `--verbose` | 打印 debug 级别日志 |
| `--enable-profiling` | 启用 profiling 和 swimlane 生成 |

### 调试选项

```bash
# 当 golden 比对失败时，导出 actual/golden .npy 到 debug/ 目录
export PTO_DUMP_MISMATCH=1
```

## 配置说明（kernel_config.py）

```python
RUNTIME_CONFIG = {
    "runtime": "host_build_graph",
    "n_devices": 2,            # 默认使用 2 张卡
    "first_device_id": 0,      # 起始设备 ID
    "requires_comm": True,     # 启用 HCCL 通信
}
```

## 关键实现文件

| 文件 | 作用 |
|------|------|
| `kernels/aic/kernel_gemm_tile.cpp` | AICore 64×64 GEMM kernel |
| `kernels/aiv/window_memcopy_in.cpp` | AIVector: dev_C → win_in |
| `kernels/aiv/gather_kernel.cpp` | AIVector: TGATHER（root only） |
| `kernels/aiv/window_memcopy_out.cpp` | AIVector: win_dst → dev_out（root only） |
| `kernels/orchestration/cpt_and_comm_orch.cpp` | 编排：构建 task graph、设置依赖 |
| `golden.py` | Golden 参考：模拟各 rank GEMM + gather |
| `../../scripts/hccl_helper/hccl_helper.cpp` | C++ HCCL 辅助库（init comm、barrier） |
| `../../scripts/hccl_bindings.py` | Python ctypes 绑定 libhccl_helper.so |
| `../../scripts/multi_card_code_runner.py` | 多卡运行框架（编译一次，多进程执行） |
| `../../scripts/multi_card_run_example.py` | 命令行入口 |

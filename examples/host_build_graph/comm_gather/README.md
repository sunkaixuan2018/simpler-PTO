# comm_gather：四卡集体 TGATHER

仅跑多卡集体通信 TGATHER（root 从各 rank 收数据），无计算 kernel。与 **gemm_gather** 独立，不覆盖原有 case。

## 依赖

- a2a3 真机，CANN 安装（ACL、HCCL、hccl_context.h）
- Linux（fork/mmap 建联）
- PTO-ISA 含 `pto/comm`（TGATHER、ParallelGroup）

## 构建 runner（在 a2a3 环境执行一次）

`run_gather_kernel.cpp` 内含 `__global__ AICORE` 设备代码，需用 **CANN 混合编译工具链** 编译（不能只用 g++）。

在工程根目录或 `kernels/host` 下：

```bash
export PTO_ISA_ROOT=/path/to/pto-isa   # 例如 examples/scripts/_deps/pto-isa
export ASCEND_HOME=/path/to/Ascend/ascend-toolkit/latest
cd examples/host_build_graph/comm_gather/kernels/host
make
```

生成 `comm_gather_runner`。若当前环境无 CANN 混合编译器，需在 a2a3 服务器上按 CANN 文档用对应命令编译上述源文件。

## 运行

在 a2a3 上（且已构建好 `comm_gather_runner`）：

```bash
python examples/scripts/run_example.py \
  -k examples/host_build_graph/comm_gather/kernels \
  -g examples/host_build_graph/comm_gather/golden.py \
  --n-ranks 4 --first-device 0
```

- `--n-ranks`：rank 数，默认 4  
- `--first-device`：起始 device ID，默认 0（使用 device 0,1,2,3）

校验在 C++ 内完成（root 检查 gathered 结果），通过则进程退出码 0，Python 据此判成功。

## 目录与流程简述

- `kernels/host/comm_common.hpp`：HCCL/ACL 建联（TestContext、ForkAndRunWithHcclRootInfo、WindowAlloc、HcclRemotePtr）
- `kernels/host/run_gather_kernel.cpp`：设备 kernel `TGatherKernelImpl` + 主机侧 `RunGatherKernel` / `RunGather`
- `kernels/host/main_comm_gather.cpp`：main，解析 `--n-ranks` / `--first-device`，调用 `ForkAndRunWithHcclRootInfo` 后执行 `RunGather`
- 建联在**各 fork 子进程内**、跑 kernel 前由 `TestContext.Init` 完成；rank0 先写 `HcclRootInfo`，其余 rank 再 `HcclCommInitRootInfo`

## 与 gemm_gather 的区别

| 项目       | gemm_gather           | comm_gather              |
|------------|------------------------|---------------------------|
| 运行方式   | 单进程 Runtime + 编排 | 多进程 fork + 独立 runner |
| 通信       | 单卡索引 Gather       | 多卡集体 TGATHER (HCCL)   |
| 计算       | 有 GEMM               | 无                        |
| 校验       | Python golden         | C++ root 自检             |

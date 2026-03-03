# 多卡 Gather Case 实现计划（comm_gather）

在 simpler-PTO 中新增一个**仅跑四卡 TGATHER** 的 case，不接计算任务，参考 pto-comm-isa 的 `tests/npu/a2a3/comm/st/testcase/tgather` 的流程与结构。

---

## 一、目标与范围

- **Case 名**：`comm_gather`（建议路径：`examples/host_build_graph/comm_gather/`）
- **功能**：4 张卡做 collective gather（root 从各 rank 收数据），无 GEMM 等计算任务
- **平台**：a2a3 真机（依赖 HCCL + ACL，不跑 a2a3sim）
- **参考实现**：pto-comm-isa 的 tgather（`ForkAndRunWithHcclRootInfo` + `TestContext` + `RunGatherKernel` + `TGatherKernelImpl`）

---

## 二、代码结构分析

### 2.1 当前 simpler-PTO 单卡流程

- **入口**：`run_example.py` → `code_runner.run()`
- **参数**：`-k kernels_dir`、`-g golden.py`、`-d device`（默认 0）、`-p platform`
- **流程**：
  1. 加载 `kernel_config.py`、`golden.py`
  2. 构建 runtime（host + aicpu + aicore）
  3. 编译 orchestration 为 SO，编译各 kernel 为 binary
  4. 单进程：`Runtime()` → `initialize(orch_so, func_args, kernel_binaries)` → `launch_runtime()` → `finalize()` → Python 侧与 golden 比对
- **特点**：单进程、单 device、无 HCCL、无 rank 概念

### 2.2 pto-comm-isa 多卡 tgather 流程

- **入口**：GTest 或 main 调用 `RunGather<T, count>(n_ranks, n_devices, first_rank_id, first_device_id)`（例如 4 卡：`RunGather<float,256>(4,4,0,0)`）
- **多进程**：`ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id, lambda)`  
  - fork 出 n_ranks 个子进程  
  - rank0 子进程：`HcclGetRootInfo` 写入 mmap 的 `SharedBootstrap`  
  - 其余 rank 子进程：轮询等待 `state==1` 后读 `rootInfo`  
  - 每个子进程执行 lambda：`RunGatherKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo, root)`
- **单 rank 内**（`RunGatherKernel`）：
  1. **建联**：`TestContext.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)`  
     - `aclInit` → `aclrtSetDevice(deviceId)` → `aclrtCreateStream`  
     - `HcclCommInitRootInfo` → `HcclGetCommName` → `HcomGetCommHandleByGroup`  
     - `HcclAllocComResourceByTiling` → 得到 `deviceCtx`（含 `windowsIn` 等），再 `aclrtMemcpy` 到 `hostCtx`
  2. **内存**：`WindowAlloc(localWinBase, winOffset, src_size)` / `dst_size`（在 HCCL 注册的 window 内分配）
  3. **数据**：host 上初始化 `src_host`，拷贝到 device 的 `src_ptr`；root 再初始化 `dst_host` 并拷到 `dst_ptr`
  4. **同步**：`HcclHostBarrier(comm, stream)`
  5. **发 kernel**：`TGatherKernelImpl<<<1, nullptr, ctx.stream>>>(dst_ptr, src_ptr, ctx.deviceCtx, n_ranks, root)`
  6. **同步**：`aclrtSynchronizeStream` → `HcclHostBarrier`
  7. **校验**：root 将 `dst` 拷回 host，在 C++ 内逐元素与期望值比较
  8. **收尾**：`TestContext.Finalize()`

**结论**：  
- **建联时机**：在每个 fork 出的子进程内、在跑 kernel **之前**，由 `TestContext.Init` 完成（HCCL 通信域 + RDMA window 等）。  
- **谁先谁后**：rank0 先执行 `HcclGetRootInfo` 并写入 mmap；其他 rank 等 `state==1` 后再用 `rootInfo` 做 `HcclCommInitRootInfo`，因此建联是“rank0 提供 rootInfo，各 rank 各自 Init”的顺序。

### 2.3 与现有 simpler-PTO 的差异

| 维度         | 现有 host_build_graph case | comm_gather（本 case）        |
|--------------|-----------------------------|--------------------------------|
| 进程/设备    | 单进程、单 device           | 多进程（fork）、每进程一 device |
| 运行时       | Runtime + orchestration     | 无 Runtime；独立 host 可执行   |
| 建联         | 无                          | 每 rank 内 Init 时建联 HCCL     |
| Kernel 调用  | 通过 launch_runtime 统一发  | 各 rank 自己 aclrtLaunchKernel   |
| 校验         | Python golden 比对          | C++ 内 root 自检（或再写文件）  |

---

## 三、需要修改/新增的内容

### 3.1 启动命令与参数

- **目标用法**（示例）：
  ```bash
  python run_example.py -k examples/host_build_graph/comm_gather/kernels \
                        -g examples/host_build_graph/comm_gather/golden.py \
                        --n-ranks 4 --first-device 0
  ```
- **修改点**：
  - 在 **run_example.py** 中增加参数：`--n-ranks`（默认 4）、`--first-device`（默认 0）。
  - 将这两个参数传入 `create_code_runner(..., n_ranks=..., first_device_id=...)`（或等价接口）。

### 3.2 何时走“多卡 gather”分支

- 在 **code_runner** 中根据 `kernel_config.py` 的 `RUNTIME_CONFIG["runtime"] == "comm_gather"` 判断。
- 若为 `comm_gather`：
  - **不**构建默认 runtime，**不**编译 orchestration，**不**走 `Runtime()` / `initialize()` / `launch_runtime()`。
  - 改为：构建 **comm_gather 专用 host 可执行** + 编译 **AIV gather kernel**，然后执行该 host 可执行（内部 fork 4 个进程），根据进程退出码或约定输出判断成功/失败。

### 3.3 通信建联的调度时机（再强调）

- 建联**不在** Python 主进程里做，也**不在**“先起好 4 个进程再一起 Init”的集中步骤。
- 建联在 **每个 fork 出的子进程内**、在调用 `RunGatherKernel(rankId, ...)` 时，由 **RunGatherKernel 开头的 `TestContext.Init(...)`** 完成。
- 顺序：rank0 生成并写入 `HcclRootInfo` → 其他 rank 读到 `rootInfo` → 各 rank 各自 `Init`（含 `HcclCommInitRootInfo`、`HcclAllocComResourceByTiling`）→ 再各自做 `WindowAlloc`、拷贝、barrier、发 kernel。

### 3.4 目录与文件规划

```
examples/host_build_graph/comm_gather/
├── README.md                    # 说明：四卡 gather、无计算、如何跑、依赖
├── golden.py                    # 多卡时以 C++ 校验为主，此处可占位或仅做形状/接口兼容
├── PLAN_multicard_gather.md     # 本计划文档（可后续删除或保留作记录）
└── kernels/
    ├── kernel_config.py         # runtime = "comm_gather", n_ranks = 4 等
    ├── aiv/
    │   └── kernel_gather_comm.cpp  # 设备侧：TGatherKernelImpl(dst, src, HcclDeviceContext*, nranks, root)
    └── host/                     # 与 pto-comm-isa 对齐的 host 侧逻辑
        ├── comm_common.hpp       # TestContext, ForkAndRunWithHcclRootInfo, WindowAlloc, HcclRemotePtr 等（可从 pto-isa 的 tests/npu/a2a3/comm/st/testcase/common.hpp 复制或引用）
        ├── run_gather_kernel.cpp  # RunGatherKernel<T,count>：Init、WindowAlloc、拷贝、Barrier、发 kernel、Sync、Barrier、root 校验、Finalize
        └── main_comm_gather.cpp   # main：解析 n_ranks/first_device（或从 env/argv 读），调用 ForkAndRunWithHcclRootInfo(n_ranks, 0, first_device, RunGatherKernel<...>)
```

- **kernel_config.py**：  
  - `RUNTIME_CONFIG = {"runtime": "comm_gather", "n_ranks": 4, "first_device_id": 0}`  
  - 只登记 AIV gather kernel（如 `kernel_gather_comm.cpp`），**不需要** orchestration（可无 `ORCHESTRATION` 或填占位，code_runner 在 comm_gather 分支下不读 orchestration）。

### 3.5 构建

- **AIV kernel**：沿用现有 `KernelCompiler.compile_incore(..., core_type="aiv")`，产出 kernel binary（与现有 case 一致，供 host 可执行加载）。
- **Host 可执行**（comm_gather 专用）：
  - 需能编译 C++（如 g++）并链接 **ACL**、**HCCL**（及 CANN 提供的 `hccl_context.h` 等）。
  - 源文件：`main_comm_gather.cpp`、`run_gather_kernel.cpp`（以及 `comm_common.hpp` 头文件）。
  - 若 simpler-PTO 当前没有 HCCL 的构建配置，需要在本 case 或公共脚本中增加：HCCL 头路径、库路径、链接 `hccl` 等（与 pto-comm-isa 的编译方式对齐）。
  - 可执行需要能**加载 kernel binary**（例如从文件路径，由 Python 在运行前写入临时文件并传入 argv），并在各 rank 内用 ACL 的 launch 接口（如 `aclrtLaunchKernel` 或当前 CANN 推荐方式）发起 kernel。

### 3.6 运行流程（comm_gather 分支）

1. **code_runner** 检测到 `runtime == "comm_gather"`。
2. 使用 `n_ranks`、`first_device_id`（来自 `kernel_config` 或命令行覆盖）。
3. 编译 AIV kernel，将得到的 kernel binary 写到临时文件（如 `comm_gather_kernel.bin`）。
4. 编译并链接 host 可执行（若尚未编译或源有更新）。
5. 执行该可执行，传入参数（例如 kernel binary 路径、n_ranks、first_device_id）；**可执行内部**执行 `ForkAndRunWithHcclRootInfo(n_ranks, 0, first_device_id, ...)`，在子进程中完成建联与 gather。
6. 根据可执行退出码判断成功/失败；可选：若可执行将 root 的 result 写到文件，再在 Python 里用 `golden.py` 的 `compute_golden` 做一次比对（本计划建议先采用 C++ 内校验，退出码即可）。

### 3.7 Golden 校验

- **建议**：与 pto-comm-isa 一致，**仅在 C++ 侧**由 root 在 `RunGatherKernel` 内校验 `dst_host`；可执行返回 0 即表示通过。
- **golden.py**：保留 `generate_inputs` / `compute_golden` 接口以满足 code_runner 的通用加载逻辑，但多卡时可不依赖其输出做比对（例如 `generate_inputs` 返回空或占位，`compute_golden` 空实现）。

### 3.8 依赖

- **pto-isa（_deps）**：已有 `pto/comm`（TGATHER、ParallelGroup）、`pto_comm_inst.hpp`；需能包含 `tests/npu/a2a3/comm/st/testcase/common.hpp`，或将 `comm_common.hpp` 复制到本 case 的 `kernels/host/` 并适配 include。
- **CANN / HCCL**：`hccl.h`、`hccl_types.h`、`hccl_context.h`、`hcom.h`（若存在）；目标环境需能编译并链接。

---

## 四、实施步骤小结（Plan）

| 步骤 | 内容 |
|------|------|
| 1 | 新增目录 `examples/host_build_graph/comm_gather/` 及 `kernels/`、`kernels/aiv/`、`kernels/host/`。 |
| 2 | 从 pto-comm-isa 抽取并适配 **comm_common.hpp**（TestContext、ForkAndRunWithHcclRootInfo、WindowAlloc、HcclRemotePtr）到 `kernels/host/comm_common.hpp`（或引用 pto-isa 的 common.hpp 并保证 include 路径）。 |
| 3 | 从 pto-comm-isa 的 tgather 抽取 **RunGatherKernel** 与 **TGatherKernelImpl**：设备侧放入 **kernel_gather_comm.cpp**，host 侧放入 **run_gather_kernel.cpp**；**main_comm_gather.cpp** 实现 main，调用 `ForkAndRunWithHcclRootInfo(4, 0, first_device_id, ...)`。 |
| 4 | 编写 **kernel_config.py**（`runtime: comm_gather`，n_ranks=4，仅注册 AIV kernel）。 |
| 5 | 编写 **golden.py**（满足接口即可，校验以 C++ 为主）。 |
| 6 | 在 **run_example.py** 中增加 `--n-ranks`、`--first-device`，并传入 code_runner。 |
| 7 | 在 **code_runner.run()** 中增加分支：当 `runtime == "comm_gather"` 时，不构建默认 runtime/orchestration，改为构建 comm_gather 的 host 可执行 + AIV kernel，执行可执行并根据退出码判断成功。 |
| 8 | 实现 **comm_gather 的构建逻辑**：编译 host 源文件并链接 ACL/HCCL；编译 AIV kernel；约定 kernel binary 路径或通过参数传入可执行。 |
| 9 | 编写 **README.md**：说明四卡 gather、如何运行、环境与依赖。 |

---

## 五、风险与注意点

- **环境**：需在 a2a3 真机且安装 CANN/HCCL，保证头与库可用；CI 若仅单卡可考虑跳过或 mock。
- **HCCL 头**：`hccl_context.h` 等可能随 CANN 版本变化，若 pto-isa 与当前 CANN 不一致，需在 `comm_common.hpp` 或本 case 的 include 中做兼容。
- **Windows**：当前方案依赖 fork/mmap，仅适用于 Linux；若需在 Windows 跑多卡，需另设计（如多进程 + 共享内存或 MPI 式启动）。

以上为多卡 gather case 的完整分析与实现计划，确认后即可按步骤修改代码。

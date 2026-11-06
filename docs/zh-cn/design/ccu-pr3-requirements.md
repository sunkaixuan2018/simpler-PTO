# CCU PR3 开发需求：callable 注册、通信域与资源生命周期

更新：2026-11-06。原生 PR1 Session / PR2 TMR / PR3 provider 已接线，三 PR 硬件联调仍未验收。

方案依据：[飞书《CCU 函数注册与调用接口方案》](https://hw-native-sys.feishu.cn/wiki/YOXZwef8QiabQfktGBYcXihVngg)，以及主工作区 `simpler-main/new-doc/index.html` v7。
本文件保留公开接口和验收要求，并记录当前实现。实现任务的勾选只表示代码已接入，不表示硬件验收通过。

## 目标与开发边界

让 CCU 函数随普通 ChipCallable 注册，在 run 前就确定函数 ID；通信域仍在 Host 的 orch 编排中创建。
运行时在相关设备任务执行前，自动准备该 callable 在该域上的 CCU 执行实例，管理模块、域、窗口及未完成请求的生命周期。

首版范围：A5 / CANN 9.3、单机两卡 UB_CTP、TMR；以 Broadcast 和最小用户函数验证连续 FP32 数据。
每个执行上下文最多一个活动请求，同一 L2 可反复调用。所有 CCU Tensor 实参必须来自选定通信域。

| 交付方 | 负责内容 | PR3 的接入要求 |
| ------ | -------- | -------------- |
| PR1 | 模块 / builder 的平台适配、建链、域窗口凭据、实例创建、排空与可确认关闭 | 通过 ABI 2 的 CcuSession 调用平台能力 |
| PR2 | AIV 设备接口、隐藏 callable 身份的设备侧接入、参数快照、TMR 完成与解依赖 | 提供已准备实例及正确的生命周期，消费其请求状态 |
| PR3 | callable 打包注册、域发现、实例准备、Host 身份传递与引用管理 | 本文件的主要工作 |
| PyPTO 后续 PR | 生成内嵌模块和 CCU 列表、域内数据规划、设备调用及依赖 | 提供稳定的构建和运行时契约 |

不在本 PR 实现 PyPTO DSL、HBG、多机或多请求并发；不迁移 orch 建域到 `worker.init()`。

## 已确定的注册与调用关系

```python
# 各 binary 是已编译的二进制对象。
# CcuFunction.build 和 ccu_functions 是新增接口。
chip_callable = ChipCallable.build(
    signature=orch_signature,
    func_name="orchestrate",
    binary=orch_binary,
    children=[(0, compute_kernel), (1, ccu_wrapper_kernel)],
    ccu_functions=[
        CcuFunction.build(binary=ccu_module_binary, entry="build_copy"),
        CcuFunction.build(binary=ccu_module_binary, entry="build_reduce"),
    ],
)
handle = worker.register(chip_callable)
```

Copy 的 CCU ID 为 0，Reduce 为 1，编号来自列表下标，与 children 的 AICore kernel ID 独立。
`worker.register` 注册定义时不要求域已存在；不再增加用户单独执行的 CCU 注册步骤。
`ChipCallable.build` 的普通任务 signature 保留，不新增 `CcuSignature / CcuTensorParam / CcuScalarParam`。

| 约束 | 要求 |
| ---- | ---- |
| 稳定 ID | 同一编译产物跨 run 固定；增删或重排列表需同步重编译 AIV 调用代码 |
| callable 隔离 | 不同 callable 都可使用 CCU ID 0，不能只按域和数字 ID 定位 |
| AIV 关联 | 一个 AIV 可调用多个 CCU ID，多个 AIV 可调用同一 CCU ID，不做一对一绑定 |
| 设备定位 | 当前 callable 身份 + domain_ctx + ccu_id → 对应执行实例中的程序 |
| 隐藏上下文 | callable 身份由运行时从任务上下文传到 AIV，不增加用户每次 Host 启动的 ID 参数 |
| 内嵌部署 | 模块内容与入口一起进入产物；多个入口可共享模块，不依赖原始独立 .so 或构建路径 |
| builder | 内嵌模块可以是 Host builder，域资源已知后再生成 / 实例化 CCU 程序；不能当作 AIV 机器码加载 |
| 缓存 | 内部可使用内容哈希，缓存必须覆盖模块、入口列表及 ABI / 版本；设备 API 仍用顺序 ID |

加载过程可以产生临时 .so。容器必须区分二进制种类和版本，并对格式、入口和范围做校验。
旧的无 CCU callable 应继续按原有路径运行；布局、序列化或缓存格式变化必须有明确的兼容或拒绝规则，不能静默误读。

## orch 建域与数据来源

```python
# Host 编排片段；worker.register 已完成。
domain = orch.get_or_allocate_domain(
    name="ccu_domain",
    members=[0, 1],
    window_size=1024,
    buffers=[
        CommBufferSpec("src", "opaque", 256, 256),
        CommBufferSpec("dst", "opaque", 256, 256),
        CommBufferSpec("scratch", "opaque", 256, 256),
        CommBufferSpec("order", "opaque", 64, 64),
    ],
    retain_after_run=True,
    workspace_engines=("ccu",),
)
ctx = domain.device_context(worker_id, workspace_engines=("ccu",))
src = domain[worker_id].buffers["src"].tensor((64,), DataType.FLOAT32)
dst = domain[worker_id].buffers["dst"].tensor((64,), DataType.FLOAT32)
scratch = domain[worker_id].buffers["scratch"].tensor((64,), DataType.FLOAT32)
order = domain[worker_id].buffers["order"].tensor((8,), DataType.UINT64)
```

`"ccu"` 是新增引擎支持。src、dst 和用户 scratch 必须来自本域；可使用已准备范围内的合法子视图。
缓冲区名字不是 CCU 调用 ABI，设备侧按位置传参；order 只用于外层任务排序，不传给本例 CCU 函数。

前置算子可直接写域内 src，不需额外数据复制。已有域外结果需要显式复制到域内后再调用。
窗口和凭据须在 CCU 访问前准备好，准备时不要求生产者已经写入数据。
`CcuCallArgs` 不注册内存、不隐式搬运数据，也不支持用 scalar 裸地址绕过 Tensor 的域检查。

```text
构建 ChipCallable → worker.register：保存模块、入口列表和顺序 ID
Host orch 创建 / 复用域：窗口、通道、访问凭据就绪
运行时发现本次 callable 所用的域：准备 callable / domain 执行实例
发布设备可见上下文：相关设备任务才可执行
AIV 调用 CCU：PR2 跟踪完成，相关模块和内存继续被持有
成功 run 后按配置保留；销毁或关闭前先排空
```

域创建或复用时，运行时按稳定 digest 顺序为当前注册的 CCU callable 提前准备实例；新增 callable 在下一次域复用时准备。提交时将 context scalar 与当前 Worker 的有效域表匹配，校验已准备的 callable、域和代次。
准备在本次编排首次任务提交前完成，有逐 callable/domain 初始化成本；设备每轮调用不执行 Host prepare，也不由 AIV 发 RPC。

## 实现任务与所有权

- [x] 增加 `CcuFunction.build(binary, entry)` 和 `ChipCallable.build(..., ccu_functions=...)`，完整支持模块与入口列表的序列化、加载及稳定顺序。
- [x] 更新 callable 二进制布局、范围 / 版本检查、静态断言、Python 绑定、编译缓存和 scene-test 构建入口，确保产物自包含。
- [x] 将 CCU 定义纳入普通 Worker 注册 / 注销流程；定义注册与按域实例化分开，不引入第二次用户注册。
- [x] 扩展 `workspace_engines` 的规范化、配置复用校验、位掩码、C++ / Python 布局和 `device_context` 检查。
- [x] 具体化域发现机制，在设备执行前准备对应实例，并将当前 callable 身份接入隐藏运行时上下文，与 PR2 接通。
- [x] 准备域窗口的访问凭据及有效范围，发布可供 PR2 验证的域 / 实例信息；域外、跨域及过期引用不得被接受。
- [x] 管理 callable / domain 实例复用及创建失败回滚；只有资源完整就绪才允许设备使用。
- [x] 处理成功 run 后保留、域重建、callable 注销、域销毁和 Worker 关闭；排空未完成请求后再释放。
- [x] 接入现有域回滚和未确认清理的所有权保留机制，并补齐 CCU 后端可确认停止 / 关闭的结果。

| 所有者 | 持有内容 | 释放条件 |
| ------ | -------- | -------- |
| callable 注册表 | 内嵌模块、入口定义及 ID 映射 | 无仍使用该模块的实例 / 请求后，才可卸载 |
| 通信域 | 窗口、访问凭据、通道及相关资源 | 该域所有使用者已排空，确认硬件停止访问 |
| callable / domain 实例 | 本域程序资源，及 callable 和域的引用 | 本实例无未完成请求，后端清理已确认 |
| 已接受请求 | 运行时参数快照和执行所需引用 | PR2 确认完成并回收；错误时按可确认的停止状态清理 |

域仅在成功 run 后按配置保留。域重建更新内部实例和凭据，拒绝旧上下文，但不改变同一产物的 CCU ID。
释放域 A 不能破坏域 B 的实例；注销一个 callable 不能卸载其他实例仍使用的模块。
失败不是成功完成；未确认硬件停止时继续保留所有相关资源，并禁止把它们交给下一次运行复用。
复用 #2835 的回滚思路不能代替 CCU 自身的排空 / 关闭确认。

## 代码入口与共享交付

以下路径对应当前接线；完整构建需要同时包含 PR1、PR2 和 PR3。

| 位置 | 开发用途 |
| ---- | -------- |
| `src/common/task_interface/callable.h` | 扩展内嵌 CCU 定义、模块与列表的 wire 格式及校验 |
| `src/common/task_interface/chip_callable_layout.h` | 上传布局、大小和缓存 / 哈希逻辑，与 AICore 子表隔离 |
| `python/bindings/task_interface.cpp`、`simpler_setup/scene_test.py` | 构建接口、绑定和正常编译链 |
| `python/simpler/worker.py`、`callable_identity.py` | Worker 注册及 callable 身份、运行和资源引用 |
| `python/simpler/orchestrator.py`、`comm_domain.py` | orch 建域、引擎配置、域发现及复用条件 |
| `src/common/worker/` | Host 注册 / 运行时桥接、隐藏上下文和生命周期 |
| `src/common/platform_comm/comm_context.h` | CCU workspace 与共享布局；与 PR1 / PR2 共用定义 |
| `src/a5/platform/onboard/host/ccu/` | PR1 Session、ABI 2 builder 平台适配、排空和关闭 |
| `src/a5/platform/onboard/host/ccu_domain_provider.inc` | PR3 真实 Session 准备、发布及运行引用 |
| `src/common/platform_comm/ccu/` | PR2/PR3 设备协议、发布、运行引用与关闭约束 |
| `tests/ut/cpp/common/task_interface/`、`tests/ut/py/test_callable_identity.py` | 产物布局、序列化、编号和 callable 隔离 |
| `tests/ut/py/test_comm_domain.py`、`tests/st/worker/comm_domain/` | 引擎配置、域准备、失败回滚、复用和销毁 |

CCU 已纳入引擎名单和共享布局。Workspace 指向 PR1 的 CcuDeviceContext，实际请求槽由 PR1 管理；两者 ABI 均为 2。ChipCallable 线格式版本保持 1，内嵌模块的 ABI 字段独立为 2。旧版未带版本的序列化产物明确拒绝并要求重新构建。Python 绑定、Host runtime、内嵌模块和设备内核须一起重建。

公开模块导出 `uint32_t simpler_ccu_module_abi_version()` 返回 2，入口为 `int(CcuBuildContext*)`，Tensor / scalar 按位置编号。旧 ABI 1 模块须重编。内部独立 `Builder/Program` 的 ABI 1 测试组件不用于生产加载路径。

每个 callable/domain 使用独立的 Session 和 communicator rootinfo，双方校验域代次及隐藏 native callable ID。native prepare 持有 RunLease；PR2 在普通生产者停止后确认 CCU 完成，再解依赖和释放引用。只有明确未启动的准备才可通过 `cancel_unlaunched` 释放自身引用。最后一个 callable handle 注销前须先释放仍发布该 callable 的域，拒绝注销时保留原注册状态。

清理先停止 CCU，再释放域窗口及父通信资源。错误码不证明生产者已停止；未确认的读者或 SDK 关闭失败必须保留整个所有权链。即使 Session 已关闭，只要父 CommHandle 尚未完整释放，也不得执行设备 reset。恢复路径跳过仍持有 CCU 通信资源的全设备同步；fatal finalize 先停止 Host 采集线程，再检查关闭条件。只有父通信对象完整销毁后才撤销 reset 阻挡。没有 CCU 的原路径保持不变。

模拟平台、非 A5、非 TMR、非两成员域及直接 L2 CCU 提交仍不支持；受支持的原生路径已进入真实执行，不再统一拒绝含 CCU 定义的 callable。

## 验收清单

- [ ] 没有原始独立 .so 仍可部署；两个入口共享一个内嵌模块，非法格式 / 版本 / 入口被正确拒绝。
- [ ] 无 CCU 的普通 callable 保持原有行为；布局和缓存变化不误读旧产物。
- [ ] CCU 与 AICore 独立编号；同一产物跨 run 稳定；不同 callable 的 ID 0 隔离；列表变化与调用代码、缓存同步。
- [ ] 先 worker.register，再在 orch 建域可用；无需第二次用户 CCU 注册，无 AIV 临时请求 Host 准备。
- [ ] 相关设备任务执行前实例已准备；缺失资源、错误引擎配置或准备失败不发布可执行上下文。
- [ ] 域内 Tensor / 合法子视图可用；域外、跨域、越界及过期上下文被拒绝；凭据覆盖实际范围。
- [ ] 生产者直接写域窗口的路径无额外复制；域外已有结果的接入路径有明确复制步骤。
- [ ] 成功 run 后可保留域；域重建更新内部资源、拒绝旧上下文并保持定义 ID；域 A 释放后域 B 正常。
- [ ] 单 rank 准备失败可回滚；注销 callable、销毁域、关闭 Worker 都等待未完成引用，未确认停止时不释放 / 复用内存。
- [ ] 与 PR2 联调：同一 L2 至少 1000 次运行时循环，参数逐轮变化，无每轮 Host prepare；延迟完成时资源持续存活。
- [ ] A5 双卡 TMR 验证内置 Broadcast 两个 root、64 / 1024 个 FP32 元素及用户函数；按长度配置足够域窗口。

Host 单测验证打包和所有权，假后端验证准备 / 失败路径；真实窗口访问、通信完成及回收必须有 A5 硬件证据。
上板按仓库流程使用 `task-submit`。组件通过不能替代完整 Worker/TMR 双卡验收。

## 当前验证结果（2026-11-06）

- CPU 最新定向回归：26 项通过，其中 21 项完成协议测试、5 项关闭与所有权测试；完整 A5 TMR 构建及离线设备编译通过。
- 三 PR 联调版本 `05d9…`，18:47:36 提交的双卡任务（设备 2、3）：首次 send 报 101（`ASYNC_COMPLETION_INVALID`），约 7.4 秒明确失败。未再出现旧路径约 100 秒的全设备 drain 和清理段错误。这只验证本次错误清理路径，联调仍失败。
- PR1 独立版本 `ba178…` 已完成双卡 1000 轮；它绕过普通 Worker/TMR 集成，不能替代三 PR 联调验收。

[端到端 scene](../../../tests/st/a5/tensormap_and_ringbuffer/ccu_deferred_completion/README.md) 已具备四轮和 1000 轮用例：两个独立 CCU ID、位置参数、变化地址 / 长度 / 数据、完整 guard 和资源清理检查。当前仍需修复首次 send 的 101，再验证四轮及 1000 轮正常完成、AIV 返回后的依赖等待和域回收；上述验收清单尚未整体通过。

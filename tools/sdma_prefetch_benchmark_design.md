# SDMA 预取路径与 benchmark 指标

本文先说明 `sdma` 模式从脚本到运行时的路径和设计思路，再分别列出 `benchmark_rounds.sh` 和 `benchmark_avg_aicore_exec.sh` 关心的指标。

## 1. `sdma` 路径和设计思路

`benchmark_rounds.sh` / `benchmark_avg_aicore_exec.sh` 都把 `compare` 解释成两次独立运行：先 `baseline`，再 `sdma`。两次运行只差 `PTO_SDMA_PREFETCH_MODE`，其余 workload、rounds、case 保持一致，便于做 A/B。

`sdma` 的路径大致是：

```text
benchmark script
  -> run_example.py
  -> resolve_sdma_runtime_env()
  -> host_prefetch_setup()
  -> runtime_maker.cpp 读取 prefetch env
  -> AicpuExecutor::init() / aicpu_prefetch_init()
  -> 调度器按任务做资格判断
  -> issue_task_prefetch()
  -> aicpu_prefetch_issue_reserved()
  -> STARS SDMA CMO PREFETCH SQE
```

设计上有三个核心点：

- 预取是“可选增强”，不是主流程依赖项；provider 不存在或 setup 失败时，直接退化为 no-op。
- 预取决策放在调度热路径里，但真正发起前有 `min_bytes` 和 suppression window 两层门槛，避免控制开销吞掉收益。
- 任务提交阶段就写好 `prefetch_addr` / `prefetch_issue_bytes` / `prefetch_filter_bytes`，调度阶段只做轻量判定。

## 2. 当前优势

- A/B 路径干净：`baseline` 和 `sdma` 共享同一套 runtime 和 workload。
- 失败可降级：provider、workspace、channel 任一步缺失，都会回退，不影响主测试。
- 开销可拆分：能分别看 setup、control path、SQE issue、以及最终的执行时间变化。
- 适合多 workload：阈值和 suppression 机制能避免对小任务过度激进。

## 3. 下一步优化方向

- 按 workload 自适应 `min_bytes` 和 suppression window，而不是固定阈值。
- 把“选下一个任务”升级成更智能的 target selection，而不是只看 batch 内邻项。
- 细化预取成功率和命中收益的统计，反推更好的调参策略。
- 统一多轮统计口径，减少 profiling 和 device log 在轮次聚合上的差异。

## 4. `benchmark_rounds.sh` 的指标

这个脚本把指标拆成两组：

- profiling 组：固定只跑 1 次，并且带 `--enable-profiling`
- device log 组：使用命令参数 `--rounds` 跑多轮，不带 `--enable-profiling`

因此它关注的是“单次 profiling 视角的跨度”加上“多轮 device log 平均 E2E”：

| 指标 | 含义 | 数据源 |
| --- | --- | --- |
| `AICore Exec` | 所有任务中最早 `start_time_us` 到最晚 `end_time_us` 的跨度 | profiling JSON |
| `AICPU Dispatch->Finish` | 最早 `dispatch_time_us` 到最晚 `finish_time_us` 的跨度 | profiling JSON |
| `Device E2E (profiling)` | profiling 里 task / sched / orch 的最早起点到最晚终点 | profiling JSON |
| `Device E2E Avg (device log)` | device log 里每轮 `orch_start/sched_start` 到 `orch_end/sched_end` 的跨度，最后取均值 | device log |

其中：

- 前三列来自单独的 1 轮 profiling run
- 最后一列来自单独的 non-profiling run，并按 `--rounds` 求平均

它的重点是比较 `baseline` vs `sdma` 的整轮性能差异，不强调预取内部开销拆分。

## 5. `benchmark_avg_aicore_exec.sh` 的指标

这个脚本关注的是“平均单 task AICore 执行时间”和“预取开销”：

| 指标 | 含义 | 数据源 |
| --- | --- | --- |
| `Avg AICore Task Exec` | 所有 task 的 `avg(end_time_us - start_time_us)` | profiling JSON |
| `Prefetch Setup Outcome` | `host_prefetch_setup()` 的结果状态 | run 输出 |
| `Prefetch Ctrl Path Total` | 调度侧预取资格判断/控制路径总耗时 | device log |
| `Prefetch SQE Issue Total` | 真正写 SQE / doorbell 的总耗时 | device log |
| `Prefetch Ctrl Counts` | considered / eligible / 各类 skip 计数 | device log |
| `Prefetch Issue Counts` | attempts / issues / suppressed / queue_full | device log |

它更适合回答“sdma 是否把控制开销压住了，以及单 task 平均执行有没有变化”，而不是整轮 E2E。

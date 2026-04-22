# benchmark_rounds.sh 指标定义说明

本文说明 `tools/benchmark_rounds.sh` 当前输出的几个性能指标的定义、数据来源、实现方式和解读方法。

脚本每次运行一个 example/case/mode 时，会尝试同时解析两类数据：

- profiling 导出的 `outputs/perf_swimlane_*.json`
- device log 中的 `orch_*` / `sched_*` timing 行

当前会输出 4 列时间指标：

- `AICore Exec`
- `AICPU Dispatch->Finish`
- `Device E2E (profiling)`
- `Device E2E (device log)`

## 指标总览

| 指标 | 数据源 | 当前实现 | 语义 |
| --- | --- | --- | --- |
| `AICore Exec` | `perf_swimlane_*.json` 的 `tasks[].start_time_us/end_time_us` | `max(end_time_us) - min(start_time_us)` | AICore 计算窗口的总跨度 |
| `AICPU Dispatch->Finish` | `perf_swimlane_*.json` 的 `tasks[].dispatch_time_us/finish_time_us` | `max(finish_time_us) - min(dispatch_time_us)` | 从最早任务被 AICPU dispatch 到最后任务被 AICPU 观察到完成的跨度 |
| `Device E2E (profiling)` | profiling JSON 中的 task / scheduler / orchestrator 时间戳 | 所有 profiling 记录里 `latest_end - earliest_start` | 完整 profiling 视角的设备端端到端跨度，包含 orch |
| `Device E2E (device log)` | device log 的 `orch_start` / `orch_end` / `orch_stage_end` / `sched_start` / `sched_end` | 每轮 `latest_end - earliest_start`，最后对各轮求平均 | device log 视角的设备端端到端跨度 |

## 1. AICore Exec

### 定义

`AICore Exec` 定义为 profiling JSON 里所有 task 的最早 `start_time_us` 到最晚 `end_time_us` 的跨度：

```text
AICore Exec = max(tasks.end_time_us) - min(tasks.start_time_us)
```

### 数据来源

- 文件：`outputs/perf_swimlane_*.json`
- 字段：`tasks[].start_time_us`
- 字段：`tasks[].end_time_us`

### 实现位置

- `tools/benchmark_rounds.sh`
- 函数：`parse_perf_json_metrics`

### 解释

这个值表示 AICore 计算窗口的 makespan，不是：

- 所有 kernel 执行时间之和
- 平均单 task 执行时间

因此它适合回答的问题是：这批 task 在 AICore 上一共“铺开”了多久。

## 2. AICPU Dispatch->Finish

### 定义

`AICPU Dispatch->Finish` 定义为 profiling JSON 里所有 task 的最早 `dispatch_time_us` 到最晚 `finish_time_us` 的跨度：

```text
AICPU Dispatch->Finish = max(tasks.finish_time_us) - min(tasks.dispatch_time_us)
```

### 数据来源

- 文件：`outputs/perf_swimlane_*.json`
- 字段：`tasks[].dispatch_time_us`
- 字段：`tasks[].finish_time_us`

### 实现位置

- `tools/benchmark_rounds.sh`
- 函数：`parse_perf_json_metrics`

### 解释

这个指标是 AICPU 视角的执行窗口，通常包含：

- head overhead：dispatch 后到 task 真正开始执行之前的等待
- AICore Exec：task 在 AICore 上的执行
- tail overhead：task 执行完成后，到 scheduler 下一次观察到完成之间的延迟

它不包含 orch 在“第一笔 dispatch 之前”的构图时间，所以它通常会小于完整 E2E。

## 3. Device E2E (profiling)

### 定义

`Device E2E (profiling)` 定义为 profiling JSON 中所有可用的 task / scheduler / orchestrator 记录里的最早起点到最晚终点：

```text
Device E2E (profiling) =
    max(all profiling ends) - min(all profiling starts)
```

当前纳入范围的时间源包括：

- `tasks[].dispatch_time_us/finish_time_us`
- `tasks[].start_time_us/end_time_us`
- `aicpu_scheduler_phases[][ ].start_time_us/end_time_us`
- `aicpu_orchestrator_phases[][ ].start_time_us/end_time_us`
- `aicpu_orchestrator.start_time_us/end_time_us`

### 数据来源

- 文件：`outputs/perf_swimlane_*.json`
- 字段：task、scheduler phase、orchestrator phase、orchestrator summary

### 实现位置

- `tools/benchmark_rounds.sh`
- 函数：`parse_perf_json_metrics`

### 解释

这是 profiling 视角下最完整的设备端 E2E。

它的设计目标是覆盖：

- orch 构图
- scheduler 生命周期
- dispatch 到 finish 的任务执行窗口

因此它应该大于或等于 `AICPU Dispatch->Finish`。

## 4. Device E2E (device log)

### 定义

`Device E2E (device log)` 从 device log 中提取每轮的起止时间：

```text
per_round_e2e = max(orch_end, orch_stage_end, sched_end) - min(orch_start, sched_start)
```

如果同一个 log 中包含多轮，脚本会先逐轮切分，然后对各轮求平均：

```text
Device E2E (device log) = avg(per_round_e2e)
```

### 数据来源

- device log
- 关键行：
  - `orch_start=...`
  - `orch_end=...`
  - `orch_stage_end=...`
  - `sched_start=...`
  - `sched_end=...`

### 实现位置

- `tools/benchmark_rounds.sh`
- 函数：`parse_device_e2e_avg`

### 单位换算

device log 里记录的是 cycle counter，脚本按平台频率转换成微秒：

- `a2a3`: `50 MHz`
- `a5`: `1000 MHz`

### 解释

这个指标的目标语义和 `Device E2E (profiling)` 一致，都是完整设备端 E2E，只是数据源不同。

因此在单轮条件下，两者预期应当比较接近。

## 指标之间的预期关系

正常情况下应满足：

```text
AICore Exec <= AICPU Dispatch->Finish <= Device E2E (profiling)
```

并且在 `ROUNDS=1` 时，预期：

```text
Device E2E (profiling) ~= Device E2E (device log)
```

这里的 “接近” 指的是趋势和量级应接近，不要求逐微秒完全一致，因为二者的记录源不同：

- profiling 来自导出的 perf JSON
- device log 来自 AICPU 侧日志打印

## 多轮运行的当前限制

当前 `benchmark_rounds.sh` 支持 `-n/--rounds` 跑多轮，但多轮下两个 E2E 的统计口径还没有完全统一。

原因是：

- profiling 数据当前只覆盖第一个 profiled round
- device log 的 E2E 会对所有 round 求平均

因此：

- 如果要严格比较 `Device E2E (profiling)` 和 `Device E2E (device log)`，建议优先使用 `-n 1`
- 如果使用 `-n > 1`，更适合看趋势，不适合直接根据两者差值下强结论

## 回退行为

如果当前 run 没有成功定位到新的 `perf_swimlane_*.json`，脚本会对部分指标回退到旧路径：

- `AICore Exec` 回退到 `run_example.py` 输出中的 `AICore Span`
- `AICPU Dispatch->Finish` 回退到 `run_example.py` 输出中的 `Total Test Time`

注意：

- `Device E2E (profiling)` 没有同等语义的旧输出回退值
- 因为 `Total Test Time` 只表示 `dispatch -> finish`，它不包含 orch

## 推荐使用方式

如果当前目标是先看指标是否稳定、两个 E2E 是否相近，推荐：

```bash
./tools/benchmark_rounds.sh -n 1
```

等观察完结果后，再决定是否需要进一步调整：

- 统一为“全都只看首轮”
- 或统一为“全都做多轮平均”
- 或分别保留“首轮 profiling”和“多轮 device log”两种视角

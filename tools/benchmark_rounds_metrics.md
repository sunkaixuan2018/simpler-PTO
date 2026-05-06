# benchmark_rounds.sh 指标定义说明

本文说明 `tools/benchmark_rounds.sh` 当前输出的几个性能指标的定义、数据来源、采样方式和解读方法。

## 运行方式

脚本每次运行一个 example/case/mode 时，会拆成两次子运行：

- profiling run：固定 `-n 1`，显式带 `--enable-profiling`
- device-log run：使用命令行传入的 `-n/--rounds`，且不带 `--enable-profiling`

因此脚本会同时读取两类数据：

- profiling 导出的 `outputs/perf_swimlane_*.json`
- device log 中的 `orch_*` / `sched_*` timing 行

当前会输出 4 列时间指标：

- `AICore Exec`
- `AICPU Dispatch->Finish`
- `Device E2E (profiling)`
- `Device E2E Avg (device log)`

## 指标总览

| 指标 | 数据源 | 采样方式 | 当前实现 | 语义 |
| --- | --- | --- | --- | --- |
| `AICore Exec` | `perf_swimlane_*.json` 的 `tasks[].start_time_us/end_time_us` | profiling run，固定 1 轮 | `max(end_time_us) - min(start_time_us)` | AICore 计算窗口的总跨度 |
| `AICPU Dispatch->Finish` | `perf_swimlane_*.json` 的 `tasks[].dispatch_time_us/finish_time_us` | profiling run，固定 1 轮 | `max(finish_time_us) - min(dispatch_time_us)` | 从最早任务被 AICPU dispatch 到最后任务被 AICPU 观察到完成的跨度 |
| `Device E2E (profiling)` | profiling JSON 中的 task / scheduler / orchestrator 时间戳 | profiling run，固定 1 轮 | 所有 profiling 记录里 `latest_end - earliest_start` | 完整 profiling 视角的设备端端到端跨度，包含 orch |
| `Device E2E Avg (device log)` | device log 的 `orch_start` / `orch_end` / `orch_stage_end` / `sched_start` / `sched_end` | device-log run，轮数来自 `-n/--rounds` | 每轮 `latest_end - earliest_start`，最后对各轮求平均 | device log 视角的设备端端到端平均跨度 |

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

### 采样方式

- 来自 profiling run
- profiling run 固定只跑 1 轮
- profiling run 显式带 `--enable-profiling`

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

### 采样方式

- 来自 profiling run
- profiling run 固定只跑 1 轮
- profiling run 显式带 `--enable-profiling`

## 3. Device E2E (profiling)

### 定义

`Device E2E (profiling)` 定义为 profiling JSON 中所有可用的 task / scheduler / orchestrator 记录里的最早起点到最晚终点：

```text
Device E2E (profiling) =
    max(all profiling ends) - min(all profiling starts)
```

### 数据来源

- 文件：`outputs/perf_swimlane_*.json`
- 字段：task、scheduler phase、orchestrator phase、orchestrator summary

### 采样方式

- 来自 profiling run
- profiling run 固定只跑 1 轮
- profiling run 显式带 `--enable-profiling`

## 4. Device E2E Avg (device log)

### 定义

`Device E2E Avg (device log)` 从 device log 中提取每轮的起止时间：

```text
per_round_e2e = max(orch_end, orch_stage_end, sched_end) - min(orch_start, sched_start)
```

随后对这次 device-log run 中的各轮求平均：

```text
Device E2E Avg (device log) = avg(per_round_e2e)
```

### 数据来源

- device log
- 关键行：
  - `orch_start=...`
  - `orch_end=...`
  - `orch_stage_end=...`
  - `sched_start=...`
  - `sched_end=...`

### 采样方式

- 来自 device-log run
- 轮数由命令行 `-n/--rounds` 决定
- device-log run 不带 `--enable-profiling`

## 推荐使用方式

- 如果要严格对齐 profiling E2E 和 device-log E2E，建议用 `-n 1`
- 如果要看更稳定的 device-log 平均 E2E，可以增大 `-n`

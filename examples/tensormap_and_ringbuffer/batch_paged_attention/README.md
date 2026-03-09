# batch_paged_attention（tensormap_and_ringbuffer）

本用例支持通过 **AICPU 绑核/亲和性** 来复用 `DAV_3510` / `DAV_2201` 两种分配策略。

## 如何启用

在 `kernels/kernel_config.py` 中设置：

- `RUNTIME_ENV["PTO2_AICPU_AFFINITY_MODE"] = "3510"`：按 die 分组（DAV_3510）
- `RUNTIME_ENV["PTO2_AICPU_AFFINITY_MODE"] = "2201"`：按 4-CPU cluster 收敛（DAV_2201）
- `off`：关闭（默认）

## 两种策略对应的行为

- **DAV_3510（die 分组）**
  - scheduler 线程（默认 3 个：thread 0/1/2）按数量对半分到 die0 / die1
  - die0：绑到 `cpu ∈ [0, max_cpu_id/2]`
  - die1：绑到 `cpu ∈ [max_cpu_id/2+1, max_cpu_id]`

- **DAV_2201（cluster 收敛）**
  - 所有 AICPU 线程先上报启动时所在 CPU（`sched_getcpu()`），从中选择第一个“4-CPU cluster 内可容纳 scheduler 线程数”的 cluster
  - scheduler 线程绑到该 cluster 的 4 个 CPU

## 运行

```bash
python3 examples/scripts/run_example.py \
  -k examples/tensormap_and_ringbuffer/batch_paged_attention/kernels \
  -g examples/tensormap_and_ringbuffer/batch_paged_attention/golden.py \
  -p a2a3sim
```


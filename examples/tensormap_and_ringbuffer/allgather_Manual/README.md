# AllGather (Manual) - tensormap_and_ringbuffer

直接 RDMA 读取的 AllGather，无 TGATHER。用于性能对比。

流程：WindowMemCopyIn → CommBarrier(pre) → AllGatherManual → WindowMemCopyOut → CommBarrier(post)

所有 rank 并行执行，每个 rank 获得完整拼接结果。

## 运行

```bash
cd simpler-PTO
./run_tensormap.sh allgather_Manual 2 0
```

需要设置 `PTO_COMM_ISA_ROOT` 及多卡 HCCL 环境。

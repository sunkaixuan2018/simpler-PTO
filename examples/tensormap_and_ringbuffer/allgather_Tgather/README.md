# AllGather (TGATHER) - tensormap_and_ringbuffer

N 次顺序 TGATHER 实现 AllGather，用于性能对比。

流程：WindowMemCopyIn -> for r in [0,n_ranks): Barrier -> Gather(root=r) -> [rank==r: WindowMemCopyOut] -> Barrier(post)

每轮仅 root 调用 TGATHER，避免死锁。

## 运行

```bash
cd simpler-PTO
./run_tensormap.sh allgather_Tgather 2 0
```

需要设置 `PTO_COMM_ISA_ROOT` 及多卡 HCCL 环境。

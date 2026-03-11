# Paged Attention + AllGather (TGATHER) - tensormap_and_ringbuffer

Paged Attention 计算后 AllGather（N 轮 TGATHER）。

流程：Paged Attention (QK->SF->PV->UP) -> WindowMemCopyIn -> for r in [0,n_ranks):
Barrier -> Gather(root=r) -> [rank r: WindowMemCopyOut] -> Barrier(post)

## 运行

```bash
./run_tensormap.sh paged_attention_allgather_Tgather 2 0
```

注意：编排文件 (paged_attention_allgather_orch.cpp) 需要完整实现。当前仅包含 kernel_config 和 golden，编排需根据 runtime 接口补充。

# Paged Attention + AllGather (Manual) - tensormap_and_ringbuffer

Paged Attention 计算后 AllGather（直接 RDMA）。

流程：Paged Attention (QK->SF->PV->UP) -> WindowMemCopyIn -> CommBarrier
-> AllGatherManual -> WindowMemCopyOut -> CommBarrier(post)

## 运行

```bash
./run_tensormap.sh paged_attention_allgather_Manual 2 0
```

注意：编排文件 (paged_attention_allgather_orch.cpp) 需要完整实现，将 Phase 1（Paged Attention）与 Phase 2（AllGather）组合。当前仅包含 kernel_config 和 golden，编排需根据 runtime 接口补充。

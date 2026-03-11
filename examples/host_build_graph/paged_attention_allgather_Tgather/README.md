# Paged Attention + AllGather (TGATHER) - host_build_graph

Paged Attention 计算后 AllGather（N 轮 TGATHER）。

流程：QK → Softmax → PV → OnlineUpdate → WindowMemCopyIn
→ for r in [0,n_ranks): Barrier → Gather(root=r) → [rank r: WindowMemCopyOut] → Barrier(post)

## 运行

```bash
./run_hostbuild.sh paged_attention_allgather_Tgather 2 0
```

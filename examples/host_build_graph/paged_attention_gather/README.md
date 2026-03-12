# Paged Attention + Gather (host_build_graph)

Paged Attention 计算后 TGATHER。Root 收集各 rank 的 attn_out 前 GATHER_COUNT 元素。

流程：QK → Softmax → PV → OnlineUpdate → WindowMemCopyIn → CommBarrier → TGATHER → WindowMemCopyOut (root only)

## 运行

```bash
./run_hostbuild.sh paged_attention_gather 2 0
```

# Paged Attention + AllGather (Manual) - host_build_graph

Paged Attention 计算后 AllGather（直接 RDMA 读取）。

流程：QK → Softmax → PV → OnlineUpdate → WindowMemCopyIn → CommBarrier(pre)
→ AllGatherManual → WindowMemCopyOut → CommBarrier(post)

所有 rank 获得完整 allgather 输出。

## 运行

```bash
./run_hostbuild.sh paged_attention_allgather_Manual 2 0
```

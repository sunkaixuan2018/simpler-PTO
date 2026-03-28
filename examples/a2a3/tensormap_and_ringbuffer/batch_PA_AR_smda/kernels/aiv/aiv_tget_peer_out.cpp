/**
 * func_id 7 — 同步 TGET：收到指定源 rank 就绪后，将其当前 chunk `out` 拉取到本地 `peer_out`。
 *
 * args[0] notify_done Tensor*, args[1] out_chunk Tensor* (仅用于计算远端窗口偏移), args[2] peer_out_chunk Tensor* (output),
 * args[3] CommDeviceContext*, args[4] total_elems (int64), args[5] source_rank (int64)
 */

#include <cstdint>
#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "pto/comm/pto_comm_inst.hpp"

#include "common/comm_context.h"
#include "tensor.h"

using namespace pto;

template <typename T>
AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)local_ptr - local_base;
    return (__gm__ T*)(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* nd_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ Tensor* peer_tensor = reinterpret_cast<__gm__ Tensor*>(args[2]);
    __gm__ CommDeviceContext* comm_ctx = reinterpret_cast<__gm__ CommDeviceContext*>(args[3]);
    int total_elems = static_cast<int>(args[4]);
    int source_rank = static_cast<int>(args[5]);

    // notify_done 只用于编排层建立任务依赖，不在核内解引用，避免对 shape=1 的临时 tensor 越界读。
    (void)nd_tensor;
    __gm__ float* out_data =
        reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;
    __gm__ float* peer_data =
        reinterpret_cast<__gm__ float*>(peer_tensor->buffer.addr) + peer_tensor->start_offset;

    int nranks = static_cast<int>(comm_ctx->rankNum);
    if (nranks <= 1 || source_rank < 0 || source_rank >= nranks || source_rank == static_cast<int>(comm_ctx->rankId)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using FlatGlobalData = GlobalTensor<float, ShapeDyn, StrideDyn>;

    ShapeDyn shape(1, 1, 1, 1, total_elems);
    StrideDyn stride(total_elems, total_elems, total_elems, total_elems, 1);
    FlatGlobalData peerGlobalFlat(peer_data, shape, stride);
    __gm__ float* remote_out = CommRemotePtr(comm_ctx, out_data, source_rank);
    FlatGlobalData remoteGlobalFlat(remote_out, shape, stride);

    constexpr int kTGetTileElems = 256;
    using StagingTile = Tile<TileType::Vec, float, 1, kTGetTileElems, BLayout::RowMajor, -1, -1>;
    StagingTile stagingTile(1, kTGetTileElems);
    TASSIGN(stagingTile, 0x0);

    pto::comm::TGET(peerGlobalFlat, remoteGlobalFlat, stagingTile);
    pipe_barrier(PIPE_ALL);
}

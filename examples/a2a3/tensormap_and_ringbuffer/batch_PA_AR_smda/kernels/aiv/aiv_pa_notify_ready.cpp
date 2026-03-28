/**
 * func_id 6 — 读本地 chunk `out`，TNOTIFY 指定对端对应 chunk 的槽位，写 `notify_done`（1 float），
 * 保证后续 TGET 仅在该 chunk 已通知后才被调度。
 *
 * args[0] out_chunk Tensor* (input), args[1] notify_done Tensor* (output),
 * args[2] local notify_counter_slot (scalar addr), args[3] CommDeviceContext* (scalar addr),
 * args[4] target_rank (int64)
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

#include "common/comm_context.h"
#include "tensor.h"

using namespace pto;

#include "pto_notify_kernel_api.h"

template <typename T>
AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)local_ptr - local_base;
    return (__gm__ T*)(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* done_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ int32_t* local_counter = reinterpret_cast<__gm__ int32_t*>(args[2]);
    __gm__ CommDeviceContext* comm_ctx = reinterpret_cast<__gm__ CommDeviceContext*>(args[3]);
    int target_rank = static_cast<int>(args[4]);

    __gm__ float* out_data =
        reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;
    __gm__ float* done_data =
        reinterpret_cast<__gm__ float*>(done_tensor->buffer.addr) + done_tensor->start_offset;

    int nranks = static_cast<int>(comm_ctx->rankNum);
    if (nranks <= 1 || target_rank < 0 || target_rank >= nranks || target_rank == static_cast<int>(comm_ctx->rankId)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    using FlatShape = Shape<1, 1, 1, 1, 16>;
    using FlatStride = Stride<16, 16, 16, 16, 1>;
    using FlatGlobal = GlobalTensor<float, FlatShape, FlatStride>;
    using TileData = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1>;

    TileData tile(1, 16);
    TASSIGN(tile, 0x0);
    FlatGlobal outG(out_data);
    TLOAD(tile, outG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    __gm__ int32_t* remote_counter = CommRemotePtr(comm_ctx, local_counter, target_rank);
    pto2_send_notification(remote_counter, 1, PTO2NotifyOp::AtomicAdd);
    pipe_barrier(PIPE_ALL);

    done_data[0] = 1.0f;
}

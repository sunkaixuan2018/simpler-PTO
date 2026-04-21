/**
 * Final allreduce add: read every rank's local_out directly from remote window
 * memory and write the element-wise sum to out.
 *
 * args[0] = &Tensor(local_out)
 * args[1] = &Tensor(notify_done) -- dependency only
 * args[2] = &Tensor(out)
 * args[3] = CommDeviceContext*
 * args[4] = total element count
 * args[5] = local notify counter
 * args[6] = expected notify count
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include <pto/comm/pto_comm_inst.hpp>

#include "common/comm_context.h"
#include "tensor.h"

using namespace pto;

template <typename T>
AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* local_ptr,
                                      int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)local_ptr - local_base;
    return (__gm__ T*)(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* local_out_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
    (void)reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[2]);
    __gm__ CommDeviceContext* comm_ctx = reinterpret_cast<__gm__ CommDeviceContext*>(args[3]);
    uint64_t total_elems = static_cast<uint64_t>(args[4]);
    __gm__ int32_t* local_counter = reinterpret_cast<__gm__ int32_t*>(args[5]);
    int32_t expected_count = static_cast<int32_t>(args[6]);

    __gm__ float* local_out =
        reinterpret_cast<__gm__ float*>(local_out_tensor->buffer.addr) + local_out_tensor->start_offset;
    __gm__ float* out =
        reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;

    if (comm_ctx == nullptr || comm_ctx->rankNum <= 1) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    int nranks = static_cast<int>(comm_ctx->rankNum);

    pto::comm::Signal ready(local_counter);
    pto::comm::TWAIT(ready, expected_count, pto::comm::WaitCmp::GE);

    if (total_elems == 0 || total_elems > 16 * 16 * 16) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    constexpr int kChunkElems = 256;
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GlobalData = GlobalTensor<float, ShapeDyn, StrideDyn, Layout::ND>;
    using TileData = Tile<TileType::Vec, float, 1, kChunkElems, BLayout::RowMajor, -1, -1>;

    for (uint64_t offset = 0; offset < total_elems; offset += kChunkElems) {
        uint64_t tile_elems = total_elems - offset;
        if (tile_elems > (uint64_t)kChunkElems) tile_elems = kChunkElems;

        ShapeDyn shape(1, 1, 1, 1, tile_elems);
        StrideDyn stride(tile_elems, tile_elems, tile_elems, tile_elems, 1);

        TileData acc_tile(1, tile_elems);
        TileData recv_tile(1, tile_elems);
        TASSIGN(acc_tile, 0x0);
        TASSIGN(recv_tile, 0x10000);

        __gm__ float* rank0_ptr = CommRemotePtr(comm_ctx, local_out + offset, 0);
        GlobalData rank0_global(rank0_ptr, shape, stride);
        GlobalData out_global(out + offset, shape, stride);

        TLOAD(acc_tile, rank0_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        for (int rank = 1; rank < nranks; ++rank) {
            __gm__ float* remote_ptr = CommRemotePtr(comm_ctx, local_out + offset, rank);
            GlobalData remote_global(remote_ptr, shape, stride);
            TLOAD(recv_tile, remote_global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            TADD(acc_tile, acc_tile, recv_tile);
            if (rank + 1 < nranks) {
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            }
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TSTORE(out_global, acc_tile);
        if (offset + tile_elems < total_elems) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

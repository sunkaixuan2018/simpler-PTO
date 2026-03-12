/**
 * Manual AllGather kernel - direct RDMA reads, no TGATHER.
 *
 * Each rank independently reads from all ranks' win_src via HcclRemotePtr
 * and writes to local dst. All ranks run in parallel.
 *
 * Tensormap_and_ringbuffer: args are TensorData* for buffers, scalars for ctx/n_ranks/rank_id.
 *   args[0] = dst (TensorData*)
 *   args[1] = src (TensorData*)
 *   args[2] = sync_done (TensorData*, dependency - ignored)
 *   args[3] = device_ctx_ptr (scalar)
 *   args[4] = nranks (scalar)
 *   args[5] = rank_id (scalar, unused)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include "hccl_context.h"
#include "hccl_helpers.h"

#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr size_t GATHER_COUNT = 64;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* dst_td = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* src_td = reinterpret_cast<__gm__ TensorData*>(args[1]);
    (void)args[2];
    __gm__ HcclDeviceContext* hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext*>(args[3]);
    int nranks = static_cast<int>(args[4]);
    (void)args[5];

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_td->buffer.addr);
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_td->buffer.addr);

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, GATHER_COUNT, pto::BLayout::RowMajor, -1, -1>;

    ShapeDyn sliceShape(1, 1, 1, 1, GATHER_COUNT);
    StrideDyn sliceStride(GATHER_COUNT, GATHER_COUNT, GATHER_COUNT, GATHER_COUNT, 1);

    TileData ubTile(1, GATHER_COUNT);
    TASSIGN(ubTile, 0x0);

    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int r = 0; r < actual_nranks; ++r) {
        __gm__ float* remote_src = HcclRemotePtr(hcclCtx, src, r);
        __gm__ float* local_dst = dst + static_cast<ptrdiff_t>(r) * GATHER_COUNT;

        Global srcG(remote_src, sliceShape, sliceStride);
        Global dstG(local_dst, sliceShape, sliceStride);

        TLOAD(ubTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, ubTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }

    pipe_barrier(PIPE_ALL);
}

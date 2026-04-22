#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>

#include "comm_utils.h"
#include "tensor.h"

using namespace pto;

static constexpr uint64_t kChunkElems = 16 * 1024;

using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
using GlobalData = GlobalTensor<float, ShapeDyn, StrideDyn, Layout::ND>;
using TileData = Tile<TileType::Vec, float, 1, kChunkElems, BLayout::RowMajor, -1, -1>;

__aicore__ __attribute__((always_inline)) inline void CopyRemoteSpan(
    __gm__ float* dst, __gm__ float* src, uint64_t elem_count) {
    for (uint64_t off = 0; off < elem_count; off += kChunkElems) {
        uint64_t chunk = elem_count - off;
        if (chunk > kChunkElems) {
            chunk = kChunkElems;
        }

        ShapeDyn shape(1, 1, 1, 1, chunk);
        StrideDyn stride(chunk, chunk, chunk, chunk, 1);
        GlobalData src_g(src + off, shape, stride);
        GlobalData dst_g(dst + off, shape, stride);
        TileData tile(1, chunk);
        TASSIGN(tile, 0x0);

        TLOAD(tile, src_g);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

        TSTORE(dst_g, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    if (get_block_idx() != 0) {
        return;
    }

    __gm__ Tensor* dst_t = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* src_t = reinterpret_cast<__gm__ Tensor*>(args[1]);
    (void)reinterpret_cast<__gm__ Tensor*>(args[2]);
    __gm__ Tensor* debug_t = reinterpret_cast<__gm__ Tensor*>(args[3]);
    __gm__ CommDeviceContext* comm_ctx = reinterpret_cast<__gm__ CommDeviceContext*>(args[4]);
    int n_ranks = static_cast<int>(args[5]);
    (void)args[6];
    (void)args[7];

    if (comm_ctx == nullptr || n_ranks <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_t->buffer.addr) + dst_t->start_offset;
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_t->buffer.addr) + src_t->start_offset;
    __gm__ int32_t* debug =
        reinterpret_cast<__gm__ int32_t*>(debug_t->buffer.addr) + debug_t->start_offset;

    uint64_t gather_count = src_t->buffer.size / sizeof(float);
    int actual_ranks = n_ranks;
    if (actual_ranks > static_cast<int>(COMM_MAX_RANK_NUM)) {
        actual_ranks = static_cast<int>(COMM_MAX_RANK_NUM);
    }

    for (int rank = 0; rank < actual_ranks; ++rank) {
        __gm__ float* remote_src = CommRemotePtr(comm_ctx, src, rank);
        CopyRemoteSpan(dst + static_cast<uint64_t>(rank) * gather_count, remote_src, gather_count);
        debug[rank] = 1;
    }

    pipe_barrier(PIPE_ALL);
}

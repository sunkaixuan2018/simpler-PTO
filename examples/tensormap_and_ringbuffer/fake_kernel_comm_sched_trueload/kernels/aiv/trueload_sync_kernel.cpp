/**
 * TrueLoadSync: explicit MTE remote-copy kernel.
 *
 * Root reads each rank's source buffer via TLOAD and writes it to the local
 * destination buffer via TSTORE. No TGATHER collective is involved.
 *
 * Args (8):
 *   args[0] = dst (TensorData*)
 *   args[1] = src (TensorData*)
 *   args[2] = sync_done (TensorData*, dependency only)
 *   args[3] = device_ctx_ptr (scalar)
 *   args[4] = nranks (scalar)
 *   args[5] = root (scalar)
 *   args[6] = sdma_workspace_ptr (scalar, unused by sync variant)
 *   args[7] = debug_poll_counts (TensorData*, unused by sync variant)
 */

#include <cstddef>
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

static constexpr size_t TRUELOAD_STAGE_CHUNK = 128 * 128;
static constexpr uint64_t TRUELOAD_PING_TILE_ADDR = 0x0;
static constexpr uint64_t TRUELOAD_PONG_TILE_ADDR = TRUELOAD_STAGE_CHUNK * sizeof(float);

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
using TileData = pto::Tile<pto::TileType::Vec, float, 1, TRUELOAD_STAGE_CHUNK, pto::BLayout::RowMajor, -1, -1>;

__aicore__ __attribute__((always_inline)) inline event_t GetTrueLoadEvent(int slot)
{
    return (slot == 0) ? EVENT_ID0 : EVENT_ID1;
}

__aicore__ __attribute__((always_inline)) inline uint64_t GetTrueLoadTileAddr(int slot)
{
    return (slot == 0) ? TRUELOAD_PING_TILE_ADDR : TRUELOAD_PONG_TILE_ADDR;
}

__aicore__ __attribute__((always_inline)) inline void TrueLoadChunk(__gm__ float *remote_src, size_t off,
                                                                    size_t chunk, int slot)
{
    TileData tile(1, chunk);
    TASSIGN(tile, GetTrueLoadTileAddr(slot));

    ShapeDyn shape(1, 1, 1, 1, chunk);
    StrideDyn stride(chunk, chunk, chunk, chunk, 1);
    Global srcG(remote_src + off, shape, stride);

    TLOAD(tile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, GetTrueLoadEvent(slot));
}

__aicore__ __attribute__((always_inline)) inline void TrueStoreChunk(__gm__ float *local_dst, size_t off,
                                                                     size_t chunk, int slot)
{
    TileData tile(1, chunk);
    TASSIGN(tile, GetTrueLoadTileAddr(slot));

    ShapeDyn shape(1, 1, 1, 1, chunk);
    StrideDyn stride(chunk, chunk, chunk, chunk, 1);
    Global dstG(local_dst + off, shape, stride);

    wait_flag(PIPE_MTE2, PIPE_MTE3, GetTrueLoadEvent(slot));
    TSTORE(dstG, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, GetTrueLoadEvent(slot));
}

__aicore__ __attribute__((always_inline)) inline void CopySpanPingPong(__gm__ float *local_dst,
                                                                       __gm__ float *remote_src,
                                                                       size_t elem_count)
{
    if (elem_count == 0) {
        return;
    }

    int pending_slot = -1;
    size_t pending_off = 0;
    size_t pending_chunk = 0;
    int next_slot = 0;

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    for (size_t off = 0; off < elem_count; off += TRUELOAD_STAGE_CHUNK) {
        size_t chunk = elem_count - off;
        if (chunk > TRUELOAD_STAGE_CHUNK) {
            chunk = TRUELOAD_STAGE_CHUNK;
        }

        wait_flag(PIPE_MTE3, PIPE_MTE2, GetTrueLoadEvent(next_slot));
        TrueLoadChunk(remote_src, off, chunk, next_slot);

        if (pending_slot >= 0) {
            TrueStoreChunk(local_dst, pending_off, pending_chunk, pending_slot);
        }

        pending_slot = next_slot;
        pending_off = off;
        pending_chunk = chunk;
        next_slot ^= 1;
    }

    if (pending_slot >= 0) {
        TrueStoreChunk(local_dst, pending_off, pending_chunk, pending_slot);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* dst_td = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* src_td = reinterpret_cast<__gm__ TensorData*>(args[1]);
    (void)args[2];
    __gm__ HcclDeviceContext* hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext*>(args[3]);
    int nranks = static_cast<int>(args[4]);
    int root = static_cast<int>(args[5]);
    (void)args[6];
    (void)args[7];

    int my_rank = static_cast<int>(hcclCtx->rankId);
    if (my_rank != root) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_td->buffer.addr);
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_td->buffer.addr);

    size_t src_count = src_td->buffer.size / sizeof(float);
    size_t dst_count = dst_td->buffer.size / sizeof(float);
    if (src_count == 0 || dst_count < src_count) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    int actual_nranks = (nranks > 16) ? 16 : nranks;
    size_t max_ranks_from_dst = dst_count / src_count;
    if (static_cast<size_t>(actual_nranks) > max_ranks_from_dst) {
        actual_nranks = static_cast<int>(max_ranks_from_dst);
    }

    for (int r = 0; r < actual_nranks; ++r) {
        __gm__ float* remote_src = (r == my_rank) ? src : HcclRemotePtr(hcclCtx, src, r);
        __gm__ float* local_dst = dst + static_cast<ptrdiff_t>(r) * src_count;
        CopySpanPingPong(local_dst, remote_src, src_count);
    }

    pipe_barrier(PIPE_ALL);
}

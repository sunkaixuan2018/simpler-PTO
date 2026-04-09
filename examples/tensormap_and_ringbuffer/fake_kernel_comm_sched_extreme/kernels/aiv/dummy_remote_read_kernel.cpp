/**
 * Dummy background-flow kernel for extreme sched debug.
 *
 * This kernel intentionally creates cross-rank communication pressure without
 * invoking collective gather primitives (TGATHER/TGET_ASYNC), to isolate
 * scheduler/concurrency behavior from collective re-entrancy issues.
 *
 * The transfer itself is implemented as an explicit MTE copy pipeline:
 * remote GM -> UB via TLOAD, then UB -> local GM via TSTORE. A ping-pong
 * pair of UB tiles keeps the background flow on the MTE path while allowing
 * consecutive load/store stages to overlap.
 *
 * Args (8, same layout as gather kernels):
 *   args[0] = dst (TensorData*)
 *   args[1] = src (TensorData*)
 *   args[2] = sync_done (TensorData*, dependency only)
 *   args[3] = device_ctx_ptr (scalar)
 *   args[4] = nranks (scalar)
 *   args[5] = root (scalar)
 *   args[6] = dummy_comm_bytes (scalar, total target bytes; large legacy values
 *             still fall back to the default 16MB traffic)
 *   args[7] = debug_poll_counts (TensorData*, unused)
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

// Use 64KB UB staging tiles for the MTE pipeline, but expose a larger
// logical communication pattern: 1MB ping + 1MB pong regions with a default
// 16MB total transfer budget per dummy task.
static constexpr size_t DUMMY_STAGE_CHUNK = 128 * 128;
static constexpr size_t DUMMY_PINGPONG_BYTES = 1 * 1024 * 1024;
static constexpr size_t DUMMY_PINGPONG_ELEMS = DUMMY_PINGPONG_BYTES / sizeof(float);
static constexpr uint64_t DUMMY_TOTAL_BYTES_DEFAULT = 16 * 1024 * 1024;
static constexpr uint64_t DUMMY_TOTAL_BYTES_MAX = 64 * 1024 * 1024;
static constexpr uint64_t DUMMY_STAGE_PING_TILE_ADDR = 0x0;
static constexpr uint64_t DUMMY_STAGE_PONG_TILE_ADDR = DUMMY_STAGE_CHUNK * sizeof(float);

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
using TileData = pto::Tile<pto::TileType::Vec, float, 1, DUMMY_STAGE_CHUNK, pto::BLayout::RowMajor, -1, -1>;

__aicore__ __attribute__((always_inline)) inline event_t GetDummyEvent(int slot)
{
    return (slot == 0) ? EVENT_ID0 : EVENT_ID1;
}

__aicore__ __attribute__((always_inline)) inline uint64_t GetDummyTileAddr(int slot)
{
    return (slot == 0) ? DUMMY_STAGE_PING_TILE_ADDR : DUMMY_STAGE_PONG_TILE_ADDR;
}

__aicore__ __attribute__((always_inline)) inline void DummyLoadChunk(__gm__ float *remote_src, size_t off,
                                                                     size_t chunk, int slot)
{
    TileData tile(1, chunk);
    TASSIGN(tile, GetDummyTileAddr(slot));

    ShapeDyn shape(1, 1, 1, 1, chunk);
    StrideDyn stride(chunk, chunk, chunk, chunk, 1);
    Global srcG(remote_src + off, shape, stride);

    TLOAD(tile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, GetDummyEvent(slot));
}

__aicore__ __attribute__((always_inline)) inline void DummyStoreChunk(__gm__ float *local_dst, size_t off,
                                                                      size_t chunk, int slot)
{
    TileData tile(1, chunk);
    TASSIGN(tile, GetDummyTileAddr(slot));

    ShapeDyn shape(1, 1, 1, 1, chunk);
    StrideDyn stride(chunk, chunk, chunk, chunk, 1);
    Global dstG(local_dst + off, shape, stride);

    wait_flag(PIPE_MTE2, PIPE_MTE3, GetDummyEvent(slot));
    TSTORE(dstG, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, GetDummyEvent(slot));
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

    for (size_t off = 0; off < elem_count; off += DUMMY_STAGE_CHUNK) {
        size_t chunk = elem_count - off;
        if (chunk > DUMMY_STAGE_CHUNK) {
            chunk = DUMMY_STAGE_CHUNK;
        }

        wait_flag(PIPE_MTE3, PIPE_MTE2, GetDummyEvent(next_slot));
        DummyLoadChunk(remote_src, off, chunk, next_slot);

        if (pending_slot >= 0) {
            DummyStoreChunk(local_dst, pending_off, pending_chunk, pending_slot);
        }

        pending_slot = next_slot;
        pending_off = off;
        pending_chunk = chunk;
        next_slot ^= 1;
    }

    if (pending_slot >= 0) {
        DummyStoreChunk(local_dst, pending_off, pending_chunk, pending_slot);
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
    uint64_t dummy_comm_bytes = static_cast<uint64_t>(args[6]);
    (void)args[7];

    // Backward compatibility:
    // - old callers may still pass sdma_workspace_ptr here (large address).
    // - invalid/zero values also fall back to the default 16MB traffic budget.
    if (dummy_comm_bytes == 0 || dummy_comm_bytes > DUMMY_TOTAL_BYTES_MAX) {
        dummy_comm_bytes = DUMMY_TOTAL_BYTES_DEFAULT;
    }

    int my_rank = static_cast<int>(hcclCtx->rankId);
    if (my_rank != root) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_td->buffer.addr);
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_td->buffer.addr);

    size_t src_count = src_td->buffer.size / sizeof(float);
    size_t dst_count = dst_td->buffer.size / sizeof(float);
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    if (src_count == 0 || dst_count == 0 || actual_nranks <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    size_t total_count = static_cast<size_t>((dummy_comm_bytes + sizeof(float) - 1) / sizeof(float));
    size_t region_count = DUMMY_PINGPONG_ELEMS;
    size_t half_dst_count = dst_count / 2;
    if (half_dst_count > 0 && region_count > half_dst_count) {
        region_count = half_dst_count;
    } else if (half_dst_count == 0 && region_count > dst_count) {
        region_count = dst_count;
    }
    if (region_count == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    bool has_two_regions = dst_count >= (2 * region_count);
    size_t remaining = total_count;
    size_t rank_cursor = 0;
    size_t region_cursor = 0;

    while (remaining > 0) {
        size_t region_base = (has_two_regions && ((region_cursor & 1) != 0)) ? region_count : 0;
        size_t region_target = remaining;
        if (region_target > region_count) {
            region_target = region_count;
        }

        size_t region_written = 0;
        while (region_written < region_target) {
            int rank = static_cast<int>(rank_cursor % static_cast<size_t>(actual_nranks));
            __gm__ float* remote_src = HcclRemotePtr(hcclCtx, src, rank);
            size_t copy_count = region_target - region_written;
            if (copy_count > src_count) {
                copy_count = src_count;
            }

            CopySpanPingPong(dst + region_base + region_written, remote_src, copy_count);
            region_written += copy_count;
            ++rank_cursor;
        }

        remaining -= region_target;
        ++region_cursor;
    }

    pipe_barrier(PIPE_ALL);
}

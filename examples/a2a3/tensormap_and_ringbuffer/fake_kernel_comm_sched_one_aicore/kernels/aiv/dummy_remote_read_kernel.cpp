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

static constexpr uint64_t kStageChunk = 16 * 1024;
static constexpr uint64_t kPingPongBytes = 1 * 1024 * 1024;
static constexpr uint64_t kPingPongElems = kPingPongBytes / sizeof(float);
static constexpr uint64_t kDefaultTotalBytes = 16 * 1024 * 1024;
static constexpr uint64_t kMaxTotalBytes = 64 * 1024 * 1024;
static constexpr uint64_t kPingTileAddr = 0x0;
static constexpr uint64_t kPongTileAddr = kStageChunk * sizeof(float);

using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
using GlobalData = GlobalTensor<float, ShapeDyn, StrideDyn, Layout::ND>;
using TileData = Tile<TileType::Vec, float, 1, kStageChunk, BLayout::RowMajor, -1, -1>;

__aicore__ __attribute__((always_inline)) inline event_t DummyEvent(int slot) {
    return slot == 0 ? EVENT_ID0 : EVENT_ID1;
}

__aicore__ __attribute__((always_inline)) inline uint64_t DummyTileAddr(int slot) {
    return slot == 0 ? kPingTileAddr : kPongTileAddr;
}

__aicore__ __attribute__((always_inline)) inline void DummyLoad(
    __gm__ float* remote_src, uint64_t off, uint64_t chunk, int slot) {
    TileData tile(1, chunk);
    TASSIGN(tile, DummyTileAddr(slot));

    ShapeDyn shape(1, 1, 1, 1, chunk);
    StrideDyn stride(chunk, chunk, chunk, chunk, 1);
    GlobalData src_g(remote_src + off, shape, stride);

    TLOAD(tile, src_g);
    set_flag(PIPE_MTE2, PIPE_MTE3, DummyEvent(slot));
}

__aicore__ __attribute__((always_inline)) inline void DummyStore(
    __gm__ float* local_dst, uint64_t off, uint64_t chunk, int slot) {
    TileData tile(1, chunk);
    TASSIGN(tile, DummyTileAddr(slot));

    ShapeDyn shape(1, 1, 1, 1, chunk);
    StrideDyn stride(chunk, chunk, chunk, chunk, 1);
    GlobalData dst_g(local_dst + off, shape, stride);

    wait_flag(PIPE_MTE2, PIPE_MTE3, DummyEvent(slot));
    TSTORE(dst_g, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, DummyEvent(slot));
}

__aicore__ __attribute__((always_inline)) inline void CopySpanPingPong(
    __gm__ float* local_dst, __gm__ float* remote_src, uint64_t elem_count) {
    if (elem_count == 0) {
        return;
    }

    int pending_slot = -1;
    uint64_t pending_off = 0;
    uint64_t pending_chunk = 0;
    int next_slot = 0;

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    for (uint64_t off = 0; off < elem_count; off += kStageChunk) {
        uint64_t chunk = elem_count - off;
        if (chunk > kStageChunk) {
            chunk = kStageChunk;
        }

        wait_flag(PIPE_MTE3, PIPE_MTE2, DummyEvent(next_slot));
        DummyLoad(remote_src, off, chunk, next_slot);

        if (pending_slot >= 0) {
            DummyStore(local_dst, pending_off, pending_chunk, pending_slot);
        }

        pending_slot = next_slot;
        pending_off = off;
        pending_chunk = chunk;
        next_slot ^= 1;
    }

    if (pending_slot >= 0) {
        DummyStore(local_dst, pending_off, pending_chunk, pending_slot);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    if (get_block_idx() != 0) {
        return;
    }

    __gm__ Tensor* dst_t = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* src_t = reinterpret_cast<__gm__ Tensor*>(args[1]);
    (void)reinterpret_cast<__gm__ Tensor*>(args[2]);
    (void)reinterpret_cast<__gm__ Tensor*>(args[3]);
    __gm__ CommDeviceContext* comm_ctx = reinterpret_cast<__gm__ CommDeviceContext*>(args[4]);
    int n_ranks = static_cast<int>(args[5]);
    (void)args[6];
    uint64_t dummy_comm_bytes = static_cast<uint64_t>(args[7]);

    if (comm_ctx == nullptr || n_ranks <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    if (dummy_comm_bytes == 0 || dummy_comm_bytes > kMaxTotalBytes) {
        dummy_comm_bytes = kDefaultTotalBytes;
    }

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_t->buffer.addr) + dst_t->start_offset;
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_t->buffer.addr) + src_t->start_offset;
    uint64_t src_count = src_t->buffer.size / sizeof(float);
    uint64_t dst_count = dst_t->buffer.size / sizeof(float);
    if (src_count == 0 || dst_count == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    int actual_ranks = n_ranks;
    if (actual_ranks > static_cast<int>(COMM_MAX_RANK_NUM)) {
        actual_ranks = static_cast<int>(COMM_MAX_RANK_NUM);
    }

    uint64_t total_count = (dummy_comm_bytes + sizeof(float) - 1) / sizeof(float);
    uint64_t region_count = kPingPongElems;
    uint64_t half_dst_count = dst_count / 2;
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
    uint64_t remaining = total_count;
    uint64_t rank_cursor = 0;
    uint64_t region_cursor = 0;

    while (remaining > 0) {
        uint64_t region_base = (has_two_regions && ((region_cursor & 1U) != 0)) ? region_count : 0;
        uint64_t region_target = remaining;
        if (region_target > region_count) {
            region_target = region_count;
        }

        uint64_t region_written = 0;
        while (region_written < region_target) {
            int rank = static_cast<int>(rank_cursor % static_cast<uint64_t>(actual_ranks));
            __gm__ float* remote_src = CommRemotePtr(comm_ctx, src, rank);
            uint64_t copy_count = region_target - region_written;
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

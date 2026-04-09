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
 *   args[6] = dummy_comm_scale (scalar, repeat factor; backward-compatible)
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

// Use 64KB chunks so two UB staging tiles fit comfortably for ping-pong MTE copy.
static constexpr size_t DUMMY_CHUNK = 128 * 128;
static constexpr int DUMMY_REPEAT_DEFAULT = 4;
static constexpr int DUMMY_REPEAT_MAX = 4096;
static constexpr uint64_t DUMMY_PING_TILE_ADDR = 0x0;
static constexpr uint64_t DUMMY_PONG_TILE_ADDR = DUMMY_CHUNK * sizeof(float);

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
using TileData = pto::Tile<pto::TileType::Vec, float, 1, DUMMY_CHUNK, pto::BLayout::RowMajor, -1, -1>;

__aicore__ __attribute__((always_inline)) inline event_t GetDummyEvent(int slot)
{
    return (slot == 0) ? EVENT_ID0 : EVENT_ID1;
}

__aicore__ __attribute__((always_inline)) inline uint64_t GetDummyTileAddr(int slot)
{
    return (slot == 0) ? DUMMY_PING_TILE_ADDR : DUMMY_PONG_TILE_ADDR;
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

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* dst_td = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* src_td = reinterpret_cast<__gm__ TensorData*>(args[1]);
    (void)args[2];
    __gm__ HcclDeviceContext* hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext*>(args[3]);
    int nranks = static_cast<int>(args[4]);
    int root = static_cast<int>(args[5]);
    uint64_t scale_raw = static_cast<uint64_t>(args[6]);
    (void)args[7];

    int dummy_repeat = static_cast<int>(scale_raw);
    // Backward compatibility:
    // - old callers pass sdma_workspace_ptr here (large address), treat as default.
    // - invalid/zero values also fall back to default.
    if (dummy_repeat <= 0 || dummy_repeat > DUMMY_REPEAT_MAX) {
        dummy_repeat = DUMMY_REPEAT_DEFAULT;
    }

    int my_rank = static_cast<int>(hcclCtx->rankId);
    if (my_rank != root) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_td->buffer.addr);
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_td->buffer.addr);

    size_t gather_count = src_td->buffer.size / sizeof(float);
    int actual_nranks = (nranks > 16) ? 16 : nranks;

    for (int rep = 0; rep < dummy_repeat; ++rep) {
        for (int r = 0; r < actual_nranks; ++r) {
            __gm__ float* remote_src = HcclRemotePtr(hcclCtx, src, r);
            __gm__ float* local_dst = dst + static_cast<ptrdiff_t>(r) * gather_count;
            int pending_slot = -1;
            size_t pending_off = 0;
            size_t pending_chunk = 0;
            int next_slot = 0;

            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

            for (size_t off = 0; off < gather_count; off += DUMMY_CHUNK) {
                size_t chunk = gather_count - off;
                if (chunk > DUMMY_CHUNK) chunk = DUMMY_CHUNK;

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
    }

    pipe_barrier(PIPE_ALL);
}

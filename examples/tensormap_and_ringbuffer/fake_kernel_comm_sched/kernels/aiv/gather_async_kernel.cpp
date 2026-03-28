/**
 * TGET_ASYNC gather kernel - root gathers from all ranks via SDMA async DMA.
 * Requires pto-comm-isa with async SDMA support.
 *
 * Args (8):
 *   args[0] = dst (TensorData*)
 *   args[1] = src (TensorData*)
 *   args[2] = sync_done (TensorData*, dependency only - ignored)
 *   args[3] = device_ctx_ptr (scalar)
 *   args[4] = nranks (scalar)
 *   args[5] = root (scalar)
 *   args[6] = sdma_workspace_ptr (scalar)
 *   args[7] = debug_poll_counts (TensorData*, shape={nranks}, dtype=INT32)
 *             Each element receives the SdmaWaitEvent poll iteration count
 *             for the corresponding rank's TGET_ASYNC operation.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include "pto/comm/pto_comm_inst.hpp"
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

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* dst_td = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* src_td = reinterpret_cast<__gm__ TensorData*>(args[1]);
    (void)args[2];  // sync_done dependency
    __gm__ HcclDeviceContext* hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext*>(args[3]);
    int nranks = static_cast<int>(args[4]);
    int root = static_cast<int>(args[5]);
    __gm__ uint8_t* sdmaWorkspace = reinterpret_cast<__gm__ uint8_t*>(args[6]);
    __gm__ TensorData* debug_td = reinterpret_cast<__gm__ TensorData*>(args[7]);

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_td->buffer.addr);
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_td->buffer.addr);
    __gm__ int32_t* debug_poll = reinterpret_cast<__gm__ int32_t*>(debug_td->buffer.addr);

    // Derive actual gather_count from dst tensor size at runtime
    size_t gather_count = dst_td->buffer.size / (static_cast<size_t>(nranks) * sizeof(float));

    int my_rank = static_cast<int>(hcclCtx->rankId);

    if (my_rank != root) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, gather_count);
    StrideDyn stride(gather_count, gather_count, gather_count, gather_count, 1);

    // Build async session
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    // Poll count tracking per rank
    uint32_t poll_counts[16] = {};

    // Async gather: TGET_ASYNC from each remote rank into local dst
    constexpr int kEventSlots = pto::comm::sdma::SDMA_EVENT_SLOT_COUNT;
    pto::comm::AsyncEvent events[kEventSlots];
    int issued = 0;

    for (int target_rank = 0; target_rank < nranks; ++target_rank) {
        if (target_rank == root) {
            // Local copy: root's own data goes to dst[root * gather_count]
            __gm__ float* localDst = dst + root * gather_count;
            Global localDstG(localDst, shape, stride);
            Global localSrcG(src, shape, stride);
            if (issued >= kEventSlots) {
                int recycled = issued - kEventSlots;
                poll_counts[recycled] = events[issued % kEventSlots].WaitCounted(session);
            }
            events[issued % kEventSlots] = pto::comm::TGET_ASYNC(localDstG, localSrcG, session);
            issued++;
            continue;
        }

        __gm__ float* remoteSrc = HcclRemotePtr(hcclCtx, src, target_rank);
        __gm__ float* localDst = dst + target_rank * gather_count;
        Global remoteSrcG(remoteSrc, shape, stride);
        Global localDstG(localDst, shape, stride);

        if (issued >= kEventSlots) {
            int recycled = issued - kEventSlots;
            poll_counts[recycled] = events[issued % kEventSlots].WaitCounted(session);
        }
        events[issued % kEventSlots] = pto::comm::TGET_ASYNC(localDstG, remoteSrcG, session);
        issued++;
    }

    // Wait for all pending events
    const int pending = (issued < kEventSlots) ? issued : kEventSlots;
    const int first_unwaited = issued - pending;
    for (int i = 0; i < pending; ++i) {
        poll_counts[first_unwaited + i] = events[i].WaitCounted(session);
    }

    // Write poll counts to debug tensor via GM DMA
    auto &tmpBuf = session.sdmaSession.eventCtx.tmpBuf;
    uint32_t syncId = session.sdmaSession.eventCtx.syncId;
    for (int i = 0; i < nranks && i < 16; ++i) {
        pto::comm::sdma::detail::SetValue<int32_t>(
            reinterpret_cast<__gm__ uint8_t *>(debug_poll + i),
            tmpBuf, syncId, static_cast<int32_t>(poll_counts[i]));
    }

    pipe_barrier(PIPE_ALL);
}

/**
 * TGET_ASYNC gather kernel - root gathers from all ranks via SDMA async DMA.
 * Requires pto-comm-isa with async SDMA support.
 *
 * Args (7):
 *   args[0] = dst (TensorData*)
 *   args[1] = src (TensorData*)
 *   args[2] = sync_done (TensorData*, dependency only - ignored)
 *   args[3] = device_ctx_ptr (scalar)
 *   args[4] = nranks (scalar)
 *   args[5] = root (scalar)
 *   args[6] = sdma_workspace_ptr (scalar)
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

static constexpr size_t GATHER_COUNT = 256;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* dst_td = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* src_td = reinterpret_cast<__gm__ TensorData*>(args[1]);
    (void)args[2];  // sync_done dependency
    __gm__ HcclDeviceContext* hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext*>(args[3]);
    int nranks = static_cast<int>(args[4]);
    int root = static_cast<int>(args[5]);
    __gm__ uint8_t* sdmaWorkspace = reinterpret_cast<__gm__ uint8_t*>(args[6]);

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_td->buffer.addr);
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_td->buffer.addr);

    int my_rank = static_cast<int>(hcclCtx->rankId);

    if (my_rank != root) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, GATHER_COUNT);
    StrideDyn stride(GATHER_COUNT, GATHER_COUNT, GATHER_COUNT, GATHER_COUNT, 1);

    // Build async session
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    // Async gather: TGET_ASYNC from each remote rank into local dst
    constexpr int kEventSlots = pto::comm::sdma::SDMA_EVENT_SLOT_COUNT;
    pto::comm::AsyncEvent events[kEventSlots];
    int issued = 0;

    for (int target_rank = 0; target_rank < nranks; ++target_rank) {
        if (target_rank == root) {
            // Local copy: root's own data goes to dst[root * GATHER_COUNT]
            __gm__ float* localDst = dst + root * GATHER_COUNT;
            Global localDstG(localDst, shape, stride);
            Global localSrcG(src, shape, stride);
            if (issued >= kEventSlots) {
                (void)events[issued % kEventSlots].Wait(session);
            }
            events[issued % kEventSlots] = pto::comm::TGET_ASYNC(localDstG, localSrcG, session);
            issued++;
            continue;
        }

        __gm__ float* remoteSrc = HcclRemotePtr(hcclCtx, src, target_rank);
        __gm__ float* localDst = dst + target_rank * GATHER_COUNT;
        Global remoteSrcG(remoteSrc, shape, stride);
        Global localDstG(localDst, shape, stride);

        if (issued >= kEventSlots) {
            (void)events[issued % kEventSlots].Wait(session);
        }
        events[issued % kEventSlots] = pto::comm::TGET_ASYNC(localDstG, remoteSrcG, session);
        issued++;
    }

    // Wait for all pending events
    const int pending = (issued < kEventSlots) ? issued : kEventSlots;
    for (int i = 0; i < pending; ++i) {
        (void)events[i].Wait(session);
    }

    pipe_barrier(PIPE_ALL);
}

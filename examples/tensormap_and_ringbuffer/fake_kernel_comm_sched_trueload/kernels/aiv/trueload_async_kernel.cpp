/**
 * TrueLoadAsync: explicit SDMA remote-copy kernel.
 *
 * Root fetches each rank's source buffer into its local destination buffer via
 * TGET_ASYNC. No TGATHER collective is involved.
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
 */

#include <cstddef>
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
    (void)args[2];
    __gm__ HcclDeviceContext* hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext*>(args[3]);
    int nranks = static_cast<int>(args[4]);
    int root = static_cast<int>(args[5]);
    __gm__ uint8_t* sdma_workspace = reinterpret_cast<__gm__ uint8_t*>(args[6]);
    __gm__ TensorData* debug_td = reinterpret_cast<__gm__ TensorData*>(args[7]);

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_td->buffer.addr);
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_td->buffer.addr);
    __gm__ int32_t* debug_poll = reinterpret_cast<__gm__ int32_t*>(debug_td->buffer.addr);

    size_t segment_count = src_td->buffer.size / sizeof(float);
    int my_rank = static_cast<int>(hcclCtx->rankId);

    if (my_rank != root || segment_count == 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, segment_count);
    StrideDyn stride(segment_count, segment_count, segment_count, segment_count, 1);

    ScratchTile scratch_tile;
    TASSIGN(scratch_tile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratch_tile, sdma_workspace, session, 0)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    uint32_t wait_status[16] = {};

    constexpr int kEventSlots = pto::comm::sdma::SDMA_EVENT_SLOT_COUNT;
    pto::comm::AsyncEvent events[kEventSlots];
    int issued = 0;

    for (int target_rank = 0; target_rank < nranks; ++target_rank) {
        __gm__ float* local_dst = dst + static_cast<ptrdiff_t>(target_rank) * segment_count;
        __gm__ float* remote_src = (target_rank == root) ? src : HcclRemotePtr(hcclCtx, src, target_rank);
        Global remote_src_g(remote_src, shape, stride);
        Global local_dst_g(local_dst, shape, stride);

        if (issued >= kEventSlots) {
            int recycled = issued - kEventSlots;
            wait_status[recycled] = events[issued % kEventSlots].Wait(session) ? 1U : 0U;
        }
        events[issued % kEventSlots] = pto::comm::TGET_ASYNC(local_dst_g, remote_src_g, session);
        issued++;
    }

    const int pending = (issued < kEventSlots) ? issued : kEventSlots;
    const int first_unwaited = issued - pending;
    for (int i = 0; i < pending; ++i) {
        wait_status[first_unwaited + i] = events[i].Wait(session) ? 1U : 0U;
    }

    auto& tmp_buf = session.sdmaSession.eventCtx.tmpBuf;
    uint32_t sync_id = session.sdmaSession.eventCtx.syncId;
    for (int i = 0; i < nranks && i < 16; ++i) {
        pto::comm::sdma::detail::SetValue<int32_t>(
            reinterpret_cast<__gm__ uint8_t*>(debug_poll + i),
            tmp_buf,
            sync_id,
            static_cast<int32_t>(wait_status[i]));
    }

    pipe_barrier(PIPE_ALL);
}

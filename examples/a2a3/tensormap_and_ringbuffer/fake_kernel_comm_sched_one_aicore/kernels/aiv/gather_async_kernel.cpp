#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/npu/comm/async/sdma/sdma_types.hpp>

#include "comm_utils.h"
#include "tensor.h"

using namespace pto;

using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
using GlobalData = GlobalTensor<float, ShapeDyn, StrideDyn, Layout::ND>;
using ScratchTile = Tile<TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

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
    __gm__ uint8_t* sdma_workspace = reinterpret_cast<__gm__ uint8_t*>(args[7]);

    __gm__ int32_t* debug =
        reinterpret_cast<__gm__ int32_t*>(debug_t->buffer.addr) + debug_t->start_offset;
    if (comm_ctx == nullptr || n_ranks <= 0 || sdma_workspace == nullptr) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_t->buffer.addr) + dst_t->start_offset;
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_t->buffer.addr) + src_t->start_offset;
    uint64_t gather_count = src_t->buffer.size / sizeof(float);

    int actual_ranks = n_ranks;
    if (actual_ranks > static_cast<int>(COMM_MAX_RANK_NUM)) {
        actual_ranks = static_cast<int>(COMM_MAX_RANK_NUM);
    }

    ShapeDyn shape(1, 1, 1, 1, gather_count);
    StrideDyn stride(gather_count, gather_count, gather_count, gather_count, 1);

    ScratchTile scratch_tile;
    TASSIGN(scratch_tile, 0x20000);

    pto::comm::AsyncSession session;
    bool ok = pto::comm::BuildAsyncSession<pto::comm::DmaEngine::SDMA>(
        scratch_tile, sdma_workspace, session, 0);
    if (!ok || !session.valid) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    constexpr int kEventSlots = pto::comm::sdma::SDMA_EVENT_SLOT_COUNT;
    pto::comm::AsyncEvent events[kEventSlots];
    uint32_t wait_status[COMM_MAX_RANK_NUM] = {};
    int issued = 0;

    for (int rank = 0; rank < actual_ranks; ++rank) {
        if (issued >= kEventSlots) {
            int recycled = issued - kEventSlots;
            wait_status[recycled] = events[issued % kEventSlots].Wait(session) ? 1U : 0U;
        }

        __gm__ float* remote_src = CommRemotePtr(comm_ctx, src, rank);
        __gm__ float* local_dst = dst + static_cast<uint64_t>(rank) * gather_count;
        GlobalData src_g(remote_src, shape, stride);
        GlobalData dst_g(local_dst, shape, stride);
        events[issued % kEventSlots] =
            pto::comm::TGET_ASYNC<pto::comm::DmaEngine::SDMA>(dst_g, src_g, session);
        ++issued;
    }

    int pending = issued < kEventSlots ? issued : kEventSlots;
    int first_unwaited = issued - pending;
    for (int i = 0; i < pending; ++i) {
        int rank_index = first_unwaited + i;
        wait_status[rank_index] = events[rank_index % kEventSlots].Wait(session) ? 1U : 0U;
    }

    for (int i = 0; i < actual_ranks; ++i) {
        debug[i] = static_cast<int32_t>(wait_status[i]);
    }

    pipe_barrier(PIPE_ALL);
}

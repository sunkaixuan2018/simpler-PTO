/**
 * All-to-all barrier: every rank waits for every other rank.
 *
 * Used by AllGather where every rank reads from all ranks' windows.
 * ALL ranks do TWAIT here.
 *
 * Tensormap_and_ringbuffer: args[0..4] as below, args[5] = sync_done (TensorData* output)
 *   args[0] = barrier_base (TensorData*)
 *   args[1] = device_ctx_ptr (scalar)
 *   args[2] = n_ranks (scalar)
 *   args[3] = root (scalar)
 *   args[4] = dependency (TensorData*, ignored)
 *   args[5] = sync_done (TensorData* output) - write 1 after barrier for task ordering
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/comm/pto_comm_inst.hpp>
#include "hccl_context.h"
#include "hccl_helpers.h"

#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* barrier_td = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ int32_t* local_barrier = reinterpret_cast<__gm__ int32_t*>(barrier_td->buffer.addr);
    __gm__ HcclDeviceContext* ctx = reinterpret_cast<__gm__ HcclDeviceContext*>(args[1]);
    int n_ranks = static_cast<int>(args[2]);
    int root = static_cast<int>(args[3]);
    int my_rank = static_cast<int>(ctx->rankId);

    (void)args[4];

    __gm__ int32_t* remote_slot = HcclRemotePtr(ctx, local_barrier, root) + my_rank;
    pto::comm::Signal sig(remote_slot);
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::Set);

    __gm__ int32_t* root_barrier = HcclRemotePtr(ctx, local_barrier, root);
    for (int i = 0; i < n_ranks; ++i) {
        pto::comm::Signal slot(root_barrier + i);
        pto::comm::TWAIT(slot, 1, pto::comm::WaitCmp::GE);
    }

    __gm__ TensorData* sync_td = reinterpret_cast<__gm__ TensorData*>(args[5]);
    __gm__ int32_t* sync_done = reinterpret_cast<__gm__ int32_t*>(sync_td->buffer.addr);
    sync_done[0] = 1;

    pipe_barrier(PIPE_ALL);
}

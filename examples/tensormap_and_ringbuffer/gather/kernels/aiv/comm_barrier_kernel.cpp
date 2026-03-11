/**
 * Device-side cross-rank barrier using TNOTIFY/TWAIT from pto-comm-isa.
 *
 * Tensormap_and_ringbuffer: args are TensorData* for barrier, scalars for ctx/n_ranks/root,
 * and optional TensorData* for dependency (win_src - ignored).
 *   args[0] = barrier_base (TensorData*)
 *   args[1] = device_ctx_ptr (scalar)
 *   args[2] = n_ranks (scalar)
 *   args[3] = root (scalar)
 *   args[4] = win_src (TensorData*, dependency only - ignored)
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

    (void)args[4];  // win_src dependency - ignored

    // Each rank writes flag=1 to root's barrier slot[my_rank] via RDMA.
    __gm__ int32_t* remote_slot = HcclRemotePtr(ctx, local_barrier, root) + my_rank;
    pto::comm::Signal sig(remote_slot);
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::Set);

    // Root waits until every rank's flag is >= 1.
    if (my_rank == root) {
        for (int i = 0; i < n_ranks; i++) {
            pto::comm::Signal slot(local_barrier + i);
            pto::comm::TWAIT(slot, 1, pto::comm::WaitCmp::GE);
        }
    }

    pipe_barrier(PIPE_ALL);
}

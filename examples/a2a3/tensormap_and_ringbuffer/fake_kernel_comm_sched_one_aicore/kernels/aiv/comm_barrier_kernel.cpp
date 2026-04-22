#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include <pto/comm/pto_comm_inst.hpp>

#include "comm_utils.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    if (get_block_idx() != 0) {
        return;
    }

    __gm__ Tensor* barrier_t = reinterpret_cast<__gm__ Tensor*>(args[0]);
    (void)reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ Tensor* sync_t = reinterpret_cast<__gm__ Tensor*>(args[2]);
    __gm__ CommDeviceContext* ctx = reinterpret_cast<__gm__ CommDeviceContext*>(args[3]);
    int n_ranks = static_cast<int>(args[4]);
    int root = static_cast<int>(args[5]);

    __gm__ int32_t* local_barrier =
        reinterpret_cast<__gm__ int32_t*>(barrier_t->buffer.addr) + barrier_t->start_offset;
    int my_rank = static_cast<int>(ctx->rankId);

    __gm__ int32_t* remote_slot = CommRemotePtr(ctx, local_barrier, root) + my_rank;
    pto::comm::Signal sig(remote_slot);
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::Set);

    __gm__ int32_t* root_barrier = CommRemotePtr(ctx, local_barrier, root);
    for (int i = 0; i < n_ranks; ++i) {
        pto::comm::Signal slot(root_barrier + i);
        pto::comm::TWAIT(slot, 1, pto::comm::WaitCmp::GE);
    }

    __gm__ int32_t* sync_done =
        reinterpret_cast<__gm__ int32_t*>(sync_t->buffer.addr) + sync_t->start_offset;
    sync_done[0] = 1;
    pipe_barrier(PIPE_ALL);
}

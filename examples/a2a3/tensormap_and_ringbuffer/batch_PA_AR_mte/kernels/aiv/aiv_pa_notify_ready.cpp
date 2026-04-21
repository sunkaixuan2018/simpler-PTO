/**
 * Send a ready notification to all peer ranks after local PA output is ready.
 *
 * args[0] = &Tensor(local_out)      -- dependency only
 * args[1] = &Tensor(notify_done)    -- output dependency token
 * args[2] = local notify counter    -- window address
 * args[3] = CommDeviceContext*
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>

#include "common/comm_context.h"
#include "tensor.h"

using namespace pto;

#include "pto_notify_kernel_api.h"

template <typename T>
AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* local_ptr,
                                      int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)local_ptr - local_base;
    return (__gm__ T*)(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    (void)reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* done_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ int32_t* local_counter = reinterpret_cast<__gm__ int32_t*>(args[2]);
    __gm__ CommDeviceContext* comm_ctx = reinterpret_cast<__gm__ CommDeviceContext*>(args[3]);

    if (comm_ctx == nullptr || comm_ctx->rankNum <= 1) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    int my_rank = static_cast<int>(comm_ctx->rankId);
    int nranks = static_cast<int>(comm_ctx->rankNum);
    for (int peer_rank = 0; peer_rank < nranks; ++peer_rank) {
        if (peer_rank == my_rank) continue;
        __gm__ int32_t* remote_counter = CommRemotePtr(comm_ctx, local_counter, peer_rank);
        pto2_send_notification(remote_counter, 1, PTO2NotifyOp::AtomicAdd);
    }

    __gm__ int32_t* done =
        reinterpret_cast<__gm__ int32_t*>(done_tensor->buffer.addr) + done_tensor->start_offset;
    done[0] = 1;
    pipe_barrier(PIPE_ALL);
}

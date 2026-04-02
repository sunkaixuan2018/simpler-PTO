/**
 * All-to-all barrier: every rank waits for every other rank.
 *
 * Args:
 *   args[0] = barrier_base (Tensor* in window memory, int32[n_ranks])
 *   args[1] = sync_done (Tensor* output) - write 1 after barrier for task ordering
 *   args[2] = device_ctx_ptr (CommDeviceContext*)
 *   args[3] = n_ranks (scalar)
 *   args[4] = root (scalar)
 */

 #include <cstdint>

 #include <pto/pto-inst.hpp>
 #include <pto/comm/pto_comm_inst.hpp>
 
 #include "common/comm_context.h"
 #include "tensor.h"
 
 #ifndef __gm__
 #define __gm__
 #endif
 
 #ifndef __aicore__
 #define __aicore__ [aicore]
 #endif
 
 template <typename T>
 AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* local_ptr,
                                       int peer_rank) {
     uint64_t local_base = ctx->windowsIn[ctx->rankId];
     uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
     return reinterpret_cast<__gm__ T*>(ctx->windowsIn[peer_rank] + offset);
 }
 
 extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
     __gm__ Tensor* barrier_t = reinterpret_cast<__gm__ Tensor*>(args[0]);
     __gm__ int32_t* local_barrier = reinterpret_cast<__gm__ int32_t*>(barrier_t->buffer.addr);
     __gm__ Tensor* sync_t = reinterpret_cast<__gm__ Tensor*>(args[1]);
     __gm__ CommDeviceContext* ctx = reinterpret_cast<__gm__ CommDeviceContext*>(args[2]);
     int n_ranks = static_cast<int>(args[3]);
     int root = static_cast<int>(args[4]);
     int my_rank = static_cast<int>(ctx->rankId);
 
     __gm__ int32_t* remote_slot = CommRemotePtr(ctx, local_barrier, root) + my_rank;
     pto::comm::Signal sig(remote_slot);
     pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::Set);
 
     __gm__ int32_t* root_barrier = CommRemotePtr(ctx, local_barrier, root);
     for (int i = 0; i < n_ranks; ++i) {
         pto::comm::Signal slot(root_barrier + i);
         pto::comm::TWAIT(slot, 1, pto::comm::WaitCmp::GE);
     }
 
     __gm__ int32_t* sync_done = reinterpret_cast<__gm__ int32_t*>(sync_t->buffer.addr);
     sync_done[0] = 1;
 
     pipe_barrier(PIPE_ALL);
 }
 
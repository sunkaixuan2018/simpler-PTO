/**
 * WindowMemCopyIn: Copy device buffer to HCCL window.
 * Used before TGATHER so remote ranks can read.
 *
 * Tensormap_and_ringbuffer: args are TensorData* for buffers, scalar for count.
 *   args[0] = win_dst (TensorData*)
 *   args[1] = dev_src (TensorData*)
 *   args[2] = count (scalar)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* win_dst_td = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* dev_src_td = reinterpret_cast<__gm__ TensorData*>(args[1]);
    int count = static_cast<int>(args[2]);

    __gm__ float* win_dst = reinterpret_cast<__gm__ float*>(win_dst_td->buffer.addr);
    __gm__ float* dev_src = reinterpret_cast<__gm__ float*>(dev_src_td->buffer.addr);

    for (int i = 0; i < count; ++i) {
        win_dst[i] = dev_src[i];
    }
    pipe_barrier(PIPE_ALL);
}

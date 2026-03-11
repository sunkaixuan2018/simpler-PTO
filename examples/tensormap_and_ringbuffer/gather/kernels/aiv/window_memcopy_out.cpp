/**
 * WindowMemCopyOut: Copy HCCL window to device buffer.
 * Root only - after TGATHER, copy gathered result to device.
 *
 * Tensormap_and_ringbuffer: args are TensorData* for buffers, scalar for count.
 *   args[0] = dev_dst (TensorData*)
 *   args[1] = win_src (TensorData*)
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
    __gm__ TensorData* dev_dst_td = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* win_src_td = reinterpret_cast<__gm__ TensorData*>(args[1]);
    int count = static_cast<int>(args[2]);

    __gm__ float* dev_dst = reinterpret_cast<__gm__ float*>(dev_dst_td->buffer.addr);
    __gm__ float* win_src = reinterpret_cast<__gm__ float*>(win_src_td->buffer.addr);

    for (int i = 0; i < count; ++i) {
        dev_dst[i] = win_src[i];
    }
    pipe_barrier(PIPE_ALL);
}

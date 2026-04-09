/**
 * DummyWindowFill: initialize a fixed-size dummy source window independent of
 * the foreground gather size.
 *
 * Args (4):
 *   args[0] = win_dst (TensorData*)
 *   args[1] = dev_src (TensorData*)
 *   args[2] = count (scalar, float elements to fill)
 *   args[3] = sync_done (TensorData*, dependency only - ignored)
 */

#include <cstdint>

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
    (void)args[3];

    __gm__ float* win_dst = reinterpret_cast<__gm__ float*>(win_dst_td->buffer.addr);
    __gm__ float* dev_src = reinterpret_cast<__gm__ float*>(dev_src_td->buffer.addr);

    int src_count = static_cast<int>(dev_src_td->buffer.size / sizeof(float));
    if (count <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    if (src_count <= 0) {
        for (int i = 0; i < count; ++i) {
            win_dst[i] = 0.0f;
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    for (int i = 0; i < count; ++i) {
        win_dst[i] = dev_src[i % src_count];
    }
    pipe_barrier(PIPE_ALL);
}

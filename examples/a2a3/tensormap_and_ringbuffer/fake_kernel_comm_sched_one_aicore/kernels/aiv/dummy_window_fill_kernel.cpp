#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>

#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    if (get_block_idx() != 0) {
        return;
    }

    __gm__ Tensor* win_dst_t = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* dev_src_t = reinterpret_cast<__gm__ Tensor*>(args[1]);
    (void)reinterpret_cast<__gm__ Tensor*>(args[2]);
    uint64_t count = static_cast<uint64_t>(args[3]);

    __gm__ float* win_dst =
        reinterpret_cast<__gm__ float*>(win_dst_t->buffer.addr) + win_dst_t->start_offset;
    __gm__ float* dev_src =
        reinterpret_cast<__gm__ float*>(dev_src_t->buffer.addr) + dev_src_t->start_offset;
    uint64_t src_count = dev_src_t->buffer.size / sizeof(float);

    if (src_count == 0) {
        for (uint64_t i = 0; i < count; ++i) {
            win_dst[i] = 0.0f;
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    for (uint64_t i = 0; i < count; ++i) {
        win_dst[i] = dev_src[i % src_count];
    }
    pipe_barrier(PIPE_ALL);
}

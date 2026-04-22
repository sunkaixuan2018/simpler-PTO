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

    __gm__ Tensor* dev_dst_t = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* win_src_t = reinterpret_cast<__gm__ Tensor*>(args[1]);
    uint64_t count = static_cast<uint64_t>(args[2]);

    __gm__ float* dev_dst =
        reinterpret_cast<__gm__ float*>(dev_dst_t->buffer.addr) + dev_dst_t->start_offset;
    __gm__ float* win_src =
        reinterpret_cast<__gm__ float*>(win_src_t->buffer.addr) + win_src_t->start_offset;

    for (uint64_t i = 0; i < count; ++i) {
        dev_dst[i] = win_src[i];
    }
    pipe_barrier(PIPE_ALL);
}

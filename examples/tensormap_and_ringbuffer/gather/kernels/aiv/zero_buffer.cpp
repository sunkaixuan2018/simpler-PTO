/**
 * ZeroBuffer: Zero a buffer. Used for non-root ranks to initialize output.
 *
 * Args:
 *   args[0] = dst (TensorData*)
 *   args[1] = count (scalar, in elements)
 *   args[2] = dependency (TensorData*, ignored - for task ordering)
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
    __gm__ TensorData* dst_td = reinterpret_cast<__gm__ TensorData*>(args[0]);
    int count = static_cast<int>(args[1]);

    (void)args[2];  // dependency - ignored

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_td->buffer.addr);

    for (int i = 0; i < count; ++i) {
        dst[i] = 0.0f;
    }
    pipe_barrier(PIPE_ALL);
}

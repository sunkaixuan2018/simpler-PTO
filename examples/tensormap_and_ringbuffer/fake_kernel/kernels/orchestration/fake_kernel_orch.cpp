/**
 * fake_kernel orchestration: dynamic kernel variant selection
 *
 * Computes out = a + b using one of two interchangeable kernels (AddV1 or AddV2).
 * Selection logic:
 *   - variant_id == -1 (default): pseudo-random selection
 *   - variant_id == 0: always use AddV1 (func_id=0)
 *   - variant_id == 1: always use AddV2 (func_id=1)
 *
 * Args layout: [a, b, out, size_a, size_b, size_out, SIZE, variant_id]
 *              + [gm_heap, heap_size] appended by runtime
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_ID_ADD_V1 0
#define FUNC_ID_ADD_V2 1

#define ARG_PTR_A   0
#define ARG_PTR_B   1
#define ARG_PTR_OUT 2
#define ARG_SIZE_A  3
#define ARG_SIZE_B  4
#define ARG_SIZE_OUT 5
#define ARG_SIZE    6
#define ARG_VARIANT 7

// Simple pseudo-random: XOR-shift based on a static counter
static uint32_t s_rand_state = 0x12345678;
static int pseudo_random_bit() {
    s_rand_state ^= s_rand_state << 13;
    s_rand_state ^= s_rand_state >> 17;
    s_rand_state ^= s_rand_state << 5;
    return s_rand_state & 1;
}

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 8,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;
    pto2_rt_init_tensor_pool(rt);

    void* a_ptr   = (void*)(uintptr_t)args[ARG_PTR_A];
    void* b_ptr   = (void*)(uintptr_t)args[ARG_PTR_B];
    void* out_ptr = (void*)(uintptr_t)args[ARG_PTR_OUT];
    int SIZE = (int)(args[ARG_SIZE] & 0x7FFFFFFF);
    int64_t variant_id = (int64_t)args[ARG_VARIANT];

    // Select kernel variant
    int kernel_id;
    if (variant_id == 0) {
        kernel_id = FUNC_ID_ADD_V1;
    } else if (variant_id == 1) {
        kernel_id = FUNC_ID_ADD_V2;
    } else {
        // Default: pseudo-random selection
        kernel_id = pseudo_random_bit() ? FUNC_ID_ADD_V2 : FUNC_ID_ADD_V1;
    }

    LOG_INFO(rt, "fake_kernel: variant_id=%lld, selected func_id=%d (AddV%d)",
             (long long)variant_id, kernel_id, kernel_id + 1);

    uint64_t shapes[1] = {(uint64_t)SIZE};
    Tensor ext_a   = make_tensor_external(a_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_b   = make_tensor_external(b_ptr, shapes, 1, DataType::FLOAT32);
    Tensor ext_out = make_tensor_external(out_ptr, shapes, 1, DataType::FLOAT32);

    PTOParam params[] = {
        make_input_param(ext_a),
        make_input_param(ext_b),
        make_output_param(ext_out),
    };
    pto2_rt_submit_task(rt, kernel_id, PTO2_WORKER_VECTOR, params, 3);
}

}  // extern "C"

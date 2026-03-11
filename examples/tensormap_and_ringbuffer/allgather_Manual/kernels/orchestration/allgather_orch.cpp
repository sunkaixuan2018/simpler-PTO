/**
 * AllGather (Manual) orchestration for tensormap_and_ringbuffer runtime.
 *
 * Flow: WindowMemCopyIn -> CommBarrier(pre) -> AllGatherManual -> WindowMemCopyOut -> CommBarrier(post)
 * All ranks run in parallel. Every rank gets full concatenation.
 *
 * Args (10): [0] dev_src, [1] dev_out, [2] size_src, [3] size_out,
 *   [4] device_ctx_ptr, [5] win_in_base, [6] win_out_base,
 *   [7] n_ranks, [8] root (unused), [9] rank_id
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

constexpr int GATHER_COUNT = 64;
constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

#define FUNC_WIN_MEMCOPY_IN  0
#define FUNC_ALLGATHER       1
#define FUNC_WIN_MEMCOPY_OUT 2
#define FUNC_COMM_BARRIER    3

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 10,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;
    pto2_rt_init_tensor_pool(rt);

    void* dev_src = reinterpret_cast<void*>(args[0]);
    void* dev_out = reinterpret_cast<void*>(args[1]);
    uint64_t device_ctx_ptr = args[4];
    uint64_t win_in_base = args[5];
    (void)args[6];  // win_out_base unused
    int n_ranks = static_cast<int>(args[7]);
    int rank_id = static_cast<int>(args[9]);

    LOG_INFO(rt, "allgather_Manual: n_ranks=%d rank_id=%d", n_ranks, rank_id);

    size_t barrier_size = static_cast<size_t>(n_ranks) * sizeof(int32_t);
    uint64_t barrier_base_pre = win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t barrier_base_post = barrier_base_pre + barrier_size;
    uint64_t win_src = barrier_base_post + barrier_size;
    uint64_t win_dst = win_src + GATHER_COUNT * sizeof(float);

    uint64_t src_shapes[1] = {GATHER_COUNT};
    uint64_t dst_shapes[1] = {static_cast<uint64_t>(n_ranks) * GATHER_COUNT};
    uint64_t barrier_shapes[1] = {static_cast<uint64_t>(n_ranks)};
    uint64_t sync_shapes[1] = {1};
    Tensor sync_done_t = make_tensor(sync_shapes, 1, DataType::INT32);

    Tensor dev_src_t = make_tensor_external(dev_src, src_shapes, 1, DataType::FLOAT32);
    Tensor dev_out_t = make_tensor_external(dev_out, dst_shapes, 1, DataType::FLOAT32);
    Tensor win_src_t = make_tensor_external(reinterpret_cast<void*>(win_src), src_shapes, 1, DataType::FLOAT32);
    Tensor win_dst_t = make_tensor_external(reinterpret_cast<void*>(win_dst), dst_shapes, 1, DataType::FLOAT32);
    Tensor barrier_pre_t = make_tensor_external(reinterpret_cast<void*>(barrier_base_pre), barrier_shapes, 1, DataType::INT32);
    Tensor barrier_post_t = make_tensor_external(reinterpret_cast<void*>(barrier_base_post), barrier_shapes, 1, DataType::INT32);

    PTO2_SCOPE(rt) {
        PTOParam params_wmin[] = {
            make_output_param(win_src_t),
            make_input_param(dev_src_t),
            make_scalar_param(static_cast<uint64_t>(GATHER_COUNT)),
        };
        pto2_rt_submit_task(rt, FUNC_WIN_MEMCOPY_IN, PTO2_WORKER_VECTOR, params_wmin, 3);

        PTOParam params_barrier_pre[] = {
            make_input_param(barrier_pre_t),
            make_scalar_param(device_ctx_ptr),
            make_scalar_param(static_cast<uint64_t>(n_ranks)),
            make_scalar_param(static_cast<uint64_t>(0)),
            make_input_param(win_src_t),
            make_output_param(sync_done_t),
        };
        pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier_pre, 6);

        PTOParam params_allgather[] = {
            make_output_param(win_dst_t),
            make_input_param(win_src_t),
            make_input_param(sync_done_t),
            make_scalar_param(device_ctx_ptr),
            make_scalar_param(static_cast<uint64_t>(n_ranks)),
            make_scalar_param(static_cast<uint64_t>(rank_id)),
        };
        pto2_rt_submit_task(rt, FUNC_ALLGATHER, PTO2_WORKER_VECTOR, params_allgather, 6);

        PTOParam params_wmout[] = {
            make_output_param(dev_out_t),
            make_input_param(win_dst_t),
            make_scalar_param(static_cast<uint64_t>(n_ranks * GATHER_COUNT)),
        };
        pto2_rt_submit_task(rt, FUNC_WIN_MEMCOPY_OUT, PTO2_WORKER_VECTOR, params_wmout, 3);

        Tensor sync_post_t = make_tensor(sync_shapes, 1, DataType::INT32);
        PTOParam params_barrier_post[] = {
            make_input_param(barrier_post_t),
            make_scalar_param(device_ctx_ptr),
            make_scalar_param(static_cast<uint64_t>(n_ranks)),
            make_scalar_param(static_cast<uint64_t>(0)),
            make_input_param(win_dst_t),
            make_output_param(sync_post_t),
        };
        pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier_post, 6);
    }

    LOG_INFO(rt, "allgather_Manual tasks submitted");
}

}  // extern "C"

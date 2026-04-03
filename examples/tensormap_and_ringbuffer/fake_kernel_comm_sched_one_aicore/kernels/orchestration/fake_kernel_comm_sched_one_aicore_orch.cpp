/**
 * one_aicore orchestration:
 * - built from fake_kernel_comm_sched
 * - keeps per-rank task graph symmetric to reduce rank skew/stalls
 * - constrained by kernel_config.py to block_dim=1 + dual-AIV-on-one-AICore
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);
constexpr int N_ITER = 200;

#define FUNC_WIN_MEMCOPY_IN  0
#define FUNC_GATHER_SYNC     1
#define FUNC_GATHER_ASYNC    2
#define FUNC_WIN_MEMCOPY_OUT 3
#define FUNC_COMM_BARRIER    4

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 13,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    pto2_rt_init_tensor_pool(rt);

    void* dev_src = reinterpret_cast<void*>(args[0]);
    void* dev_out = reinterpret_cast<void*>(args[1]);
    int64_t size_src = static_cast<int64_t>(args[2]);
    uint64_t device_ctx_ptr = args[4];
    uint64_t win_in_base = args[5];
    (void)args[6];
    int n_ranks = static_cast<int>(args[7]);
    int root = static_cast<int>(args[8]);
    int rank_id = static_cast<int>(args[9]);
    uint64_t sdma_workspace_ptr = args[10];
    int strategy = (arg_count > 11) ? static_cast<int>(args[11]) : 0;
    void* dev_debug = (arg_count > 12) ? reinterpret_cast<void*>(args[12]) : nullptr;

    int gather_count = static_cast<int>(size_src / static_cast<int64_t>(sizeof(float)));

    LOG_INFO(rt, "one_aicore: strategy=%d gather_count=%d n_ranks=%d rank=%d",
             strategy, gather_count, n_ranks, rank_id);

    size_t barrier_size = static_cast<size_t>(n_ranks) * sizeof(int32_t);
    uint64_t barrier_base = win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t win_src = barrier_base + barrier_size;
    uint64_t win_dst = win_src + static_cast<uint64_t>(gather_count) * sizeof(float);

    uint64_t src_shapes[1] = {static_cast<uint64_t>(gather_count)};
    uint64_t dst_shapes[1] = {static_cast<uint64_t>(n_ranks) * static_cast<uint64_t>(gather_count)};
    uint64_t barrier_shapes[1] = {static_cast<uint64_t>(n_ranks)};
    uint64_t sync_shapes[1] = {1};
    uint64_t debug_all_shapes[1] = {static_cast<uint64_t>(N_ITER) * static_cast<uint64_t>(n_ranks)};
    uint64_t debug_row_shapes[1] = {static_cast<uint64_t>(n_ranks)};

    Tensor dev_src_t = make_tensor_external(dev_src, src_shapes, 1, DataType::FLOAT32);
    Tensor dev_out_t = make_tensor_external(dev_out, dst_shapes, 1, DataType::FLOAT32);
    Tensor win_src_t = make_tensor_external(reinterpret_cast<void*>(win_src), src_shapes, 1, DataType::FLOAT32);
    Tensor win_dst_t = make_tensor_external(reinterpret_cast<void*>(win_dst), dst_shapes, 1, DataType::FLOAT32);

    Tensor dev_debug_t;
    if (dev_debug != nullptr) {
        dev_debug_t = make_tensor_external(dev_debug, debug_all_shapes, 1, DataType::INT32);
    } else {
        dev_debug_t = make_tensor(debug_all_shapes, 1, DataType::INT32);
    }
    (void)dev_debug_t;

    PTO2_SCOPE(rt) {
        Tensor barrier_t = make_tensor_external(reinterpret_cast<void*>(barrier_base), barrier_shapes, 1, DataType::INT32);
        Tensor sync_t0 = make_tensor(sync_shapes, 1, DataType::INT32);

        PTOParam params_barrier0[] = {
            make_input_param(barrier_t),
            make_scalar_param(device_ctx_ptr),
            make_scalar_param(static_cast<uint64_t>(n_ranks)),
            make_scalar_param(static_cast<uint64_t>(root)),
            make_input_param(barrier_t),
            make_output_param(sync_t0),
        };
        pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier0, 6);

        Tensor prev_barrier_sync = sync_t0;

        for (int iter = 0; iter < N_ITER; ++iter) {
            Tensor sync_after_barrier = make_tensor(sync_shapes, 1, DataType::INT32);

            PTOParam params_wmin[] = {
                make_output_param(win_src_t),
                make_input_param(dev_src_t),
                make_scalar_param(static_cast<uint64_t>(gather_count)),
                make_input_param(prev_barrier_sync),
            };
            pto2_rt_submit_task(rt, FUNC_WIN_MEMCOPY_IN, PTO2_WORKER_VECTOR, params_wmin, 4);

            PTOParam params_barrier[] = {
                make_input_param(barrier_t),
                make_scalar_param(device_ctx_ptr),
                make_scalar_param(static_cast<uint64_t>(n_ranks)),
                make_scalar_param(static_cast<uint64_t>(root)),
                make_input_param(win_src_t),
                make_output_param(sync_after_barrier),
            };
            pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier, 6);

            Tensor gather_out = (rank_id == root && iter == N_ITER - 1)
                ? win_dst_t
                : make_tensor(dst_shapes, 1, DataType::FLOAT32);

            void* iter_debug_ptr = (rank_id == root && dev_debug != nullptr)
                ? reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dev_debug)
                    + static_cast<size_t>(iter) * static_cast<size_t>(n_ranks) * sizeof(int32_t))
                : nullptr;
            Tensor debug_out = (iter_debug_ptr != nullptr)
                ? make_tensor_external(iter_debug_ptr, debug_row_shapes, 1, DataType::INT32)
                : make_tensor(debug_row_shapes, 1, DataType::INT32);

            PTOParam params_gather[] = {
                make_output_param(gather_out),
                make_input_param(win_src_t),
                make_input_param(sync_after_barrier),
                make_scalar_param(device_ctx_ptr),
                make_scalar_param(static_cast<uint64_t>(n_ranks)),
                make_scalar_param(static_cast<uint64_t>(root)),
                make_scalar_param(sdma_workspace_ptr),
                make_output_param(debug_out),
            };
            if (strategy == 1) {
                pto2_rt_submit_task(rt, FUNC_GATHER_SYNC, PTO2_WORKER_VECTOR, params_gather, 8);
            } else if (strategy == 2) {
                pto2_rt_submit_task(rt, FUNC_GATHER_ASYNC, PTO2_WORKER_VECTOR, params_gather, 8);
            } else {
                int32_t gather_variants[] = {FUNC_GATHER_SYNC, FUNC_GATHER_ASYNC};
                pto2_rt_submit_variant_task(rt, gather_variants, 2, PTO2_WORKER_VECTOR, params_gather, 8);
            }

            Tensor wmout_dst = (rank_id == root && iter == N_ITER - 1)
                ? dev_out_t
                : make_tensor(dst_shapes, 1, DataType::FLOAT32);
            PTOParam params_wmout[] = {
                make_output_param(wmout_dst),
                make_input_param(gather_out),
                make_scalar_param(static_cast<uint64_t>(n_ranks) * static_cast<uint64_t>(gather_count)),
            };
            pto2_rt_submit_task(rt, FUNC_WIN_MEMCOPY_OUT, PTO2_WORKER_VECTOR, params_wmout, 3);

            prev_barrier_sync = sync_after_barrier;
        }
    }

    LOG_INFO(rt, "one_aicore tasks submitted");
}

}  // extern "C"


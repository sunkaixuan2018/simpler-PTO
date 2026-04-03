/**
 * fake_kernel_comm_sched_extreme orchestration:
 * - single AICPU scheduling thread
 * - two concurrent communication tasks per iteration
 * - second "dummy" gather increases lane contention on same AICore dual AIV
 */

#include <stddef.h>
#include <stdint.h>
#include "pto_orchestration_api.h"

constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);
constexpr int DEFAULT_N_ITER = 200;

#define FUNC_WIN_MEMCOPY_IN    0
#define FUNC_GATHER_SYNC       1
#define FUNC_GATHER_ASYNC      2
#define FUNC_WIN_MEMCOPY_OUT   3
#define FUNC_COMM_BARRIER      4
#define FUNC_DUMMY_GATHER_SYNC 5
#define FUNC_DUMMY_GATHER_ASYNC 6

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{.expected_arg_count = 15};
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
    int n_iter = (arg_count > 13) ? static_cast<int>(args[13]) : DEFAULT_N_ITER;
    int serialize_dummy = (arg_count > 14) ? static_cast<int>(args[14]) : 1;
    if (n_iter <= 0) n_iter = DEFAULT_N_ITER;

    int gather_count = static_cast<int>(size_src / static_cast<int64_t>(sizeof(float)));
    LOG_INFO(
        rt,
        "fake_kernel_comm_sched_extreme: strategy=%d gather_count=%d n_ranks=%d rank=%d n_iter=%d serialize_dummy=%d",
        strategy, gather_count, n_ranks, rank_id, n_iter, serialize_dummy);

    size_t barrier_size = static_cast<size_t>(n_ranks) * sizeof(int32_t);
    uint64_t barrier_base_start = win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t barrier_base_pre_main = barrier_base_start + barrier_size;
    uint64_t barrier_base_post_main = barrier_base_pre_main + barrier_size;
    uint64_t barrier_base_post_dummy = barrier_base_post_main + barrier_size;
    uint64_t win_src = barrier_base_post_dummy + barrier_size;
    uint64_t win_dst = win_src + static_cast<uint64_t>(gather_count) * sizeof(float);

    uint64_t src_shapes[1] = {static_cast<uint64_t>(gather_count)};
    uint64_t dst_shapes[1] = {static_cast<uint64_t>(n_ranks) * static_cast<uint64_t>(gather_count)};
    uint64_t barrier_shapes[1] = {static_cast<uint64_t>(n_ranks)};
    uint64_t sync_shapes[1] = {1};
    uint64_t debug_all_shapes[1] = {static_cast<uint64_t>(n_iter) * static_cast<uint64_t>(n_ranks)};
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
        Tensor barrier_start_t = make_tensor_external(reinterpret_cast<void*>(barrier_base_start), barrier_shapes, 1, DataType::INT32);
        Tensor barrier_pre_main_t = make_tensor_external(reinterpret_cast<void*>(barrier_base_pre_main), barrier_shapes, 1, DataType::INT32);
        Tensor barrier_post_main_t = make_tensor_external(reinterpret_cast<void*>(barrier_base_post_main), barrier_shapes, 1, DataType::INT32);
        Tensor barrier_post_dummy_t = make_tensor_external(reinterpret_cast<void*>(barrier_base_post_dummy), barrier_shapes, 1, DataType::INT32);
        Tensor sync_t0 = make_tensor(sync_shapes, 1, DataType::INT32);

        PTOParam params_barrier0[] = {
            make_input_param(barrier_start_t),
            make_scalar_param(device_ctx_ptr),
            make_scalar_param(static_cast<uint64_t>(n_ranks)),
            make_scalar_param(static_cast<uint64_t>(root)),
            make_input_param(barrier_start_t),
            make_output_param(sync_t0),
        };
        pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier0, 6);

        Tensor prev_barrier_sync = sync_t0;

        for (int iter = 0; iter < n_iter; ++iter) {
            Tensor sync_after_barrier = make_tensor(sync_shapes, 1, DataType::INT32);
            Tensor sync_after_main = make_tensor(sync_shapes, 1, DataType::INT32);
            Tensor sync_after_dummy = make_tensor(sync_shapes, 1, DataType::INT32);

            PTOParam params_wmin[] = {
                make_output_param(win_src_t),
                make_input_param(dev_src_t),
                make_scalar_param(static_cast<uint64_t>(gather_count)),
                make_input_param(prev_barrier_sync),
            };
            pto2_rt_submit_task(rt, FUNC_WIN_MEMCOPY_IN, PTO2_WORKER_VECTOR, params_wmin, 4);

            PTOParam params_barrier[] = {
                make_input_param(barrier_pre_main_t),
                make_scalar_param(device_ctx_ptr),
                make_scalar_param(static_cast<uint64_t>(n_ranks)),
                make_scalar_param(static_cast<uint64_t>(root)),
                make_input_param(win_src_t),
                make_output_param(sync_after_barrier),
            };
            pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier, 6);

            // Keep task graph symmetric across ranks: all ranks submit main/dummy tasks.
            // Non-root kernels early-return internally, but this keeps dependency topology aligned.
            Tensor gather_out_main = (rank_id == root && iter == n_iter - 1)
                ? win_dst_t
                : make_tensor(dst_shapes, 1, DataType::FLOAT32);
            Tensor gather_out_dummy = make_tensor(dst_shapes, 1, DataType::FLOAT32);

            void* iter_debug_ptr = (rank_id == root && dev_debug != nullptr)
                ? reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(dev_debug)
                      + static_cast<size_t>(iter) * static_cast<size_t>(n_ranks) * sizeof(int32_t))
                : nullptr;
            Tensor debug_main = (iter_debug_ptr != nullptr)
                ? make_tensor_external(iter_debug_ptr, debug_row_shapes, 1, DataType::INT32)
                : make_tensor(debug_row_shapes, 1, DataType::INT32);
            Tensor debug_dummy = make_tensor(debug_row_shapes, 1, DataType::INT32);

            PTOParam params_gather_main[] = {
                make_output_param(gather_out_main),
                make_input_param(win_src_t),
                make_input_param(sync_after_barrier),
                make_scalar_param(device_ctx_ptr),
                make_scalar_param(static_cast<uint64_t>(n_ranks)),
                make_scalar_param(static_cast<uint64_t>(root)),
                make_scalar_param(sdma_workspace_ptr),
                make_output_param(debug_main),
            };
            PTOParam params_gather_dummy[] = {
                make_output_param(gather_out_dummy),
                make_input_param(win_src_t),
                make_input_param((serialize_dummy != 0) ? sync_after_main : sync_after_barrier),
                make_scalar_param(device_ctx_ptr),
                make_scalar_param(static_cast<uint64_t>(n_ranks)),
                make_scalar_param(static_cast<uint64_t>(root)),
                make_scalar_param(sdma_workspace_ptr),
                make_output_param(debug_dummy),
            };

            if (strategy == 1) {
                pto2_rt_submit_task(rt, FUNC_GATHER_SYNC, PTO2_WORKER_VECTOR, params_gather_main, 8);
            } else if (strategy == 2) {
                pto2_rt_submit_task(rt, FUNC_GATHER_ASYNC, PTO2_WORKER_VECTOR, params_gather_main, 8);
            } else {
                int32_t main_variants[] = {FUNC_GATHER_SYNC, FUNC_GATHER_ASYNC};
                pto2_rt_submit_variant_task(rt, main_variants, 2, PTO2_WORKER_VECTOR, params_gather_main, 8);
            }

            if (serialize_dummy != 0) {
                PTOParam params_barrier_after_main[] = {
                    make_input_param(barrier_post_main_t),
                    make_scalar_param(device_ctx_ptr),
                    make_scalar_param(static_cast<uint64_t>(n_ranks)),
                    make_scalar_param(static_cast<uint64_t>(root)),
                    make_input_param(gather_out_main),
                    make_output_param(sync_after_main),
                };
                pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier_after_main, 6);
            }

            if (strategy == 1) {
                pto2_rt_submit_task(rt, FUNC_DUMMY_GATHER_SYNC, PTO2_WORKER_VECTOR, params_gather_dummy, 8);
            } else if (strategy == 2) {
                pto2_rt_submit_task(rt, FUNC_DUMMY_GATHER_ASYNC, PTO2_WORKER_VECTOR, params_gather_dummy, 8);
            } else {
                int32_t dummy_variants[] = {FUNC_DUMMY_GATHER_SYNC, FUNC_DUMMY_GATHER_ASYNC};
                pto2_rt_submit_variant_task(rt, dummy_variants, 2, PTO2_WORKER_VECTOR, params_gather_dummy, 8);
            }

            if (rank_id == root && iter == n_iter - 1) {
                PTOParam params_wmout[] = {
                    make_output_param(dev_out_t),
                    make_input_param(gather_out_main),
                    make_scalar_param(static_cast<uint64_t>(n_ranks) * static_cast<uint64_t>(gather_count)),
                };
                pto2_rt_submit_task(rt, FUNC_WIN_MEMCOPY_OUT, PTO2_WORKER_VECTOR, params_wmout, 3);
            }

            if (serialize_dummy != 0) {
                PTOParam params_barrier_after_dummy[] = {
                    make_input_param(barrier_post_dummy_t),
                    make_scalar_param(device_ctx_ptr),
                    make_scalar_param(static_cast<uint64_t>(n_ranks)),
                    make_scalar_param(static_cast<uint64_t>(root)),
                    make_input_param(gather_out_dummy),
                    make_output_param(sync_after_dummy),
                };
                pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier_after_dummy, 6);
                prev_barrier_sync = sync_after_dummy;
            } else {
                prev_barrier_sync = gather_out_dummy;
            }
        }
    }

    LOG_INFO(rt, "fake_kernel_comm_sched_extreme tasks submitted");
}

}  // extern "C"

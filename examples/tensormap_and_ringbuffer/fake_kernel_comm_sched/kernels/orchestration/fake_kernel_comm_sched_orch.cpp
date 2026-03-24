/**
 * fake_kernel_comm_sched orchestration: scheduler-level variant selection.
 *
 * Unlike fake_kernel_comm (orchestration-level selection), this example
 * uses pto2_rt_submit_variant_task() to let the scheduler pick between
 * GatherSync (func_id=1) and GatherAsync (func_id=2) at dispatch time.
 *
 * Scheduler strategy (in aicpu_executor.cpp select_variant_kernel):
 *   - Large data (>= 4KB) → Async (SDMA bulk transfer)
 *   - Small data + all other cores busy → Async (don't block AICore)
 *   - Small data + idle cores available → Sync (lower latency)
 *
 * Args (11): [0] dev_src, [1] dev_out, [2] size_src, [3] size_out,
 *   [4] device_ctx_ptr, [5] win_in_base, [6] win_out_base,
 *   [7] n_ranks, [8] root, [9] rank_id, [10] sdma_workspace_ptr
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

constexpr int GATHER_COUNT = 256;
constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

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
        .expected_arg_count = 11,
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
    (void)args[6];
    int n_ranks = static_cast<int>(args[7]);
    int root = static_cast<int>(args[8]);
    int rank_id = static_cast<int>(args[9]);
    uint64_t sdma_workspace_ptr = args[10];

    LOG_INFO(rt, "fake_kernel_comm_sched: scheduler will select variant, n_ranks=%d rank=%d",
             n_ranks, rank_id);

    size_t barrier_size = static_cast<size_t>(n_ranks) * sizeof(int32_t);
    uint64_t barrier_base = win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t win_src = barrier_base + barrier_size;
    uint64_t win_dst = win_src + GATHER_COUNT * sizeof(float);

    uint64_t src_shapes[1] = {GATHER_COUNT};
    uint64_t dst_shapes[1] = {static_cast<uint64_t>(n_ranks) * GATHER_COUNT};
    uint64_t barrier_shapes[1] = {static_cast<uint64_t>(n_ranks)};
    uint64_t sync_shapes[1] = {1};

    Tensor dev_src_t = make_tensor_external(dev_src, src_shapes, 1, DataType::FLOAT32);
    Tensor dev_out_t = make_tensor_external(dev_out, dst_shapes, 1, DataType::FLOAT32);
    Tensor win_src_t = make_tensor_external(reinterpret_cast<void*>(win_src), src_shapes, 1, DataType::FLOAT32);
    Tensor win_dst_t = make_tensor_external(reinterpret_cast<void*>(win_dst), dst_shapes, 1, DataType::FLOAT32);

    PTO2_SCOPE(rt) {
        Tensor barrier_t = make_tensor_external(reinterpret_cast<void*>(barrier_base), barrier_shapes, 1, DataType::INT32);
        Tensor sync_t0 = make_tensor(sync_shapes, 1, DataType::INT32);
        Tensor sync_t = make_tensor(sync_shapes, 1, DataType::INT32);

        // System-start barrier: synchronize ranks before entering copyin/barrier/gather/copyout flow.
        PTOParam params_barrier0[] = {
            make_input_param(barrier_t),
            make_scalar_param(device_ctx_ptr),
            make_scalar_param(static_cast<uint64_t>(n_ranks)),
            make_scalar_param(static_cast<uint64_t>(root)),
            make_input_param(barrier_t),
            make_output_param(sync_t0),
        };
        pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier0, 6);

        PTOParam params_wmin[] = {
            make_output_param(win_src_t),
            make_input_param(dev_src_t),
            make_scalar_param(static_cast<uint64_t>(GATHER_COUNT)),
            // Dependency only: enforce barrier0 completion before copyin.
            make_input_param(sync_t0),
        };
        pto2_rt_submit_task(rt, FUNC_WIN_MEMCOPY_IN, PTO2_WORKER_VECTOR, params_wmin, 4);

        PTOParam params_barrier[] = {
            make_input_param(barrier_t),
            make_scalar_param(device_ctx_ptr),
            make_scalar_param(static_cast<uint64_t>(n_ranks)),
            make_scalar_param(static_cast<uint64_t>(root)),
            make_input_param(win_src_t),
            make_output_param(sync_t),
        };
        pto2_rt_submit_task(rt, FUNC_COMM_BARRIER, PTO2_WORKER_VECTOR, params_barrier, 6);

        if (rank_id == root) {
            PTOParam params_gather[] = {
                make_output_param(win_dst_t),
                make_input_param(win_src_t),
                make_input_param(sync_t),
                make_scalar_param(device_ctx_ptr),
                make_scalar_param(static_cast<uint64_t>(n_ranks)),
                make_scalar_param(static_cast<uint64_t>(root)),
                make_scalar_param(sdma_workspace_ptr),
            };
            // Scheduler-level selection: provide both variants, let scheduler pick
            int32_t gather_variants[] = {FUNC_GATHER_SYNC, FUNC_GATHER_ASYNC};
            pto2_rt_submit_variant_task(rt, gather_variants, 2,
                                         PTO2_WORKER_VECTOR, params_gather, 7);

            PTOParam params_wmout[] = {
                make_output_param(dev_out_t),
                make_input_param(win_dst_t),
                make_scalar_param(static_cast<uint64_t>(n_ranks * GATHER_COUNT)),
            };
            pto2_rt_submit_task(rt, FUNC_WIN_MEMCOPY_OUT, PTO2_WORKER_VECTOR, params_wmout, 3);
        }
    }

    LOG_INFO(rt, "fake_kernel_comm_sched tasks submitted");
}

}  // extern "C"

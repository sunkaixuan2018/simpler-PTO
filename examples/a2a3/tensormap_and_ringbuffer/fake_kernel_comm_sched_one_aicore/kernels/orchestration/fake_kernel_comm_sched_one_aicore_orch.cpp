#include <stdint.h>

#include "common/comm_context.h"
#include "pto_orchestration_api.h"

#define FUNC_WIN_MEMCOPY_IN 0
#define FUNC_GATHER_SYNC 1
#define FUNC_GATHER_ASYNC 2
#define FUNC_WIN_MEMCOPY_OUT 3
#define FUNC_COMM_BARRIER 4
#define FUNC_DUMMY_COMM_SYNC 5
#define FUNC_DUMMY_COMM_ASYNC 6
#define FUNC_DUMMY_WINDOW_FILL 7

static constexpr uint64_t kHybridAsyncThresholdBytes = 512 * 1024;

static int select_comm_kernel(int strategy, uint64_t bytes, int sync_func, int async_func) {
    if (strategy == 1) {
        return sync_func;
    }
    if (strategy == 2) {
        return async_func;
    }
    return (bytes >= kHybridAsyncThresholdBytes) ? async_func : sync_func;
}

static Tensor make_i32_tensor(void* ptr, uint32_t count) {
    uint32_t shape[1] = {count};
    return make_tensor_external(ptr, shape, 1, DataType::INT32);
}

static Tensor submit_barrier(
    Tensor& barrier, const Tensor& dep, CommDeviceContext* comm_ctx, int n_ranks, int root) {
    uint32_t sync_shape[1] = {1};
    TensorCreateInfo sync_ci(sync_shape, 1, DataType::INT32);

    Arg params;
    params.add_inout(barrier);
    params.add_input(dep);
    params.add_output(sync_ci);
    params.add_scalar((uint64_t)(uintptr_t)comm_ctx);
    params.add_scalar((uint64_t)n_ranks);
    params.add_scalar((uint64_t)root);
    TaskOutputTensors outs = pto2_rt_submit_aiv_task(FUNC_COMM_BARRIER, params);
    return outs.get_ref(0);
}

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 10,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    if (orch_args.scalar_count() < 10) {
        LOG_ERROR("fake_kernel_comm_sched_one_aicore expects 10 scalar args, got %d", orch_args.scalar_count());
        return;
    }

    const uint64_t* args = orch_args.scalars();
    void* dev_src = reinterpret_cast<void*>(static_cast<uintptr_t>(args[0]));
    void* dev_out = reinterpret_cast<void*>(static_cast<uintptr_t>(args[1]));
    void* win_src_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(args[2]));
    void* win_dst_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(args[3]));
    void* dummy_src_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(args[4]));
    void* dummy_dst_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(args[5]));
    void* debug_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(args[6]));
    void* barrier_base_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(args[7]));
    int64_t* config = reinterpret_cast<int64_t*>(static_cast<uintptr_t>(args[8]));
    auto* comm_ctx = reinterpret_cast<CommDeviceContext*>(static_cast<uintptr_t>(args[9]));

    if (comm_ctx == nullptr || comm_ctx->rankNum <= 1) {
        LOG_ERROR(
            "fake_kernel_comm_sched_one_aicore expects rankNum > 1, got comm_ctx=%p rankNum=%u",
            comm_ctx,
            comm_ctx ? (unsigned)comm_ctx->rankNum : 0U);
        return;
    }

    uint64_t gather_count = static_cast<uint64_t>(config[0]);
    int n_iter = static_cast<int>(config[1]);
    int strategy = static_cast<int>(config[2]);
    int serialize_dummy = static_cast<int>(config[3]);
    uint64_t dummy_comm_bytes = static_cast<uint64_t>(config[4]);
    uint64_t dummy_source_elems = static_cast<uint64_t>(config[5]);
    uint64_t dummy_buffer_elems = static_cast<uint64_t>(config[6]);
    int root = static_cast<int>(config[7]);
    int n_ranks = static_cast<int>(comm_ctx->rankNum);

    if (gather_count == 0 || n_iter <= 0 || dummy_source_elems == 0 || dummy_buffer_elems == 0) {
        LOG_ERROR(
            "invalid one_aicore config: gather_count=%lu n_iter=%d dummy_source=%lu dummy_buffer=%lu",
            (unsigned long)gather_count,
            n_iter,
            (unsigned long)dummy_source_elems,
            (unsigned long)dummy_buffer_elems);
        return;
    }
    if (root < 0 || root >= n_ranks) {
        root = 0;
    }

    uint32_t src_shape[1] = {(uint32_t)gather_count};
    uint32_t dst_shape[1] = {(uint32_t)(gather_count * (uint64_t)n_ranks)};
    uint32_t dummy_src_shape[1] = {(uint32_t)dummy_source_elems};
    uint32_t dummy_dst_shape[1] = {(uint32_t)dummy_buffer_elems};
    uint32_t debug_row_shape[1] = {(uint32_t)n_ranks};

    Tensor src = make_tensor_external(dev_src, src_shape, 1, DataType::FLOAT32);
    Tensor out = make_tensor_external(dev_out, dst_shape, 1, DataType::FLOAT32);
    Tensor win_src = make_tensor_external(win_src_ptr, src_shape, 1, DataType::FLOAT32);
    Tensor win_dst = make_tensor_external(win_dst_ptr, dst_shape, 1, DataType::FLOAT32);
    Tensor dummy_src = make_tensor_external(dummy_src_ptr, dummy_src_shape, 1, DataType::FLOAT32);
    Tensor dummy_dst = make_tensor_external(dummy_dst_ptr, dummy_dst_shape, 1, DataType::FLOAT32);

    LOG_INFO(
        "one_aicore: rank=%u/%u strategy=%d gather_count=%lu n_iter=%d dummy_comm_bytes=%lu",
        (unsigned)comm_ctx->rankId,
        (unsigned)comm_ctx->rankNum,
        strategy,
        (unsigned long)gather_count,
        n_iter,
        (unsigned long)dummy_comm_bytes);

    uintptr_t barrier_base = reinterpret_cast<uintptr_t>(barrier_base_ptr);
    uintptr_t debug_base = reinterpret_cast<uintptr_t>(debug_ptr);

    Tensor barrier0 = make_i32_tensor(reinterpret_cast<void*>(barrier_base), (uint32_t)n_ranks);
    Tensor start_sync = submit_barrier(barrier0, barrier0, comm_ctx, n_ranks, root);

    Arg copy_in_params;
    copy_in_params.add_inout(win_src);
    copy_in_params.add_input(src);
    copy_in_params.add_input(start_sync);
    copy_in_params.add_scalar(gather_count);
    pto2_rt_submit_aiv_task(FUNC_WIN_MEMCOPY_IN, copy_in_params);

    uintptr_t copy_sync_ptr = barrier_base + (uint64_t)n_ranks * sizeof(int32_t);
    Tensor copy_sync_barrier = make_i32_tensor(reinterpret_cast<void*>(copy_sync_ptr), (uint32_t)n_ranks);
    Tensor copy_sync = submit_barrier(copy_sync_barrier, win_src, comm_ctx, n_ranks, root);

    Tensor prev_gather_sync = copy_sync;
    Tensor prev_dummy_sync = copy_sync;

    for (int iter = 0; iter < n_iter; ++iter) {
        uintptr_t debug_row_ptr = debug_base + (uint64_t)iter * (uint64_t)n_ranks * sizeof(int32_t);
        Tensor debug_row = make_i32_tensor(reinterpret_cast<void*>(debug_row_ptr), (uint32_t)n_ranks);

        bool is_final_iter = (iter == n_iter - 1);
        Tensor gather_result = win_dst;
        int gather_func = select_comm_kernel(
            strategy, gather_count * sizeof(float), FUNC_GATHER_SYNC, FUNC_GATHER_ASYNC);

        Arg gather_params;
        gather_params.add_inout(win_dst);
        gather_params.add_input(win_src);
        gather_params.add_input(prev_gather_sync);
        gather_params.add_output(debug_row);
        gather_params.add_scalar((uint64_t)(uintptr_t)comm_ctx);
        gather_params.add_scalar((uint64_t)n_ranks);
        gather_params.add_scalar((uint64_t)root);
        gather_params.add_scalar((uint64_t)comm_ctx->workSpace);
        pto2_rt_submit_aiv_task(gather_func, gather_params);
        gather_result = win_dst;

        if (!is_final_iter) {
            uintptr_t barrier_ptr =
                barrier_base + (uint64_t)(iter + 2) * (uint64_t)n_ranks * sizeof(int32_t);
            Tensor barrier = make_i32_tensor(reinterpret_cast<void*>(barrier_ptr), (uint32_t)n_ranks);
            prev_gather_sync = submit_barrier(barrier, gather_result, comm_ctx, n_ranks, root);
        } else {
            prev_gather_sync = gather_result;
        }

        TensorCreateInfo dummy_debug_ci(debug_row_shape, 1, DataType::INT32);
        const Tensor& dummy_dep = (serialize_dummy != 0) ? prev_gather_sync : prev_dummy_sync;
        int dummy_func = select_comm_kernel(
            strategy, dummy_comm_bytes, FUNC_DUMMY_COMM_SYNC, FUNC_DUMMY_COMM_ASYNC);

        Arg dummy_params;
        dummy_params.add_inout(dummy_dst);
        dummy_params.add_input(dummy_src);
        dummy_params.add_input(dummy_dep);
        dummy_params.add_output(dummy_debug_ci);
        dummy_params.add_scalar((uint64_t)(uintptr_t)comm_ctx);
        dummy_params.add_scalar((uint64_t)n_ranks);
        dummy_params.add_scalar((uint64_t)root);
        dummy_params.add_scalar(dummy_comm_bytes);
        pto2_rt_submit_aiv_task(dummy_func, dummy_params);
        prev_dummy_sync = dummy_dst;
    }

    Arg copy_out_params;
    copy_out_params.add_output(out);
    copy_out_params.add_input(win_dst);
    copy_out_params.add_scalar(gather_count * (uint64_t)n_ranks);
    pto2_rt_submit_aiv_task(FUNC_WIN_MEMCOPY_OUT, copy_out_params);

    LOG_INFO("one_aicore: rank=%u submitted %d iterations", (unsigned)comm_ctx->rankId, n_iter);
}

}  // extern "C"

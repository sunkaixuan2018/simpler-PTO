/**
 * batch_PA_AR_distributed_v1 orchestration
 *
 * Flow:
 * 1. Split the batch into IN_CORE_BATCH-sized chunks.
 * 2. Each chunk runs its own local batch paged attention subgraph into the
 *    corresponding local_out slice.
 * 3. As soon as one chunk becomes ready, submit a peer notification for that
 *    chunk and gate the chunk-local allreduce on the matching counter slot.
 * 4. The final AIV kernel directly reads the peer chunk from the remote window
 *    and writes local_chunk + peer_local_chunk into the out slice.
 */

#include <stddef.h>
#include <stdint.h>

#include "common/comm_context.h"
#include "pto_orchestration_api.h"

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5
#define FUNC_PA_NOTIFY_READY 6
#define FUNC_ALLREDUCE_ADD 7
#define FUNC_COMM_BARRIER 8

static uint64_t float_to_u64(float f) {
    union {
        float f32;
        uint64_t u64;
    } conv;
    conv.u64 = 0;
    conv.f32 = f;
    return conv.u64;
}

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
void aicpu_orchestration_entry(uint64_t* args, int arg_count,
                               int orch_thread_num, int orch_thread_index) {
    (void)arg_count;
    (void)orch_thread_num;
    if (orch_thread_index != 0) return;

    void* host_query = (void*)(uintptr_t)args[0];
    void* host_key_cache = (void*)(uintptr_t)args[1];
    void* host_value_cache = (void*)(uintptr_t)args[2];
    int* host_block_table = (int*)(uintptr_t)args[3];
    int* host_context_lens = (int*)(uintptr_t)args[4];
    void* host_local_out = (void*)(uintptr_t)args[5];
    void* host_out = (void*)(uintptr_t)args[6];
    int64_t* host_config = (int64_t*)(uintptr_t)args[7];
    void* host_notify_counter = (void*)(uintptr_t)args[8];
    void* host_comm_barrier = (void*)(uintptr_t)args[9];
    auto* comm_ctx = reinterpret_cast<CommDeviceContext*>((uintptr_t)args[10]);

    uint64_t batch = (uint64_t)(int)host_config[0];
    uint64_t num_heads = (uint64_t)(int)host_config[1];
    uint64_t head_dim = (uint64_t)(int)host_config[3];
    uint64_t block_size = (uint64_t)(int)host_config[4];
    uint64_t block_num = (uint64_t)(int)host_config[5];
    uint64_t total_blocks = (uint64_t)(int)host_config[7];

    union { uint32_t u; float f; } scale_conv;
    scale_conv.u = (uint32_t)host_config[6];
    float scale_value = scale_conv.f;

    if (comm_ctx == nullptr || comm_ctx->rankNum <= 1) {
        LOG_ERROR("batch_PA_AR_distributed_v1 expects rankNum > 1, got comm_ctx=%p rankNum=%u",
                  comm_ctx, comm_ctx ? (unsigned)comm_ctx->rankNum : 0U);
        return;
    }

    uint64_t q_tile = 16;
    uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;
    DataType data_type = DataType::FLOAT16;

    uint32_t query_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    uint32_t key_cache_shapes[2] = {(uint32_t)(total_blocks * block_size), (uint32_t)head_dim};
    uint32_t value_cache_shapes[2] = {(uint32_t)(total_blocks * block_size), (uint32_t)head_dim};
    uint32_t out_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};

    Tensor query = make_tensor_external(host_query, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(host_key_cache, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor local_out = make_tensor_external(host_local_out, out_shapes, 2, DataType::FLOAT32);
    Tensor out = make_tensor_external(host_out, out_shapes, 2, DataType::FLOAT32);
    uint32_t barrier_shapes[1] = {(uint32_t)comm_ctx->rankNum};
    uint32_t sync_shapes[1] = {1};
    Tensor comm_barrier = make_tensor_external(host_comm_barrier, barrier_shapes, 1, DataType::INT32);
    TensorCreateInfo barrier_sync_ci(sync_shapes, 1, DataType::INT32);

    Arg params_barrier;
    params_barrier.add_input(comm_barrier);
    params_barrier.add_output(barrier_sync_ci);
    params_barrier.add_scalar((uint64_t)(uintptr_t)comm_ctx);
    params_barrier.add_scalar((uint64_t)comm_ctx->rankNum);
    params_barrier.add_scalar((uint64_t)0);
    TaskOutputTensors barrier_outs = pto2_rt_submit_aiv_task(FUNC_COMM_BARRIER, params_barrier);
    const Tensor &barrier_sync = barrier_outs.get_ref(0);

    uint64_t bt_addr = (uint64_t)(uintptr_t)host_block_table;
    uint64_t cl_addr = (uint64_t)(uintptr_t)host_context_lens;
    uintptr_t notify_counter_base = (uintptr_t)host_notify_counter;
    uintptr_t local_out_base = (uintptr_t)host_local_out;
    uintptr_t out_base = (uintptr_t)host_out;

    uint64_t max_bn = 0;
    for (uint64_t b = 0; b < batch; ++b) {
        uint64_t cur_seq = (uint64_t)host_context_lens[b];
        uint64_t bn_b = (cur_seq + block_size - 1) / block_size;
        if (bn_b > max_bn) max_bn = bn_b;
    }

    uint64_t IN_CORE_BATCH = 16;
    uint64_t num_chunks = (batch + IN_CORE_BATCH - 1) / IN_CORE_BATCH;

    LOG_INFO("batch_PA_AR_distributed_v1: rank=%u batch=%lu num_heads=%lu chunks=%lu",
             (unsigned)comm_ctx->rankId, (unsigned long)batch, (unsigned long)num_heads,
             (unsigned long)num_chunks);

    for (uint64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        uint64_t chunk_bc = batch - chunk_idx * IN_CORE_BATCH;
        if (chunk_bc > IN_CORE_BATCH) chunk_bc = IN_CORE_BATCH;
        uint64_t global_batch_start = chunk_idx * IN_CORE_BATCH;
        uint64_t chunk_rows = chunk_bc * num_heads;
        uint64_t chunk_elems = chunk_rows * head_dim;
        uintptr_t chunk_local_out_ptr = local_out_base + chunk_idx * IN_CORE_BATCH * num_heads * head_dim * sizeof(float);
        uintptr_t chunk_out_ptr = out_base + chunk_idx * IN_CORE_BATCH * num_heads * head_dim * sizeof(float);
        uintptr_t chunk_notify_ptr = notify_counter_base + chunk_idx * sizeof(int32_t);

        uint32_t chunk_out_shapes[2] = {(uint32_t)chunk_rows, (uint32_t)head_dim};
        Tensor chunk_local_out = make_tensor_external((void*)chunk_local_out_ptr, chunk_out_shapes, 2, DataType::FLOAT32);
        Tensor chunk_out = make_tensor_external((void*)chunk_out_ptr, chunk_out_shapes, 2, DataType::FLOAT32);

        for (uint64_t q_idx = 0; q_idx < q_loop; ++q_idx) {
            uint64_t q_offset = q_idx * q_tile;

            PTO2_SCOPE() {
                uint32_t oi_acc_shapes[2] = {(uint32_t)(chunk_bc * q_tile), (uint32_t)head_dim};
                uint32_t scalar_acc_shapes[1] = {(uint32_t)(chunk_bc * q_tile)};
                TensorCreateInfo oi_batch_ci(oi_acc_shapes, 2, DataType::FLOAT32);
                TensorCreateInfo scalar_acc_ci(scalar_acc_shapes, 1, DataType::FLOAT32);

                Arg params_hub;
                params_hub.add_input(barrier_sync);
                params_hub.add_output(oi_batch_ci, scalar_acc_ci, scalar_acc_ci);
                TaskOutputTensors hub_outs = pto2_rt_submit_aiv_task(FUNC_AIV_HUB, params_hub);
                const Tensor &oi_batch = hub_outs.get_ref(0);
                const Tensor &li_batch = hub_outs.get_ref(1);
                const Tensor &mi_batch = hub_outs.get_ref(2);

                for (uint64_t bn = 0; bn < max_bn; ++bn) {
                    uint32_t sij_shapes[2] = {(uint32_t)(chunk_bc * q_tile), (uint32_t)block_size};
                    uint32_t vec_shapes[1] = {(uint32_t)(chunk_bc * q_tile)};
                    uint32_t oi_new_shapes[2] = {(uint32_t)(chunk_bc * q_tile), (uint32_t)head_dim};

                    TensorCreateInfo sij_ci(sij_shapes, 2, DataType::FLOAT32);
                    TensorCreateInfo pij_ci(sij_shapes, 2, data_type);
                    TensorCreateInfo vec_ci(vec_shapes, 1, DataType::FLOAT32);
                    TensorCreateInfo oi_new_ci(oi_new_shapes, 2, DataType::FLOAT32);

                    Arg params_qk;
                    params_qk.add_input(query);
                    params_qk.add_input(key_cache);
                    params_qk.add_input(barrier_sync);
                    params_qk.add_output(sij_ci);
                    params_qk.add_scalar(bt_addr);
                    params_qk.add_scalar(chunk_bc);
                    params_qk.add_scalar(bn);
                    params_qk.add_scalar(q_offset);
                    params_qk.add_scalar(block_num);
                    params_qk.add_scalar(num_heads);
                    params_qk.add_scalar(global_batch_start);
                    TaskOutputTensors qk_outs = pto2_rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);
                    const Tensor &sij_b = qk_outs.get_ref(0);

                    Arg params_sf;
                    params_sf.add_input(sij_b);
                    params_sf.add_output(pij_ci, vec_ci, vec_ci);
                    params_sf.add_scalar(float_to_u64(scale_value));
                    params_sf.add_scalar(cl_addr);
                    params_sf.add_scalar(chunk_bc);
                    params_sf.add_scalar(bn);
                    params_sf.add_scalar(global_batch_start);
                    TaskOutputTensors sf_outs = pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);
                    const Tensor &pij_b = sf_outs.get_ref(0);
                    const Tensor &mij_b = sf_outs.get_ref(1);
                    const Tensor &lij_b = sf_outs.get_ref(2);

                    Arg params_pv;
                    params_pv.add_input(pij_b);
                    params_pv.add_input(value_cache);
                    params_pv.add_output(oi_new_ci);
                    params_pv.add_scalar(bt_addr);
                    params_pv.add_scalar(chunk_bc);
                    params_pv.add_scalar(bn);
                    params_pv.add_scalar(block_num);
                    params_pv.add_scalar(global_batch_start);
                    TaskOutputTensors pv_outs = pto2_rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);
                    const Tensor &oi_new_b = pv_outs.get_ref(0);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == max_bn - 1) ? 1 : 0;
                    Arg params_up;
                    params_up.add_input(mij_b);
                    params_up.add_input(lij_b);
                    params_up.add_input(oi_new_b);
                    params_up.add_inout(mi_batch);
                    params_up.add_inout(li_batch);
                    params_up.add_inout(oi_batch);
                    params_up.add_inout(chunk_local_out);
                    params_up.add_scalar(is_first);
                    params_up.add_scalar(is_last);
                    params_up.add_scalar(chunk_bc);
                    params_up.add_scalar(q_offset);
                    params_up.add_scalar(num_heads);
                    params_up.add_scalar(0);
                    pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
                }
            }
        }

        uint32_t notify_done_shape[1] = {1};
        TensorCreateInfo notify_done_ci(notify_done_shape, 1, DataType::INT32);

        Arg params_notify;
        params_notify.add_input(chunk_local_out);
        params_notify.add_output(notify_done_ci);
        params_notify.add_scalar((uint64_t)chunk_notify_ptr);
        params_notify.add_scalar((uint64_t)(uintptr_t)comm_ctx);
        TaskOutputTensors notify_outs = pto2_rt_submit_aiv_task(FUNC_PA_NOTIFY_READY, params_notify);
        const Tensor &notify_done = notify_outs.get_ref(0);

        Arg params_add;
        params_add.add_input(chunk_local_out);
        params_add.add_input(notify_done);
        params_add.add_output(chunk_out);
        params_add.add_scalar((uint64_t)(uintptr_t)comm_ctx);
        params_add.add_scalar(chunk_elems);
        params_add.add_scalar((uint64_t)chunk_notify_ptr);
        params_add.add_scalar((uint64_t)comm_ctx->rankNum - 1);
        pto2_rt_submit_aiv_task(FUNC_ALLREDUCE_ADD, params_add);
    }

    LOG_INFO("batch_PA_AR_distributed_v1: rank=%u submitted chunked PA + notify + gated add",
             (unsigned)comm_ctx->rankId);
}

}  // extern "C"

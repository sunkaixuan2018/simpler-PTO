/**
 * batch_PA_AR_distributed — 多卡分布式：本地 batch_paged_attention 仍按 16-batch chunk 计算；
 * 每个 chunk 在全部 q_idx 完成后，立即对该 chunk 的 out 子区域执行 notify -> TGET -> ADD，
 * 以与后续 chunk 的计算形成流水掩盖（多卡等价于 allreduce sum）。
 *
 * 单卡 / 仿真：arg_count == 10，与 batch_paged_attention 相同（args[8] 为 key_cache 字节数标量）。
 * 双卡：arg_count == 11，args[7] 指向设备上 int64，内容为 key_cache 字节数；args[8]=peer_out，
 * args[9]=notify_counter（window），args[10]=CommDeviceContext*。
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
#define FUNC_TGET_PEER_OUT 7
#define FUNC_ALLREDUCE_ADD 8

static uint64_t float_to_u64(float f) {
    union {
        float f32;
        uint64_t u64;
    } conv;
    conv.u64 = 0;
    conv.f32 = f;
    return conv.u64;
}

static void submit_chunk_distributed_phase(
    void* host_local_out,
    void* host_out,
    void* peer_out_ptr,
    void* notify_counter_ptr,
    CommDeviceContext* comm_ctx,
    uint64_t batch,
    uint64_t num_heads,
    uint64_t head_dim,
    uint64_t batch_start,
    uint64_t chunk_bc,
    uint64_t chunk_idx,
    uint64_t num_chunks) {
    int my_rank = (int)comm_ctx->rankId;
    int nranks = (int)comm_ctx->rankNum;
    if (nranks <= 1) {
        return;
    }
    uint32_t out_shapes_3d[3] = {(uint32_t)batch, (uint32_t)num_heads, (uint32_t)head_dim};
    uint32_t peer_out_shapes_4d[4] = {(uint32_t)nranks, (uint32_t)batch, (uint32_t)num_heads, (uint32_t)head_dim};
    uint32_t chunk_shapes_3d[3] = {(uint32_t)chunk_bc, (uint32_t)num_heads, (uint32_t)head_dim};
    uint32_t chunk_offsets_3d[3] = {(uint32_t)batch_start, 0, 0};
    uint32_t peer_chunk_shapes_4d[4] = {1, (uint32_t)chunk_bc, (uint32_t)num_heads, (uint32_t)head_dim};
    uint64_t chunk_elems = chunk_bc * num_heads * head_dim;

    Tensor local_out = make_tensor_external(host_local_out, out_shapes_3d, 3, DataType::FLOAT32);
    Tensor out = make_tensor_external(host_out, out_shapes_3d, 3, DataType::FLOAT32);
    Tensor peer_out = make_tensor_external(peer_out_ptr, peer_out_shapes_4d, 4, DataType::FLOAT32);
    Tensor local_out_chunk = local_out.view(chunk_shapes_3d, chunk_offsets_3d);
    Tensor out_chunk = out.view(chunk_shapes_3d, chunk_offsets_3d);
    Tensor local_out_chunk_for_tget = local_out.view(chunk_shapes_3d, chunk_offsets_3d, true);

    auto* notify_base = reinterpret_cast<int32_t*>(notify_counter_ptr);
    auto* local_source_slot = notify_base + my_rank * num_chunks + chunk_idx;

    PTOParam params_init_add;
    params_init_add.add_inout(out_chunk);
    params_init_add.add_input(local_out_chunk);
    params_init_add.add_scalar(chunk_elems);
    pto2_rt_submit_aiv_task(FUNC_ALLREDUCE_ADD, params_init_add);

    for (int peer_rank = 0; peer_rank < nranks; ++peer_rank) {
        if (peer_rank == my_rank) {
            continue;
        }

        uint32_t notify_done_shape[1] = {1};
        Tensor notify_done = make_tensor(notify_done_shape, 1, DataType::FLOAT32);
        uint32_t peer_chunk_offsets_4d[4] = {(uint32_t)peer_rank, (uint32_t)batch_start, 0, 0};
        Tensor peer_out_chunk = peer_out.view(peer_chunk_shapes_4d, peer_chunk_offsets_4d);
        auto* wait_counter_slot = notify_base + peer_rank * num_chunks + chunk_idx;

        PTOParam params_notify;
        params_notify.add_input(local_out_chunk);
        params_notify.add_output(notify_done);
        params_notify.add_scalar((uint64_t)(uintptr_t)local_source_slot);
        params_notify.add_scalar((uint64_t)(uintptr_t)comm_ctx);
        params_notify.add_scalar((uint64_t)peer_rank);
        pto2_rt_submit_aiv_task(FUNC_PA_NOTIFY_READY, params_notify);

        PTOParam params_tget;
        params_tget.add_input(notify_done);
        params_tget.add_input(local_out_chunk_for_tget);
        params_tget.add_output(peer_out_chunk);
        params_tget.add_scalar((uint64_t)(uintptr_t)comm_ctx);
        params_tget.add_scalar(chunk_elems);
        params_tget.add_scalar((uint64_t)peer_rank);
        pto2_rt_expect_notification_counter(params_tget, (uint64_t)(uintptr_t)wait_counter_slot, 1);
        pto2_rt_submit_aiv_task(FUNC_TGET_PEER_OUT, params_tget);

        PTOParam params_add;
        params_add.add_inout(out_chunk);
        params_add.add_input(peer_out_chunk);
        params_add.add_scalar(chunk_elems);
        pto2_rt_submit_aiv_task(FUNC_ALLREDUCE_ADD, params_add);
    }
}

static void run_batch_paged_attention_phase(
    uint64_t* args, size_t key_cache_size, int orch_thread_num, int orch_thread_index, bool distributed) {
    void* host_query = (void*)(uintptr_t)args[0];
    void* host_key_cache = (void*)(uintptr_t)args[1];
    void* host_value_cache = (void*)(uintptr_t)args[2];
    int* host_block_table = (int*)(uintptr_t)args[3];
    int* host_context_lens = (int*)(uintptr_t)args[4];
    void* host_local_out = distributed ? (void*)(uintptr_t)args[5] : (void*)(uintptr_t)args[5];
    void* host_out = distributed ? (void*)(uintptr_t)args[6] : (void*)(uintptr_t)args[5];
    int64_t* host_config = distributed ? (int64_t*)(uintptr_t)args[7] : (int64_t*)(uintptr_t)args[6];

    uint64_t batch = (uint64_t)(int)host_config[0];
    uint64_t num_heads = (uint64_t)(int)host_config[1];
    uint64_t head_dim = (uint64_t)(int)host_config[3];
    uint64_t block_size = (uint64_t)(int)host_config[4];
    uint64_t block_num = (uint64_t)(int)host_config[5];
    union {
        uint32_t u;
        float f;
    } scale_conv;
    scale_conv.u = (uint32_t)host_config[6];
    float scale_value = scale_conv.f;

    uint64_t q_tile = 16;
    uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;
    DataType data_type = DataType::FLOAT16;
    uint64_t elem_size = get_element_size(data_type);

    uint64_t max_bn = 0;
    for (uint64_t b = 0; b < batch; b++) {
        uint64_t cur_seq = (uint64_t)host_context_lens[b];
        uint64_t bn_b = (cur_seq + block_size - 1) / block_size;
        if (bn_b > max_bn) max_bn = bn_b;
    }

    uint32_t query_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};
    uint64_t kv_total_rows = key_cache_size / (head_dim * elem_size);
    uint32_t key_cache_shapes[2] = {(uint32_t)kv_total_rows, (uint32_t)head_dim};
    uint32_t value_cache_shapes[2] = {(uint32_t)kv_total_rows, (uint32_t)head_dim};
    uint32_t out_shapes[2] = {(uint32_t)(batch * num_heads), (uint32_t)head_dim};

    Tensor query = make_tensor_external(host_query, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(host_key_cache, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(host_local_out, out_shapes, 2, DataType::FLOAT32);
    void* peer_out_ptr = distributed ? (void*)(uintptr_t)args[9] : nullptr;
    void* notify_counter_ptr = distributed ? (void*)(uintptr_t)args[10] : nullptr;
    auto* comm_ctx = distributed ? reinterpret_cast<CommDeviceContext*>((uintptr_t)args[11]) : nullptr;

    uint64_t bt_addr = (uint64_t)(uintptr_t)host_block_table;
    uint64_t cl_addr = (uint64_t)(uintptr_t)host_context_lens;

    uint64_t IN_CORE_BATCH = 16;
    uint64_t num_chunks = (batch + IN_CORE_BATCH - 1) / IN_CORE_BATCH;

    for (uint64_t chunk_idx = orch_thread_index; chunk_idx < num_chunks; chunk_idx += orch_thread_num) {
        uint64_t chunk_bc = batch - chunk_idx * IN_CORE_BATCH;
        if (chunk_bc > IN_CORE_BATCH) chunk_bc = IN_CORE_BATCH;
        uint64_t batch_start = chunk_idx * IN_CORE_BATCH;

        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            uint64_t q_offset = q_idx * q_tile;

            PTO2_SCOPE() {
                uint32_t oi_acc_shapes[2] = {(uint32_t)(chunk_bc * q_tile), (uint32_t)head_dim};
                uint32_t scalar_acc_shapes[1] = {(uint32_t)(chunk_bc * q_tile)};
                Tensor oi_batch = make_tensor(oi_acc_shapes, 2, DataType::FLOAT32);
                Tensor li_batch = make_tensor(scalar_acc_shapes, 1, DataType::FLOAT32);
                Tensor mi_batch = make_tensor(scalar_acc_shapes, 1, DataType::FLOAT32);

                PTOParam params_hub;
                params_hub.add_output(oi_batch);
                params_hub.add_output(li_batch);
                params_hub.add_output(mi_batch);
                pto2_rt_submit_aiv_task(FUNC_AIV_HUB, params_hub);

                for (uint64_t bn = 0; bn < max_bn; bn++) {
                    uint32_t sij_shapes[2] = {(uint32_t)(chunk_bc * q_tile), (uint32_t)block_size};
                    uint32_t vec_shapes[1] = {(uint32_t)(chunk_bc * q_tile)};
                    uint32_t oi_new_shapes[2] = {(uint32_t)(chunk_bc * q_tile), (uint32_t)head_dim};

                    Tensor sij_b = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_b = make_tensor(sij_shapes, 2, data_type);
                    Tensor mij_b = make_tensor(vec_shapes, 1, DataType::FLOAT32);
                    Tensor lij_b = make_tensor(vec_shapes, 1, DataType::FLOAT32);
                    Tensor oi_new_b = make_tensor(oi_new_shapes, 2, DataType::FLOAT32);

                    PTOParam params_qk;
                    params_qk.add_input(query);
                    params_qk.add_input(key_cache);
                    params_qk.add_output(sij_b);
                    params_qk.add_scalar(bt_addr);
                    params_qk.add_scalar(chunk_bc);
                    params_qk.add_scalar(bn);
                    params_qk.add_scalar(q_offset);
                    params_qk.add_scalar(block_num);
                    params_qk.add_scalar(num_heads);
                    params_qk.add_scalar(batch_start);
                    pto2_rt_submit_aic_task(FUNC_QK_MATMUL, params_qk);

                    PTOParam params_sf;
                    params_sf.add_input(sij_b);
                    params_sf.add_output(pij_b);
                    params_sf.add_output(mij_b);
                    params_sf.add_output(lij_b);
                    params_sf.add_scalar(float_to_u64(scale_value));
                    params_sf.add_scalar(cl_addr);
                    params_sf.add_scalar(chunk_bc);
                    params_sf.add_scalar(bn);
                    params_sf.add_scalar(batch_start);
                    pto2_rt_submit_aiv_task(FUNC_SOFTMAX_PREPARE, params_sf);

                    PTOParam params_pv;
                    params_pv.add_input(pij_b);
                    params_pv.add_input(value_cache);
                    params_pv.add_output(oi_new_b);
                    params_pv.add_scalar(bt_addr);
                    params_pv.add_scalar(chunk_bc);
                    params_pv.add_scalar(bn);
                    params_pv.add_scalar(block_num);
                    params_pv.add_scalar(batch_start);
                    pto2_rt_submit_aic_task(FUNC_PV_MATMUL, params_pv);

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == max_bn - 1) ? 1 : 0;
                    PTOParam params_up;
                    params_up.add_input(mij_b);
                    params_up.add_input(lij_b);
                    params_up.add_input(oi_new_b);
                    params_up.add_inout(mi_batch);
                    params_up.add_inout(li_batch);
                    params_up.add_inout(oi_batch);
                    params_up.add_output(out);
                    params_up.add_scalar(is_first);
                    params_up.add_scalar(is_last);
                    params_up.add_scalar(chunk_bc);
                    params_up.add_scalar(q_offset);
                    params_up.add_scalar(num_heads);
                    params_up.add_scalar(batch_start);
                    pto2_rt_submit_aiv_task(FUNC_ONLINE_UPDATE, params_up);
                }
            }
        }

        if (distributed) {
            submit_chunk_distributed_phase(
                host_local_out, host_out, peer_out_ptr, notify_counter_ptr, comm_ctx, batch, num_heads, head_dim, batch_start,
                chunk_bc, chunk_idx, num_chunks);
        }
    }
}

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    int expected = 10;
    if (arg_count >= 12) {
        expected = 12;
    }
    return PTO2OrchestrationConfig{.expected_arg_count = expected};
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(uint64_t* args, int arg_count, int orch_thread_num, int orch_thread_index) {
    const bool distributed = (arg_count >= 12);
    size_t key_cache_size = 0;

    if (distributed) {
        int64_t* ksz_dev = (int64_t*)(uintptr_t)args[8];
        key_cache_size = (size_t)(*ksz_dev);
    } else {
        key_cache_size = (size_t)args[8];
    }

    int64_t* host_config = distributed ? (int64_t*)(uintptr_t)args[7] : (int64_t*)(uintptr_t)args[6];
    uint64_t batch = (uint64_t)(int)host_config[0];
    uint64_t num_heads = (uint64_t)(int)host_config[1];
    uint64_t head_dim = (uint64_t)(int)host_config[3];

    run_batch_paged_attention_phase(args, key_cache_size, orch_thread_num, orch_thread_index, distributed);

    if (!distributed) {
        LOG_INFO("batch_PA_AR_distributed (local PA only): batch=%lu num_heads=%lu",
                 (unsigned long)batch, (unsigned long)num_heads);
        return;
    }
    uint64_t in_core_batch = 16;
    uint64_t num_chunks = (batch + in_core_batch - 1) / in_core_batch;
    auto* comm_ctx = reinterpret_cast<CommDeviceContext*>((uintptr_t)args[11]);
    LOG_INFO("batch_PA_AR_distributed: rank %d submitted chunked PA + multi-peer notify + sync TGET + AR-add (chunks=%lu ranks=%u)",
             (int)comm_ctx->rankId, (unsigned long)num_chunks, (unsigned int)comm_ctx->rankNum);
}

}  // extern "C"

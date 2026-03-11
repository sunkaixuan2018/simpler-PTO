/**
 * Paged Attention + AllGather (Manual): Paged Attention → AllGather (direct RDMA).
 *
 * Phase 1: QK → Softmax → PV → OnlineUpdate (paged attention)
 * Phase 2: WindowMemCopyIn → CommBarrier(pre) → AllGatherManual → WindowMemCopyOut → CommBarrier(post)
 * All ranks get the full allgather output.
 */

#include "runtime.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>

#define FUNC_QK_MATMUL       0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL       2
#define FUNC_ONLINE_UPDATE   3
#define FUNC_WIN_MEMCOPY_IN  4
#define FUNC_ALLGATHER       5
#define FUNC_WIN_MEMCOPY_OUT 6
#define FUNC_COMM_BARRIER    7

constexpr int GATHER_COUNT = 64;
constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

extern "C" {

int build_paged_attention_allgather_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 22) {
        std::cerr << "build_paged_attention_allgather_graph: Expected at least 22 args, got " << arg_count << '\n';
        return -1;
    }

    void* host_query = reinterpret_cast<void*>(args[0]);
    void* host_key_cache = reinterpret_cast<void*>(args[1]);
    void* host_value_cache = reinterpret_cast<void*>(args[2]);
    int* host_block_table = reinterpret_cast<int*>(args[3]);
    int* host_context_lens = reinterpret_cast<int*>(args[4]);
    void* host_attn_out = reinterpret_cast<void*>(args[5]);
    void* host_allgather_out = reinterpret_cast<void*>(args[6]);
    int64_t* host_config = reinterpret_cast<int64_t*>(args[7]);

    size_t query_size = static_cast<size_t>(args[8]);
    size_t key_cache_size = static_cast<size_t>(args[9]);
    size_t value_cache_size = static_cast<size_t>(args[10]);
    size_t block_table_size = static_cast<size_t>(args[11]);
    size_t context_lens_size = static_cast<size_t>(args[12]);
    size_t attn_out_size = static_cast<size_t>(args[13]);
    size_t allgather_out_size = static_cast<size_t>(args[14]);
    size_t config_size = static_cast<size_t>(args[15]);
    uint64_t device_ctx_ptr = args[16];
    uint64_t win_in_base = args[17];
    uint64_t win_out_base = args[18];
    int n_ranks = static_cast<int>(args[19]);
    int root = static_cast<int>(args[20]);
    int rank_id = static_cast<int>(args[21]);

    int batch = static_cast<int>(host_config[0]);
    int num_heads = static_cast<int>(host_config[1]);
    int kv_head_num = static_cast<int>(host_config[2]);
    int head_dim = static_cast<int>(host_config[3]);
    int block_size = static_cast<int>(host_config[4]);
    int max_num_blocks = static_cast<int>(host_config[5]);
    uint64_t scale_value_bits = static_cast<uint64_t>(host_config[6]);

    int q_tile_size = std::min(num_heads, 128);
    int num_head_tiles = (num_heads + q_tile_size - 1) / q_tile_size;

    std::cout << "\n=== build_paged_attention_allgather_graph (Manual) ===" << '\n';
    std::cout << "  n_ranks=" << n_ranks << " rank_id=" << rank_id << '\n';

    void* dev_query = runtime->host_api.device_malloc(query_size);
    void* dev_key_cache = runtime->host_api.device_malloc(key_cache_size);
    void* dev_value_cache = runtime->host_api.device_malloc(value_cache_size);
    void* dev_attn_out = runtime->host_api.device_malloc(attn_out_size);
    void* dev_allgather_out = runtime->host_api.device_malloc(allgather_out_size);

    if (!dev_query || !dev_key_cache || !dev_value_cache || !dev_attn_out || !dev_allgather_out) {
        std::cerr << "Error: Failed to allocate device memory\n";
        return -1;
    }

    runtime->host_api.copy_to_device(dev_query, host_query, query_size);
    runtime->host_api.copy_to_device(dev_key_cache, host_key_cache, key_cache_size);
    runtime->host_api.copy_to_device(dev_value_cache, host_value_cache, value_cache_size);
    runtime->record_tensor_pair(host_attn_out, dev_attn_out, attn_out_size);
    runtime->record_tensor_pair(host_allgather_out, dev_allgather_out, allgather_out_size);

    size_t sij_size = static_cast<size_t>(q_tile_size) * block_size * sizeof(float);
    size_t pij_size = static_cast<size_t>(q_tile_size) * block_size * sizeof(uint16_t);
    size_t mij_size = static_cast<size_t>(q_tile_size) * sizeof(float);
    size_t lij_size = mij_size;
    size_t oi_new_size = static_cast<size_t>(q_tile_size) * head_dim * sizeof(float);

    int total_buffers = batch * max_num_blocks;
    void** dev_sij_arr = new void*[total_buffers];
    void** dev_pij_arr = new void*[total_buffers];
    void** dev_mij_arr = new void*[total_buffers];
    void** dev_lij_arr = new void*[total_buffers];
    void** dev_oi_new_arr = new void*[total_buffers];

    for (int i = 0; i < total_buffers; i++) {
        dev_sij_arr[i] = runtime->host_api.device_malloc(sij_size);
        dev_pij_arr[i] = runtime->host_api.device_malloc(pij_size);
        dev_mij_arr[i] = runtime->host_api.device_malloc(mij_size);
        dev_lij_arr[i] = runtime->host_api.device_malloc(lij_size);
        dev_oi_new_arr[i] = runtime->host_api.device_malloc(oi_new_size);
    }

    int total_accums = batch * num_head_tiles;
    size_t mi_size = static_cast<size_t>(q_tile_size) * sizeof(float);
    size_t li_size = mi_size;
    size_t oi_size = static_cast<size_t>(q_tile_size) * head_dim * sizeof(float);

    void** dev_mi_arr = new void*[total_accums];
    void** dev_li_arr = new void*[total_accums];
    void** dev_oi_arr = new void*[total_accums];

    for (int i = 0; i < total_accums; i++) {
        dev_mi_arr[i] = runtime->host_api.device_malloc(mi_size);
        dev_li_arr[i] = runtime->host_api.device_malloc(li_size);
        dev_oi_arr[i] = runtime->host_api.device_malloc(oi_size);
    }

    std::vector<int> last_pa_tasks;

    for (int b_idx = 0; b_idx < batch; b_idx++) {
        int cur_seq = host_context_lens[b_idx];
        int bn_this_batch = (cur_seq + block_size - 1) / block_size;

        for (int ht = 0; ht < num_head_tiles; ht++) {
            int cur_offset = ht * q_tile_size;
            uint8_t* qi_ptr = reinterpret_cast<uint8_t*>(dev_query)
                + static_cast<int64_t>(b_idx * num_heads + cur_offset) * head_dim * sizeof(uint16_t);
            uint8_t* out_ptr = reinterpret_cast<uint8_t*>(dev_attn_out)
                + static_cast<int64_t>(b_idx * num_heads + cur_offset) * head_dim * sizeof(float);
            int kv_head_idx = cur_offset / (num_heads / kv_head_num);
            int accum_idx = b_idx * num_head_tiles + ht;
            void* dev_mi = dev_mi_arr[accum_idx];
            void* dev_li = dev_li_arr[accum_idx];
            void* dev_oi = dev_oi_arr[accum_idx];

            int t_up_prev = -1;

            for (int bn = 0; bn < bn_this_batch; bn++) {
                int cur_block_idx = host_block_table[b_idx * max_num_blocks + bn];
                uint8_t* kj_ptr = reinterpret_cast<uint8_t*>(dev_key_cache)
                    + (static_cast<int64_t>(cur_block_idx) * block_size * kv_head_num + kv_head_idx)
                      * head_dim * sizeof(uint16_t);
                uint8_t* vj_ptr = reinterpret_cast<uint8_t*>(dev_value_cache)
                    + (static_cast<int64_t>(cur_block_idx) * block_size * kv_head_num + kv_head_idx)
                      * head_dim * sizeof(uint16_t);

                int buf_idx = b_idx * max_num_blocks + bn;
                void* dev_sij = dev_sij_arr[buf_idx];
                void* dev_pij = dev_pij_arr[buf_idx];
                void* dev_mij = dev_mij_arr[buf_idx];
                void* dev_lij = dev_lij_arr[buf_idx];
                void* dev_oi_new = dev_oi_new_arr[buf_idx];

                uint64_t qk_args[6] = {
                    reinterpret_cast<uint64_t>(qi_ptr),
                    reinterpret_cast<uint64_t>(kj_ptr),
                    reinterpret_cast<uint64_t>(dev_sij),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(head_dim),
                    static_cast<uint64_t>(block_size)
                };
                int t_qk = runtime->add_task(qk_args, 6, FUNC_QK_MATMUL, CoreType::AIC);

                uint64_t sf_args[7] = {
                    reinterpret_cast<uint64_t>(dev_sij),
                    scale_value_bits,
                    reinterpret_cast<uint64_t>(dev_pij),
                    reinterpret_cast<uint64_t>(dev_mij),
                    reinterpret_cast<uint64_t>(dev_lij),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(block_size)
                };
                int t_sf = runtime->add_task(sf_args, 7, FUNC_SOFTMAX_PREPARE, CoreType::AIV);

                uint64_t pv_args[6] = {
                    reinterpret_cast<uint64_t>(dev_pij),
                    reinterpret_cast<uint64_t>(vj_ptr),
                    reinterpret_cast<uint64_t>(dev_oi_new),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(block_size),
                    static_cast<uint64_t>(head_dim)
                };
                int t_pv = runtime->add_task(pv_args, 6, FUNC_PV_MATMUL, CoreType::AIC);

                runtime->add_successor(t_qk, t_sf);
                runtime->add_successor(t_sf, t_pv);

                int is_first = (bn == 0) ? 1 : 0;
                int is_last = (bn == bn_this_batch - 1) ? 1 : 0;

                uint64_t up_args[11] = {
                    reinterpret_cast<uint64_t>(dev_mij),
                    reinterpret_cast<uint64_t>(dev_lij),
                    reinterpret_cast<uint64_t>(dev_oi_new),
                    reinterpret_cast<uint64_t>(dev_mi),
                    reinterpret_cast<uint64_t>(dev_li),
                    reinterpret_cast<uint64_t>(dev_oi),
                    static_cast<uint64_t>(is_first),
                    static_cast<uint64_t>(is_last),
                    reinterpret_cast<uint64_t>(out_ptr),
                    static_cast<uint64_t>(q_tile_size),
                    static_cast<uint64_t>(head_dim)
                };
                int t_up = runtime->add_task(up_args, 11, FUNC_ONLINE_UPDATE, CoreType::AIV);

                runtime->add_successor(t_pv, t_up);
                if (t_up_prev >= 0) {
                    runtime->add_successor(t_up_prev, t_up);
                }
                t_up_prev = t_up;
            }
            last_pa_tasks.push_back(t_up_prev);
        }
    }

    /* Phase 2: AllGather (Manual) */
    size_t barrier_size = static_cast<size_t>(n_ranks) * sizeof(int32_t);
    uint64_t barrier_base_pre = win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t barrier_base_post = barrier_base_pre + barrier_size;
    uint64_t win_src = barrier_base_post + barrier_size;
    uint64_t win_dst = win_src + GATHER_COUNT * sizeof(float);

    int32_t zeros[64] = {};
    std::memset(zeros, 0, sizeof(zeros));
    runtime->host_api.copy_to_device(reinterpret_cast<void*>(barrier_base_pre), zeros, barrier_size);
    runtime->host_api.copy_to_device(reinterpret_cast<void*>(barrier_base_post), zeros, barrier_size);

    uint64_t args_wmin[3] = {
        win_src,
        reinterpret_cast<uint64_t>(dev_attn_out),
        static_cast<uint64_t>(GATHER_COUNT)
    };
    int t_wmin = runtime->add_task(args_wmin, 3, FUNC_WIN_MEMCOPY_IN, CoreType::AIV);
    for (int t : last_pa_tasks) {
        runtime->add_successor(t, t_wmin);
    }

    uint64_t args_barrier_pre[4] = {
        barrier_base_pre, device_ctx_ptr,
        static_cast<uint64_t>(n_ranks), static_cast<uint64_t>(0)
    };
    int t_barrier_pre = runtime->add_task(args_barrier_pre, 4, FUNC_COMM_BARRIER, CoreType::AIV);
    runtime->add_successor(t_wmin, t_barrier_pre);

    uint64_t args_allgather[5] = {
        win_dst, win_src, device_ctx_ptr,
        static_cast<uint64_t>(n_ranks), static_cast<uint64_t>(rank_id)
    };
    int t_allgather = runtime->add_task(args_allgather, 5, FUNC_ALLGATHER, CoreType::AIV);
    runtime->add_successor(t_barrier_pre, t_allgather);

    uint64_t args_wmout[3] = {
        reinterpret_cast<uint64_t>(dev_allgather_out),
        win_dst,
        static_cast<uint64_t>(n_ranks * GATHER_COUNT)
    };
    int t_wmout = runtime->add_task(args_wmout, 3, FUNC_WIN_MEMCOPY_OUT, CoreType::AIV);
    runtime->add_successor(t_allgather, t_wmout);

    uint64_t args_barrier_post[4] = {
        barrier_base_post, device_ctx_ptr,
        static_cast<uint64_t>(n_ranks), static_cast<uint64_t>(0)
    };
    int t_barrier_post = runtime->add_task(args_barrier_post, 4, FUNC_COMM_BARRIER, CoreType::AIV);
    runtime->add_successor(t_wmout, t_barrier_post);

    delete[] dev_sij_arr;
    delete[] dev_pij_arr;
    delete[] dev_mij_arr;
    delete[] dev_lij_arr;
    delete[] dev_oi_new_arr;
    delete[] dev_mi_arr;
    delete[] dev_li_arr;
    delete[] dev_oi_arr;

    std::cout << "Created paged_attention_allgather (Manual) graph\n";
    runtime->print_runtime();

    return 0;
}

}  // extern "C"

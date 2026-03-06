/**
 * cpt_and_comm orchestration: GEMM -> WindowMemCopyIn -> TGATHER -> WindowMemCopyOut (root only).
 *
 * Args: host_A, host_B, host_C, host_out, size_A, size_B, size_C, size_out,
 *       device_ctx_ptr, win_in_base, win_out_base, n_ranks, root, rank_id
 *
 * Window layout (per rank, relative to win_in_base):
 *   [0 .. HCCL_WIN_SYNC_PREFIX)                 : HCCL sync metadata
 *   [SYNC .. SYNC + GATHER_COUNT*4)             : win_src  (local GEMM slice, RDMA-readable)
 *   [SYNC + GATHER_COUNT*4 .. SYNC + (1+n_ranks)*GATHER_COUNT*4) : win_dst (gather result, root only)
 */

#include "runtime.h"
#include <iostream>
#include <cstdint>

extern "C" {

constexpr int TILE = 64;
constexpr int GATHER_COUNT = 64;
constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

int build_cpt_and_comm_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 14) {
        std::cerr << "build_cpt_and_comm_graph: Expected at least 14 args, got " << arg_count << '\n';
        return -1;
    }

    void* host_A = reinterpret_cast<void*>(args[0]);
    void* host_B = reinterpret_cast<void*>(args[1]);
    void* host_C = reinterpret_cast<void*>(args[2]);
    void* host_out = reinterpret_cast<void*>(args[3]);
    size_t size_A = static_cast<size_t>(args[4]);
    size_t size_B = static_cast<size_t>(args[5]);
    size_t size_C = static_cast<size_t>(args[6]);
    size_t size_out = static_cast<size_t>(args[7]);
    uint64_t device_ctx_ptr = args[8];
    uint64_t win_in_base = args[9];
    uint64_t win_out_base = args[10];
    int n_ranks = static_cast<int>(args[11]);
    int root = static_cast<int>(args[12]);
    int rank_id = static_cast<int>(args[13]);

    std::cout << "\n=== build_cpt_and_comm_graph ===" << '\n';
    std::cout << "  n_ranks=" << n_ranks << " root=" << root << '\n';
    std::cout << "  win_in_base=0x" << std::hex << win_in_base
              << " win_out_base=0x" << win_out_base << std::dec << '\n';

    // Allocate device memory
    void* dev_A = runtime->host_api.device_malloc(size_A);
    if (!dev_A) return -1;
    runtime->host_api.copy_to_device(dev_A, host_A, size_A);

    void* dev_B = runtime->host_api.device_malloc(size_B);
    if (!dev_B) {
        runtime->host_api.device_free(dev_A);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_B, host_B, size_B);

    void* dev_C = runtime->host_api.device_malloc(size_C);
    if (!dev_C) {
        runtime->host_api.device_free(dev_A);
        runtime->host_api.device_free(dev_B);
        return -1;
    }
    runtime->host_api.copy_to_device(dev_C, host_C, size_C);

    void* dev_out = nullptr;
    if (rank_id == root) {
        dev_out = runtime->host_api.device_malloc(size_out);
        if (!dev_out) {
            runtime->host_api.device_free(dev_A);
            runtime->host_api.device_free(dev_B);
            runtime->host_api.device_free(dev_C);
            return -1;
        }
        runtime->record_tensor_pair(host_out, dev_out, size_out);
    }

    // Both win_src and win_dst are based on win_in_base (RDMA-visible region)
    uint64_t win_src = win_in_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t win_dst = win_in_base + HCCL_WIN_SYNC_PREFIX + GATHER_COUNT * sizeof(float);

    // Task 0: GEMM C = A @ B
    uint64_t args_gemm[3];
    args_gemm[0] = reinterpret_cast<uint64_t>(dev_A);
    args_gemm[1] = reinterpret_cast<uint64_t>(dev_B);
    args_gemm[2] = reinterpret_cast<uint64_t>(dev_C);
    int t0 = runtime->add_task(args_gemm, 3, 0, CoreType::AIC);

    // Task 1: WindowMemCopyIn - copy first GATHER_COUNT of dev_C to window
    uint64_t args_wmin[3];
    args_wmin[0] = win_src;
    args_wmin[1] = reinterpret_cast<uint64_t>(dev_C);
    args_wmin[2] = static_cast<uint64_t>(GATHER_COUNT);
    int t1 = runtime->add_task(args_wmin, 3, 1, CoreType::AIV);

    // Task 2: Gather - root collects from all ranks
    uint64_t args_gather[5];
    args_gather[0] = win_dst;
    args_gather[1] = win_src;
    args_gather[2] = device_ctx_ptr;
    args_gather[3] = static_cast<uint64_t>(n_ranks);
    args_gather[4] = static_cast<uint64_t>(root);
    int t2 = runtime->add_task(args_gather, 5, 2, CoreType::AIV);

    runtime->add_successor(t0, t1);
    runtime->add_successor(t1, t2);

    int t3 = -1;
    if (dev_out != nullptr) {
        // Task 3: WindowMemCopyOut - root copies gathered result to device
        uint64_t args_wmout[3];
        args_wmout[0] = reinterpret_cast<uint64_t>(dev_out);
        args_wmout[1] = win_dst;
        args_wmout[2] = static_cast<uint64_t>(n_ranks * GATHER_COUNT);
        t3 = runtime->add_task(args_wmout, 3, 3, CoreType::AIV);
        runtime->add_successor(t2, t3);
    }

    std::cout << "  task" << t0 << ": GEMM [AIC]\n";
    std::cout << "  task" << t1 << ": WindowMemCopyIn [AIV]\n";
    std::cout << "  task" << t2 << ": Gather [AIV]\n";
    if (t3 >= 0) std::cout << "  task" << t3 << ": WindowMemCopyOut [AIV]\n";

    return 0;
}

}  // extern "C"

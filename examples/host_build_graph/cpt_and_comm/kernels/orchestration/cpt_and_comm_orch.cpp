/**
 * cpt_and_comm orchestration: GEMM -> WindowMemCopyIn -> TGATHER -> WindowMemCopyOut (root only).
 *
 * Args: host_A, host_B, host_C, host_out, size_A, size_B, size_C, size_out,
 *       device_ctx_ptr, win_base, n_ranks, root, rank_id
 */

#include "runtime.h"
#include <iostream>
#include <cstdint>
#include <vector>
#include <algorithm>

extern "C" {

constexpr int TILE = 64;
constexpr int GATHER_COUNT = 64;
constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

static void DebugDumpWindow(Runtime* runtime, const char* tag, uint64_t addr, size_t count) {
    if (runtime == nullptr || runtime->host_api.copy_from_device == nullptr) {
        std::cout << "[dump] " << tag << ": runtime/copy_from_device unavailable\n";
        return;
    }
    if (addr == 0 || count == 0) {
        std::cout << "[dump] " << tag << ": invalid addr/count\n";
        return;
    }

    std::vector<float> host(count, 0.0f);
    int ret = runtime->host_api.copy_from_device(
        host.data(),
        reinterpret_cast<void*>(addr),
        count * sizeof(float)
    );
    if (ret != 0) {
        std::cout << "[dump] " << tag << ": copy_from_device failed, ret=" << ret << '\n';
        return;
    }

    size_t n_show = std::min<size_t>(count, 16);
    std::cout << "[dump] " << tag << " addr=0x" << std::hex << addr << std::dec
              << " count=" << count << " first " << n_show << ": ";
    for (size_t i = 0; i < n_show; ++i) {
        std::cout << host[i];
        if (i + 1 < n_show) {
            std::cout << ", ";
        }
    }
    std::cout << '\n';
}

int build_cpt_and_comm_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    if (arg_count < 13) {
        std::cerr << "build_cpt_and_comm_graph: Expected at least 13 args, got " << arg_count << '\n';
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
    uint64_t win_base = args[9];
    int n_ranks = static_cast<int>(args[10]);
    int root = static_cast<int>(args[11]);
    int rank_id = static_cast<int>(args[12]);
    int phase = (arg_count >= 14) ? static_cast<int>(args[13]) : -1; // -1: full, 0: gemm+wmin, 1: gather+wmout

    std::cout << "\n=== build_cpt_and_comm_graph ===" << '\n';
    std::cout << "  n_ranks=" << n_ranks << " root=" << root << " rank_id=" << rank_id << " phase=" << phase << '\n';

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

    // Window layout: sync_prefix, src (GATHER_COUNT*4), dst (n_ranks*GATHER_COUNT*4)
    uint64_t win_src = win_base + HCCL_WIN_SYNC_PREFIX;
    uint64_t win_dst = win_base + HCCL_WIN_SYNC_PREFIX + GATHER_COUNT * sizeof(float);
    std::cout << "  win_base=0x" << std::hex << win_base
              << " win_src=0x" << win_src
              << " win_dst=0x" << win_dst
              << std::dec << '\n';

    int t0 = -1, t1 = -1, t2 = -1, t3 = -1;
    bool run_phase0 = (phase != 1); // full or phase0
    bool run_phase1 = (phase != 0); // full or phase1

    if (run_phase0) {
        // Task 0: GEMM C = A @ B
        uint64_t args_gemm[3];
        args_gemm[0] = reinterpret_cast<uint64_t>(dev_A);
        args_gemm[1] = reinterpret_cast<uint64_t>(dev_B);
        args_gemm[2] = reinterpret_cast<uint64_t>(dev_C);
        t0 = runtime->add_task(args_gemm, 3, 0, CoreType::AIC);

        // Task 1: WindowMemCopyIn - copy first GATHER_COUNT of dev_C to window
        uint64_t args_wmin[3];
        args_wmin[0] = win_src;
        args_wmin[1] = reinterpret_cast<uint64_t>(dev_C);
        args_wmin[2] = static_cast<uint64_t>(GATHER_COUNT);
        t1 = runtime->add_task(args_wmin, 3, 1, CoreType::AIV);
        runtime->add_successor(t0, t1);
    }

    if (run_phase1) {
        // Task 2: Gather - root collects from all ranks
        uint64_t args_gather[5];
        args_gather[0] = win_dst;
        args_gather[1] = win_src;
        args_gather[2] = device_ctx_ptr;
        args_gather[3] = static_cast<uint64_t>(n_ranks);
        args_gather[4] = static_cast<uint64_t>(root);
        t2 = runtime->add_task(args_gather, 5, 2, CoreType::AIV);

        if (t1 >= 0) {
            runtime->add_successor(t1, t2);
        }

        if (dev_out != nullptr) {
            // Task 3: WindowMemCopyOut - root copies gathered result to device
            uint64_t args_wmout[3];
            args_wmout[0] = reinterpret_cast<uint64_t>(dev_out);
            args_wmout[1] = win_dst;
            args_wmout[2] = static_cast<uint64_t>(n_ranks * GATHER_COUNT);
            t3 = runtime->add_task(args_wmout, 3, 3, CoreType::AIV);
            runtime->add_successor(t2, t3);
        }
    }

    // Debug dump snapshot (build-time): observe win_src/win_dst memory content.
    DebugDumpWindow(runtime, "after_wmin", win_src, GATHER_COUNT);
    DebugDumpWindow(runtime, "after_tgather", win_dst, static_cast<size_t>(n_ranks * GATHER_COUNT));

    if (t0 >= 0) std::cout << "  task" << t0 << ": GEMM [AIC]\n";
    if (t1 >= 0) std::cout << "  task" << t1 << ": WindowMemCopyIn [AIV]\n";
    if (t2 >= 0) std::cout << "  task" << t2 << ": Gather [AIV]\n";
    if (t3 >= 0) std::cout << "  task" << t3 << ": WindowMemCopyOut [AIV]\n";

    return 0;
}

}  // extern "C"

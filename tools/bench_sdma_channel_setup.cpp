// Bench only the host-side SDMA/STARS channel setup cost.
// Intentionally does not build or launch any runtime/kernels.

#include <acl/acl.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "host/host_prefetch_setup.h"

static int parse_int(const char* s, int default_value) {
    if (s == nullptr || *s == '\0') return default_value;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return default_value;
    if (v < INT32_MIN) return INT32_MIN;
    if (v > INT32_MAX) return INT32_MAX;
    return static_cast<int>(v);
}

int main(int argc, char** argv) {
    const int device_id = (argc >= 2) ? parse_int(argv[1], 0) : 0;
    const int channel_count = (argc >= 3) ? parse_int(argv[2], 72) : 72;
    const int iters = (argc >= 4) ? parse_int(argv[3], 1) : 1;
    const int keep = (argc >= 5) ? parse_int(argv[4], 0) : 0;  // 1 = teardown once at end

    int rc = aclInit(nullptr);
    if (rc != 0) {
        std::cerr << "aclInit failed: rc=" << rc << "\n";
    }
    rc = aclrtSetDevice(device_id);
    if (rc != 0) {
        std::cerr << "aclrtSetDevice(" << device_id << ") failed: rc=" << rc << "\n";
        return 2;
    }

    // Baseline timing: empty loop (measurement overhead).
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        asm volatile("" ::: "memory");
    }
    auto t1 = std::chrono::steady_clock::now();

    // Setup timing. Optionally keep the workspace between iterations to measure cache/reuse.
    long long setup_only_ns = 0;
    long long teardown_only_ns = 0;
    auto t2 = std::chrono::steady_clock::now();
    void* ws = nullptr;
    for (int i = 0; i < iters; ++i) {
        auto s0 = std::chrono::steady_clock::now();
        ws = host_prefetch_setup(channel_count);
        auto s1 = std::chrono::steady_clock::now();
        if (!keep) {
            host_prefetch_teardown(ws);
            ws = nullptr;
        }
        auto s2 = std::chrono::steady_clock::now();
        setup_only_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count();
        teardown_only_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(s2 - s1).count();
    }
    if (keep) {
        auto d0 = std::chrono::steady_clock::now();
        host_prefetch_teardown(ws);
        auto d1 = std::chrono::steady_clock::now();
        teardown_only_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(d1 - d0).count();
    }
    auto t3 = std::chrono::steady_clock::now();

    const auto baseline_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const auto setup_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();

    std::cout << "device=" << device_id << " channel_count=" << channel_count
              << " iters=" << iters << " keep=" << keep << "\n";
    std::cout << "baseline_total_ns=" << baseline_ns << "\n";
    std::cout << "setup_total_ns=" << setup_ns << "\n";
    std::cout << "setup_only_ns=" << setup_only_ns << "\n";
    std::cout << "teardown_only_ns=" << teardown_only_ns << "\n";
    if (iters > 0) {
        std::cout << "baseline_avg_us=" << (baseline_ns / 1000.0) / iters << "\n";
        std::cout << "setup_avg_ms=" << (setup_ns / 1000000.0) / iters << "\n";
        std::cout << "setup_only_avg_ms=" << (setup_only_ns / 1000000.0) / iters << "\n";
        std::cout << "teardown_only_avg_ms=" << (teardown_only_ns / 1000000.0) / iters << "\n";
    }

    // Do not call aclFinalize() here; we want to minimize extra teardown noise.
    return 0;
}

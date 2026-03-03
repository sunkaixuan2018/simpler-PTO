/**
 * comm_gather entry: 4-card TGATHER, no compute.
 * Usage: comm_gather_runner [--n-ranks N] [--first-device D]
 * Default: n_ranks=4, first_device=0.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "comm_common.hpp"
#include "run_gather_kernel.cpp"

int main(int argc, char **argv)
{
    int n_ranks = 4;
    int first_device_id = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n-ranks") == 0 && i + 1 < argc) {
            n_ranks = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--first-device") == 0 && i + 1 < argc) {
            first_device_id = std::atoi(argv[++i]);
        }
    }

    std::cout << "[comm_gather] n_ranks=" << n_ranks << " first_device=" << first_device_id << std::endl;

    bool ok = RunGather<float, COMM_GATHER_COUNT>(n_ranks, n_ranks, 0, first_device_id);

    std::cout << "[comm_gather] " << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}

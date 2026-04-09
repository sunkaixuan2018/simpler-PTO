/**
 * @file device_prefetch.cpp
 * @brief SDMA Prefetch Stub for Simulation (a2a3sim)
 *
 * No-op implementation: simulation has no SDMA engine or separate L2 caches.
 */

#include "aicpu/device_prefetch.h"

void aicpu_prefetch_init(void* sdma_workspace) {
    (void)sdma_workspace;
}

void aicpu_prefetch_deinit() {
}

void aicpu_prefetch_tensor(void* addr, size_t size, int thread_idx) {
    (void)addr;
    (void)size;
    (void)thread_idx;
}

bool aicpu_prefetch_available() {
    return false;
}

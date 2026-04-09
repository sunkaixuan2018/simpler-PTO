/**
 * @file device_prefetch.h
 * @brief SDMA Prefetch Interface for AICPU
 *
 * Provides device-side SDMA prefetch to warm AICore's L2 cache before
 * kernel execution. AICPU and AICore have separate L2 caches, so this
 * uses the SDMA engine (via direct STARS SQ submission) to push data
 * into AICore's L2.
 *
 * The mechanism writes CMO PREFETCH SQEs directly to the STARS submission
 * queue and rings the hardware doorbell — same protocol as shmem but from
 * AICPU instead of AICore.
 *
 * Platform Support:
 * - a2a3: Real hardware using STARS SDMA SQ (channel info set by host)
 * - a2a3sim: No-op (no SDMA engine in simulation)
 */

#ifndef PLATFORM_DEVICE_PREFETCH_H_
#define PLATFORM_DEVICE_PREFETCH_H_

#include <cstddef>
#include <cstdint>

/**
 * Initialize SDMA prefetch subsystem.
 *
 * On a2a3: Reads STARS channel info (SQ base, doorbell register) from the
 * workspace pointer set by the host. If workspace is null, prefetch is
 * silently disabled.
 *
 * On a2a3sim: No-op.
 *
 * @param sdma_workspace  Pointer to STARS channel workspace in device GM
 *                        (set up by host via aclnnShmemSdmaStarsQuery).
 *                        NULL to disable prefetch.
 */
void aicpu_prefetch_init(void* sdma_workspace);

/**
 * Shut down SDMA prefetch subsystem.
 *
 * On a2a3: Clears state. Host is responsible for freeing workspace.
 * On a2a3sim: No-op.
 */
void aicpu_prefetch_deinit();

/**
 * Issue an asynchronous SDMA prefetch for a device memory region.
 *
 * Writes a CMO PREFETCH SQE to the STARS submission queue and rings
 * the hardware doorbell. Non-blocking: SDMA runs in background.
 *
 * Safe to call when prefetch is unavailable (no-op).
 *
 * @param addr         Device memory address to prefetch
 * @param size         Number of bytes to prefetch
 * @param thread_idx   Scheduler thread index (selects which channel to use)
 */
void aicpu_prefetch_tensor(void* addr, size_t size, int thread_idx);

/**
 * Check whether SDMA prefetch is available and enabled.
 *
 * @return true if prefetch calls will actually issue SDMA commands
 */
bool aicpu_prefetch_available();

#endif  // PLATFORM_DEVICE_PREFETCH_H_

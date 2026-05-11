/**
 * @file host_prefetch_setup.h
 * @brief Host-side SDMA prefetch channel setup
 *
 * Sets up STARS SDMA channels for AICPU-initiated prefetch.
 * Host creates device-only streams and writes their SQ/CQ IDs into a
 * device workspace. AICPU resolves SQ base/register/depth with halSqCqQuery.
 */

#ifndef PLATFORM_HOST_PREFETCH_SETUP_H_
#define PLATFORM_HOST_PREFETCH_SETUP_H_

#include <cstdint>

/**
 * Set up SDMA prefetch channels for AICPU use.
 *
 * Creates device-only streams, allocates a workspace, and writes initial
 * STARS channel info for AICPU-side HAL completion.
 *
 * @param channel_count  Number of STARS channels/streams to create.
 *
 * @return  Device GM pointer to the sdma_workspace, or nullptr on failure.
 *          Caller must free with aclrtFree when done.
 */
void* host_prefetch_setup(int channel_count);

/**
 * Tear down SDMA prefetch channels.
 *
 * Frees the workspace and destroys the stream.
 *
 * @param workspace  The pointer returned by host_prefetch_setup()
 */
void host_prefetch_teardown(void* workspace);

#endif  // PLATFORM_HOST_PREFETCH_SETUP_H_

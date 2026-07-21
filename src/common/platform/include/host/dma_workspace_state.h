/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PLATFORM_HOST_DMA_WORKSPACE_STATE_H_
#define PLATFORM_HOST_DMA_WORKSPACE_STATE_H_

#include <cstdint>

#include "common/dma_workspace.h"

/**
 * Per-runner state machine for demand-driven async-DMA workspace publication.
 *
 * The callbacks keep CANN out of this header and make the lifecycle contract
 * deterministic to unit-test: unsupported demands fail before provisioning,
 * addresses are provisioned once, and a failed publish retries the publish
 * without acquiring a second workspace.
 */
class DmaWorkspaceState {
public:
    template <typename ProvisionFn, typename PublishFn>
    int ensure(uint32_t required_mask, uint32_t supported_mask, ProvisionFn &&provision, PublishFn &&publish) {
        constexpr uint32_t kKnownMask = (uint32_t{1} << DMA_WORKSPACE_KIND_COUNT) - 1U;
        if (required_mask == 0) return 0;
        if ((required_mask & ~kKnownMask) != 0 || (required_mask & ~supported_mask) != 0) return -1;

        const uint32_t missing_mask = required_mask & ~available_mask();
        if (missing_mask != 0) {
            uint64_t provisioned[DMA_WORKSPACE_KIND_COUNT] = {0};
            int rc = provision(missing_mask, provisioned, DMA_WORKSPACE_KIND_COUNT);
            if (rc != 0) return rc;

            // Validate the entire request before committing any address. This
            // keeps mixed requirements transactional from the runner's view.
            for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind) {
                if ((missing_mask & (uint32_t{1} << kind)) != 0 && provisioned[kind] == 0) return -1;
            }
            for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind) {
                if ((missing_mask & (uint32_t{1} << kind)) != 0) addresses_[kind] = provisioned[kind];
            }
        }

        if ((required_mask & ~published_mask_) == 0) return 0;
        int rc = publish();
        if (rc != 0) return rc;
        published_mask_ = available_mask();
        return 0;
    }

    uint64_t address(int kind) const {
        if (kind < 0 || kind >= DMA_WORKSPACE_KIND_COUNT) return 0;
        return addresses_[kind];
    }

    uint32_t published_mask() const { return published_mask_; }

    void reset() {
        for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind)
            addresses_[kind] = 0;
        published_mask_ = 0;
    }

private:
    uint32_t available_mask() const {
        uint32_t mask = 0;
        for (int kind = 0; kind < DMA_WORKSPACE_KIND_COUNT; ++kind) {
            if (addresses_[kind] != 0) mask |= uint32_t{1} << kind;
        }
        return mask;
    }

    uint64_t addresses_[DMA_WORKSPACE_KIND_COUNT] = {0};
    uint32_t published_mask_{0};
};

#endif  // PLATFORM_HOST_DMA_WORKSPACE_STATE_H_

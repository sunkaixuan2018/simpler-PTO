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

#include <gtest/gtest.h>

#include <cstdint>

#include "host/dma_workspace_state.h"

namespace {

constexpr uint32_t kSdmaMask = uint32_t{1} << DMA_WORKSPACE_SDMA;
constexpr uint32_t kUrmaMask = uint32_t{1} << DMA_WORKSPACE_URMA;

struct FakeProvider {
    int provision_calls{0};
    int publish_calls{0};
    int publish_failures{0};
    bool return_zero_address{false};
    uint64_t address_seen_on_publish{0};

    int provision(uint32_t mask, uint64_t *addresses, int count) {
        ++provision_calls;
        if (!return_zero_address && count > DMA_WORKSPACE_SDMA && (mask & kSdmaMask) != 0) {
            addresses[DMA_WORKSPACE_SDMA] = 0x12340000;
        }
        return 0;
    }

    int publish() {
        ++publish_calls;
        if (publish_failures > 0) {
            --publish_failures;
            return 507018;
        }
        return 0;
    }
};

int ensure(DmaWorkspaceState &state, FakeProvider &provider, uint32_t required, uint32_t supported = kSdmaMask) {
    return state.ensure(
        required, supported,
        [&provider](uint32_t mask, uint64_t *addresses, int count) {
            return provider.provision(mask, addresses, count);
        },
        [&provider, &state]() {
            provider.address_seen_on_publish = state.address(DMA_WORKSPACE_SDMA);
            return provider.publish();
        }
    );
}

TEST(DmaWorkspaceState, UnmarkedCallableDoesNotTouchProvider) {
    DmaWorkspaceState state;
    FakeProvider provider;

    EXPECT_EQ(ensure(state, provider, 0), 0);
    EXPECT_EQ(provider.provision_calls, 0);
    EXPECT_EQ(provider.publish_calls, 0);
}

TEST(DmaWorkspaceState, FirstMarkedUseProvisionsAndPublishesExactlyOnce) {
    DmaWorkspaceState state;
    FakeProvider provider;

    ASSERT_EQ(ensure(state, provider, kSdmaMask), 0);
    EXPECT_EQ(provider.provision_calls, 1);
    EXPECT_EQ(provider.publish_calls, 1);
    EXPECT_EQ(provider.address_seen_on_publish, 0x12340000u);
    EXPECT_EQ(state.address(DMA_WORKSPACE_SDMA), 0x12340000u);
    EXPECT_EQ(state.published_mask(), kSdmaMask);

    EXPECT_EQ(ensure(state, provider, kSdmaMask), 0);
    EXPECT_EQ(provider.provision_calls, 1);
    EXPECT_EQ(provider.publish_calls, 1);
}

TEST(DmaWorkspaceState, ZeroAddressFromProviderIsNotCommittedOrPublished) {
    DmaWorkspaceState state;
    FakeProvider provider;
    provider.return_zero_address = true;

    EXPECT_NE(ensure(state, provider, kSdmaMask), 0);
    EXPECT_EQ(provider.provision_calls, 1);
    EXPECT_EQ(provider.publish_calls, 0);
    EXPECT_EQ(state.address(DMA_WORKSPACE_SDMA), 0u);
    EXPECT_EQ(state.published_mask(), 0u);
}

TEST(DmaWorkspaceState, FailedPublishRetriesWithoutReprovisioning) {
    DmaWorkspaceState state;
    FakeProvider provider;
    provider.publish_failures = 1;

    EXPECT_EQ(ensure(state, provider, kSdmaMask), 507018);
    EXPECT_EQ(provider.provision_calls, 1);
    EXPECT_EQ(provider.publish_calls, 1);
    EXPECT_EQ(state.published_mask(), 0u);

    EXPECT_EQ(ensure(state, provider, kSdmaMask), 0);
    EXPECT_EQ(provider.provision_calls, 1);
    EXPECT_EQ(provider.publish_calls, 2);
    EXPECT_EQ(state.published_mask(), kSdmaMask);
}

TEST(DmaWorkspaceState, UnsupportedOrMixedDemandFailsBeforeProvisioning) {
    DmaWorkspaceState state;
    FakeProvider provider;

    EXPECT_NE(ensure(state, provider, kUrmaMask), 0);
    EXPECT_NE(ensure(state, provider, kSdmaMask | kUrmaMask), 0);
    EXPECT_EQ(provider.provision_calls, 0);
    EXPECT_EQ(provider.publish_calls, 0);
}

TEST(DmaWorkspaceState, ResetClearsAddressesAndPublication) {
    DmaWorkspaceState state;
    FakeProvider provider;
    ASSERT_EQ(ensure(state, provider, kSdmaMask), 0);

    state.reset();

    EXPECT_EQ(state.address(DMA_WORKSPACE_SDMA), 0u);
    EXPECT_EQ(state.published_mask(), 0u);
    EXPECT_EQ(ensure(state, provider, kSdmaMask), 0);
    EXPECT_EQ(provider.provision_calls, 2);
    EXPECT_EQ(provider.publish_calls, 2);
}

}  // namespace

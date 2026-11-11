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
#include "platform_comm/ccu/ccu_completion.h"
namespace {
using namespace simpler::ccu;
TEST(CcuCompletion, ClearsDoneBeforePublishingCompletionAndSupportsReuse) {
    uint64_t event = 0;
    const auto address = reinterpret_cast<uint64_t>(&event);
    EXPECT_EQ(poll_completion(address, 1).state, CompletionPollState::pending);
    for (int i = 0; i < 1000; ++i) {
        event = 1;
        EXPECT_EQ(poll_completion(address, 1).state, CompletionPollState::ready);
        EXPECT_EQ(event, 0U);
        EXPECT_EQ(poll_completion(address, 1).state, CompletionPollState::pending);
    }
}
struct StuckIo {
    static uint64_t load(uint64_t) { return 1; }
    static void clear(uint64_t) {}
};
TEST(CcuCompletion, RejectsUnconfirmedClearAndInvalidMapping) {
    EXPECT_EQ(poll_completion<StuckIo>(8, 1).state, CompletionPollState::protocol_error);
    EXPECT_EQ(poll_completion(0, 1).state, CompletionPollState::protocol_error);
    EXPECT_EQ(poll_completion(8, 2).state, CompletionPollState::protocol_error);
    EXPECT_EQ(poll_completion(9, 1).state, CompletionPollState::protocol_error);
}
}  // namespace

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

#include "async_condition_writer.h"
#include "async_wait.h"

namespace {

struct alignas(SIMPLER_CCU_COMPLETION_SLOT_ALIGNMENT) Slot {
    SimplerCcuCompletionSlotV1 value{};
};

void publish(Slot &slot, uint64_t sequence, uint32_t status, int64_t native_error = 0) {
    slot.value.abi_version = SIMPLER_CCU_MODULE_ABI_VERSION;
    slot.value.struct_size = sizeof(SimplerCcuCompletionSlotV1);
    slot.value.status = status;
    slot.value.operation_sequence = sequence;
    slot.value.native_error = native_error;
    slot.value.completed_sequence = sequence;
}

CompletionCondition condition(Slot &slot, uint64_t sequence) {
    CompletionCondition cond;
    cond.engine = ASYNC_ENGINE_CCU;
    cond.completion_type = COMPLETION_TYPE_CCU;
    cond.addr = reinterpret_cast<uint64_t>(&slot.value);
    cond.backend_cookie = sequence;
    return cond;
}

TEST(HbgCcuCompletion, PollsAndRetiresOneConditionWith64BitSequence) {
    constexpr uint64_t kSequence = (uint64_t{1} << 40) + 7;
    Slot slot;
    auto cond = condition(slot, kSequence);

    EXPECT_EQ(cond.test().state, CompletionPollState::PENDING);
    publish(slot, kSequence, SIMPLER_CCU_COMPLETION_SUCCEEDED);
    EXPECT_EQ(cond.test().state, CompletionPollState::READY);
    cond.retire();
    EXPECT_TRUE(cond.retired);
}

TEST(HbgCcuCompletion, MapsNativeFailureAndSequenceJumpToTaskFailure) {
    Slot slot;
    auto cond = condition(slot, 9);

    publish(slot, 9, SIMPLER_CCU_COMPLETION_FAILED, 507018);
    EXPECT_EQ(cond.test().error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    publish(slot, 10, SIMPLER_CCU_COMPLETION_SUCCEEDED);
    EXPECT_EQ(cond.test().state, CompletionPollState::FAILED);
}

TEST(HbgCcuCompletion, ResidentWriterAcceptsOnlyCanonicalCcuToken) {
    Slot slot;
    CompletionToken token{
        reinterpret_cast<uint64_t>(&slot.value), 0, COMPLETION_ENGINE_CCU, COMPLETION_TYPE_CCU, uint64_t{1} << 40
    };
    EXPECT_TRUE(pto2::resident::detail::token_valid(token));
    token.expected_value = 1;
    EXPECT_FALSE(pto2::resident::detail::token_valid(token));
}

}  // namespace

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

#include "async_wait.h"

namespace {

struct alignas(SIMPLER_CCU_COMPLETION_SLOT_ALIGNMENT) Slot {
    SimplerCcuCompletionSlotV1 value{};
};

void publish(Slot &slot, uint64_t sequence) {
    slot.value.abi_version = SIMPLER_CCU_MODULE_ABI_VERSION;
    slot.value.struct_size = sizeof(SimplerCcuCompletionSlotV1);
    slot.value.status = SIMPLER_CCU_COMPLETION_SUCCEEDED;
    slot.value.operation_sequence = sequence;
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

TEST(TmrCcuCompletion, KeepsMissionSlotsIndependentInReverseCompletionOrder) {
    constexpr uint64_t kSequence = (uint64_t{1} << 40) + 3;
    Slot mission_a;
    Slot mission_b;
    auto cond_a = condition(mission_a, kSequence);
    auto cond_b = condition(mission_b, kSequence);

    publish(mission_b, kSequence);
    EXPECT_EQ(cond_a.test().state, CompletionPollState::PENDING);
    EXPECT_EQ(cond_b.test().state, CompletionPollState::READY);

    publish(mission_a, kSequence);
    EXPECT_EQ(cond_a.test().state, CompletionPollState::READY);
    cond_b.retire();
    EXPECT_TRUE(cond_b.retired);
    EXPECT_FALSE(cond_a.retired);
}

TEST(TmrCcuCompletion, RejectsSequenceJump) {
    Slot slot;
    auto cond = condition(slot, 4);
    publish(slot, 5);
    const auto result = cond.test();
    EXPECT_EQ(result.state, CompletionPollState::FAILED);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

}  // namespace

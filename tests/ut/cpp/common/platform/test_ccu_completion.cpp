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

#include "platform_comm/ccu/ccu_completion.h"

namespace {

using simpler::ccu::CompletionPollState;

struct alignas(SIMPLER_CCU_COMPLETION_SLOT_ALIGNMENT) Slot {
    SimplerCcuCompletionSlotV1 value{};
};

void publish(Slot &slot, uint32_t mission_id, uint64_t sequence, uint32_t status, int64_t native_error = 0) {
    slot.value.abi_version = SIMPLER_CCU_MODULE_ABI_VERSION;
    slot.value.struct_size = sizeof(SimplerCcuCompletionSlotV1);
    slot.value.mission_id = mission_id;
    slot.value.status = status;
    slot.value.run_generation = 7;
    slot.value.operation_sequence = sequence;
    slot.value.native_error = native_error;
    slot.value.completed_sequence = sequence;
}

TEST(CcuCompletion, UsesExactSequenceComparison) {
    Slot slot;

    EXPECT_EQ(simpler::ccu::poll_completion(&slot.value, 2).state, CompletionPollState::pending);
    publish(slot, 3, 2, SIMPLER_CCU_COMPLETION_SUCCEEDED);
    EXPECT_EQ(simpler::ccu::poll_completion(&slot.value, 2).state, CompletionPollState::ready);
    EXPECT_EQ(simpler::ccu::poll_completion(&slot.value, 1).state, CompletionPollState::protocol_error);
}

TEST(CcuCompletion, KeepsDifferentMissionSlotsIndependentInReverseCompletionOrder) {
    Slot mission_a;
    Slot mission_b;

    publish(mission_b, 4, 1, SIMPLER_CCU_COMPLETION_SUCCEEDED);
    EXPECT_EQ(simpler::ccu::poll_completion(&mission_a.value, 1).state, CompletionPollState::pending);
    EXPECT_EQ(simpler::ccu::poll_completion(&mission_b.value, 1).state, CompletionPollState::ready);

    publish(mission_a, 3, 1, SIMPLER_CCU_COMPLETION_SUCCEEDED);
    EXPECT_EQ(simpler::ccu::poll_completion(&mission_a.value, 1).state, CompletionPollState::ready);
}

TEST(CcuCompletion, PropagatesNativeFailure) {
    Slot slot;
    publish(slot, 3, 9, SIMPLER_CCU_COMPLETION_FAILED, 507018);

    const auto result = simpler::ccu::poll_completion(&slot.value, 9);
    EXPECT_EQ(result.state, CompletionPollState::failed);
    EXPECT_EQ(result.native_error, 507018);
}

TEST(CcuCompletion, RejectsMalformedCommittedSlots) {
    Slot slot;
    publish(slot, 3, 5, SIMPLER_CCU_COMPLETION_SUCCEEDED);

    slot.value.struct_size = 0;
    EXPECT_EQ(simpler::ccu::poll_completion(&slot.value, 5).state, CompletionPollState::protocol_error);
    slot.value.struct_size = sizeof(SimplerCcuCompletionSlotV1);
    slot.value.operation_sequence = 4;
    EXPECT_EQ(simpler::ccu::poll_completion(&slot.value, 5).state, CompletionPollState::protocol_error);
    slot.value.operation_sequence = 5;
    slot.value.native_error = 1;
    EXPECT_EQ(simpler::ccu::poll_completion(&slot.value, 5).state, CompletionPollState::protocol_error);
}

}  // namespace

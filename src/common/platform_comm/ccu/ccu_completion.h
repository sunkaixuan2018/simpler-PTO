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

#pragma once

#include <atomic>
#include <cstdint>

#include "ccu_native_abi.h"

namespace simpler::ccu {

enum class CompletionPollState : uint8_t { pending = 0, ready = 1, failed = 2, protocol_error = 3 };

struct CompletionPollResult {
    CompletionPollState state{CompletionPollState::pending};
    int64_t native_error{};
};

inline uint64_t load_completed_sequence(const volatile SimplerCcuCompletionSlotV1 &slot) {
#if defined(__aarch64__)
    return __atomic_load_n(&slot.completed_sequence, __ATOMIC_ACQUIRE);
#elif defined(_MSC_VER)
    const uint64_t sequence = slot.completed_sequence;
    std::atomic_thread_fence(std::memory_order_acquire);
    return sequence;
#else
    return __atomic_load_n(&slot.completed_sequence, __ATOMIC_ACQUIRE);
#endif
}

inline CompletionPollResult
poll_completion(const volatile SimplerCcuCompletionSlotV1 *slot, uint64_t expected_sequence) {
    if (slot == nullptr || reinterpret_cast<uintptr_t>(slot) % SIMPLER_CCU_COMPLETION_SLOT_ALIGNMENT != 0 ||
        expected_sequence == 0)
        return {CompletionPollState::protocol_error, 0};

    const uint64_t completed_sequence = load_completed_sequence(*slot);
    if (completed_sequence < expected_sequence) return {CompletionPollState::pending, 0};
    if (completed_sequence > expected_sequence) return {CompletionPollState::protocol_error, 0};

    if (slot->abi_version != SIMPLER_CCU_MODULE_ABI_VERSION ||
        slot->struct_size != sizeof(SimplerCcuCompletionSlotV1) || slot->operation_sequence != expected_sequence)
        return {CompletionPollState::protocol_error, 0};
    if (slot->status == SIMPLER_CCU_COMPLETION_SUCCEEDED && slot->native_error == 0)
        return {CompletionPollState::ready, 0};
    if (slot->status == SIMPLER_CCU_COMPLETION_FAILED) return {CompletionPollState::failed, slot->native_error};
    return {CompletionPollState::protocol_error, slot->native_error};
}

}  // namespace simpler::ccu

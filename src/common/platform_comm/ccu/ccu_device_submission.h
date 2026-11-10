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
#include <cstddef>
#include <cstdint>

#include "ccu_native_abi.h"

#if defined(__CCE_AICORE__)
#define SIMPLER_CCU_SUBMIT_DEVICE [aicore]
#define SIMPLER_CCU_SUBMIT_GM __gm__
#else
#define SIMPLER_CCU_SUBMIT_DEVICE
#define SIMPLER_CCU_SUBMIT_GM
#endif

namespace simpler::ccu {

struct DeviceSubmissionIo {
    static SIMPLER_CCU_SUBMIT_DEVICE uint64_t load_u64(const volatile SIMPLER_CCU_SUBMIT_GM uint64_t *address) {
#if defined(__CCE_AICORE__)
        return *address;
#elif defined(_MSC_VER)
        const uint64_t value = *address;
        std::atomic_thread_fence(std::memory_order_acquire);
        return value;
#else
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
#endif
    }

    static SIMPLER_CCU_SUBMIT_DEVICE void publish(volatile SIMPLER_CCU_SUBMIT_GM void *address, uint64_t bytes) {
#if defined(__CCE_AICORE__)
        (void)bytes;
        dcci(const_cast<SIMPLER_CCU_SUBMIT_GM void *>(address), cache_line_t::ENTIRE_DATA_CACHE);
        dsb(DSB_DDR);
#else
        (void)address;
        (void)bytes;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    }
};

template <typename Io = DeviceSubmissionIo, typename RegisterCompletion>
SIMPLER_CCU_SUBMIT_DEVICE bool submit_mission(
    const volatile SIMPLER_CCU_SUBMIT_GM SimplerCcuMissionDescriptorV1 *mission, uint64_t operation_sequence,
    const void *parameters, uint64_t parameter_size, RegisterCompletion &&register_completion
) {
    if (mission == nullptr || operation_sequence == 0 || mission->abi_version != SIMPLER_CCU_MODULE_ABI_VERSION ||
        mission->struct_size != sizeof(SimplerCcuMissionDescriptorV1) || mission->run_generation == 0 ||
        mission->parameter_addr == 0 || mission->completion_slot_addr == 0 || mission->trigger_gate_addr == 0 ||
        mission->parameter_size != parameter_size || (parameter_size != 0 && parameters == nullptr) ||
        mission->reserved != 0)
        return false;

    auto *slot = reinterpret_cast<volatile SIMPLER_CCU_SUBMIT_GM SimplerCcuCompletionSlotV1 *>(
        static_cast<uintptr_t>(mission->completion_slot_addr)
    );
    if (reinterpret_cast<uintptr_t>(slot) % SIMPLER_CCU_COMPLETION_SLOT_ALIGNMENT != 0 ||
        slot->abi_version != SIMPLER_CCU_MODULE_ABI_VERSION || slot->struct_size != sizeof(*slot) ||
        slot->mission_id != mission->mission_id || slot->run_generation != mission->run_generation)
        return false;

    const uint64_t completed_sequence = Io::load_u64(&slot->completed_sequence);
    if (completed_sequence != operation_sequence - 1) return false;
    if (completed_sequence != 0 && (slot->operation_sequence != completed_sequence ||
                                    slot->status != SIMPLER_CCU_COMPLETION_SUCCEEDED || slot->native_error != 0))
        return false;

    if (!register_completion(mission->completion_slot_addr, operation_sequence)) return false;

    slot->operation_sequence = operation_sequence;
    slot->status = SIMPLER_CCU_COMPLETION_PENDING;
    slot->native_error = 0;
    Io::publish(slot, offsetof(SimplerCcuCompletionSlotV1, completed_sequence));

    auto *destination =
        reinterpret_cast<volatile SIMPLER_CCU_SUBMIT_GM uint8_t *>(static_cast<uintptr_t>(mission->parameter_addr));
    const auto *source = static_cast<const uint8_t *>(parameters);
    for (uint64_t i = 0; i < parameter_size; ++i)
        destination[i] = source[i];
    Io::publish(destination, parameter_size);

    auto *trigger =
        reinterpret_cast<volatile SIMPLER_CCU_SUBMIT_GM uint64_t *>(static_cast<uintptr_t>(mission->trigger_gate_addr));
    *trigger = 1;
    Io::publish(trigger, sizeof(*trigger));
    return true;
}

}  // namespace simpler::ccu

#undef SIMPLER_CCU_SUBMIT_DEVICE
#undef SIMPLER_CCU_SUBMIT_GM

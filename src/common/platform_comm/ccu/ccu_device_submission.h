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
#include <cstdint>
#include "ccu_native_abi.h"

#if defined(__CCE_AICORE__)
#define SIMPLER_CCU_SUBMIT_DEVICE __aicore__
#define SIMPLER_CCU_SUBMIT_GM __gm__
#else
#define SIMPLER_CCU_SUBMIT_DEVICE
#define SIMPLER_CCU_SUBMIT_GM
#endif

namespace simpler::ccu {
inline SIMPLER_CCU_SUBMIT_DEVICE const volatile SIMPLER_CCU_SUBMIT_GM SimplerCcuControlV2 *
ccu_control(const volatile SIMPLER_CCU_SUBMIT_GM SimplerCcuWorkspaceV2 *workspace, uint32_t ccu_id) {
    if (workspace == nullptr || workspace->abi_version != SIMPLER_CCU_MODULE_ABI_VERSION ||
        workspace->struct_size != sizeof(*workspace) ||
        workspace->layout_version != SIMPLER_CCU_WORKSPACE_LAYOUT_VERSION || workspace->run_generation == 0 ||
        workspace->controls_addr == 0 || workspace->controls_addr % alignof(SimplerCcuControlV2) != 0 ||
        ccu_id >= workspace->ccu_count || workspace->control_size != sizeof(SimplerCcuControlV2) ||
        workspace->flags != 0 || workspace->reserved[0] != 0 || workspace->reserved[1] != 0)
        return nullptr;
    const auto *controls =
        reinterpret_cast<const volatile SIMPLER_CCU_SUBMIT_GM SimplerCcuControlV2 *>(workspace->controls_addr);
    const auto *control = &controls[ccu_id];
    if (control->ccu_id != ccu_id || control->run_generation != workspace->run_generation) return nullptr;
    return control;
}

struct DeviceSubmissionIo {
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
    const volatile SIMPLER_CCU_SUBMIT_GM SimplerCcuControlV2 *mission, const void *parameters, uint64_t parameter_size,
    RegisterCompletion &&register_completion
) {
    if (mission == nullptr || mission->abi_version != SIMPLER_CCU_MODULE_ABI_VERSION ||
        mission->struct_size != sizeof(*mission) || mission->run_generation == 0 || mission->flags != 0 ||
        mission->reserved0 != 0 || mission->reserved1 != 0 || mission->parameter_addr == 0 ||
        mission->parameter_alignment == 0 || (mission->parameter_alignment & (mission->parameter_alignment - 1)) != 0 ||
        mission->parameter_addr % mission->parameter_alignment != 0 || mission->parameter_size != parameter_size ||
        (parameter_size != 0 && parameters == nullptr))
        return false;
    const uint64_t addresses[]{mission->ready.address, mission->trigger.address, mission->done.address};
    if (mission->ready.mask != SIMPLER_CCU_EVENT_MASK || mission->trigger.mask != SIMPLER_CCU_EVENT_MASK ||
        mission->done.mask != SIMPLER_CCU_EVENT_MASK)
        return false;
    for (uint32_t i = 0; i < 3; ++i) {
        if (addresses[i] == 0 || addresses[i] % alignof(uint64_t) != 0) return false;
        for (uint32_t j = 0; j < i; ++j)
            if (addresses[i] == addresses[j]) return false;
    }
    if (!register_completion(mission->done.address, mission->done.mask)) return false;
    auto *destination = reinterpret_cast<volatile SIMPLER_CCU_SUBMIT_GM uint8_t *>(mission->parameter_addr);
    const auto *source = static_cast<const uint8_t *>(parameters);
    for (uint64_t i = 0; i < parameter_size; ++i)
        destination[i] = source[i];
    Io::publish(destination, parameter_size);
    auto *trigger = reinterpret_cast<volatile SIMPLER_CCU_SUBMIT_GM uint64_t *>(mission->trigger.address);
    *trigger = (uint64_t{1} << 63) | mission->trigger.mask;
    Io::publish(trigger, sizeof(*trigger));
    return true;
}
}  // namespace simpler::ccu
#undef SIMPLER_CCU_SUBMIT_DEVICE
#undef SIMPLER_CCU_SUBMIT_GM

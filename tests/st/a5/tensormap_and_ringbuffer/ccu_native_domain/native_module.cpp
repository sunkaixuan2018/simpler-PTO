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

#include <array>
#include <cstring>
#include <stdexcept>

#include "hcomm/ccu/ccu_primitives.hpp"
#include "hcomm/ccu/ccu_res.h"
#include "platform_comm/ccu/ccu_native_abi.h"

extern const uint8_t ccu_test_sdk_identity[32];
extern const uint8_t ccu_test_isa_identity[32];

namespace {
namespace ccu = AscendC::ccu;

void check(CcuResult result) {
    if (result != CCU_SUCCESS) throw std::runtime_error("native CCU instruction rejected");
}

ccu::Variable address(ccu::Variable base, uint64_t offset) {
    ccu::Variable constant;
    constant = offset;
    ccu::Variable result;
    result = base + constant;
    return result;
}

void store(ccu::Variable base, uint64_t offset, uint64_t value) {
    ccu::Variable source;
    source = value;
    check(ccu::Store(address(base, offset), source));
}

int32_t native_mission(const SimplerCcuRegistrationBindingV1 *binding) noexcept {
    try {
        std::array<ccu::Variable, SIMPLER_CCU_TASK_ARGUMENT_COUNT> args;
        for (uint32_t i = 0; i < args.size(); ++i)
            check(ccu::LoadArg(args[i], i));
        uint64_t trigger_address{};
        check(HcommCcuEventGetAddr(binding->trigger_event_handle, 0, &trigger_address));
        // A final wake on the operation gate also stops a mission with no submitted operation.
        store(args[3], offsetof(SimplerCcuRunControlV1, stop_gate_addr), trigger_address);
        const uint64_t mission = binding->kernel_id;
        check(ccu::NotifyRecord(binding->channel_handle, binding->kernel_id - 1, 1));
        check(ccu::NotifyWait(binding->channel_handle, binding->kernel_id - 1, 1));
        store(
            args[3], offsetof(SimplerCcuRunControlV1, mission_id),
            mission | (uint64_t{SIMPLER_CCU_RUN_STATE_READY} << 32)
        );
        ccu::Event trigger(binding->trigger_event_handle);
        ccu::Variable stop;
        stop = 0;
        CCU_WHILE(stop == 0) {
            check(ccu::EventWait(trigger, 1));
            check(ccu::Load(address(args[3], offsetof(SimplerCcuRunControlV1, stop_requested)), stop));
            CCU_IF(stop == 0) {
                ccu::Variable output, value, sequence;
                check(ccu::Load(args[5], output));
                check(ccu::Load(address(args[5], 8), value));
                check(ccu::Load(address(args[7], offsetof(SimplerCcuCompletionSlotV1, operation_sequence)), sequence));
                check(ccu::Store(output, value));
                store(
                    args[7], offsetof(SimplerCcuCompletionSlotV1, mission_id),
                    mission | (uint64_t{SIMPLER_CCU_COMPLETION_SUCCEEDED} << 32)
                );
                check(
                    ccu::Store(address(args[3], offsetof(SimplerCcuRunControlV1, last_completed_sequence)), sequence)
                );
                check(ccu::Store(address(args[7], offsetof(SimplerCcuCompletionSlotV1, completed_sequence)), sequence));
            }
        }
        store(
            args[3], offsetof(SimplerCcuRunControlV1, mission_id),
            mission | (uint64_t{SIMPLER_CCU_RUN_STATE_EXITED} << 32)
        );
        return CCU_SUCCESS;
    } catch (...) {
        return CCU_E_INTERNAL;
    }
}

struct Manifest {
    SimplerCcuModuleManifestV1 header;
    SimplerCcuKernelRecordV1 kernels[2];
    char symbols[26];
};

Manifest make_manifest() {
    Manifest result{};
    result.header.abi_version = SIMPLER_CCU_MODULE_ABI_VERSION;
    result.header.struct_size = sizeof(result.header);
    result.header.manifest_size = sizeof(result);
    result.header.kernel_record_size = sizeof(result.kernels[0]);
    result.header.kernel_count = 2;
    result.header.kernel_records_offset = offsetof(Manifest, kernels);
    result.header.string_table_offset = offsetof(Manifest, symbols);
    result.header.string_table_size = sizeof(result.symbols);
    std::memcpy(result.header.sdk_build_id, ccu_test_sdk_identity, 32);
    std::memcpy(result.header.pto_isa_build_id, ccu_test_isa_identity, 32);
    constexpr char names[] = "ccu_mission_accu_mission_b";
    static_assert(sizeof(names) - 1 == sizeof(result.symbols));
    std::memcpy(result.symbols, names, sizeof(result.symbols));
    for (uint32_t i = 0; i < 2; ++i) {
        result.kernels[i] = {
            sizeof(SimplerCcuKernelRecordV1),
            i + 1,
            7,
            SIMPLER_CCU_TASK_ARGUMENT_LAYOUT_VERSION,
            16,
            64,
            SIMPLER_CCU_KERNEL_CAP_COMPLETION_SLOT | SIMPLER_CCU_KERNEL_CAP_STOP_GATE,
            static_cast<uint32_t>(offsetof(Manifest, symbols)) + i * 13,
            13,
            SIMPLER_CCU_REGISTRATION_BINDING_LAYOUT_VERSION,
            sizeof(SimplerCcuRegistrationBindingV1),
            SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT,
            SIMPLER_CCU_TASK_ARGUMENT_COUNT,
            {0, 0}
        };
    }
    return result;
}
}  // namespace

extern "C" int32_t ccu_mission_a(const SimplerCcuRegistrationBindingV1 *binding) noexcept {
    return native_mission(binding);
}
extern "C" int32_t ccu_mission_b(const SimplerCcuRegistrationBindingV1 *binding) noexcept {
    return native_mission(binding);
}
extern "C" const SimplerCcuModuleManifestV1 *simpler_ccu_module_manifest_v1() noexcept {
    static const Manifest manifest = make_manifest();
    return &manifest.header;
}

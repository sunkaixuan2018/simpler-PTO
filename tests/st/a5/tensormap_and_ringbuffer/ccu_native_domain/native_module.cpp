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

int32_t native_mission(const SimplerCcuRegistrationBindingV2 *binding) noexcept {
    try {
        std::array<ccu::Variable, SIMPLER_CCU_TASK_ARGUMENT_COUNT> args;
        for (uint32_t i = 0; i < args.size(); ++i)
            check(ccu::LoadArg(args[i], i));
        ccu::Event ready(binding->ready_event_handle);
        ccu::Event trigger(binding->trigger_event_handle);
        ccu::Event done(binding->done_event_handle);
        check(ccu::NotifyRecord(binding->channel_handle, binding->ccu_id, 1));
        check(ccu::NotifyWait(binding->channel_handle, binding->ccu_id, 1));
        check(ccu::EventRecord(ready, 1));
        check(ccu::EventWait(trigger, 1));
        ccu::Variable output, value;
        check(ccu::Load(args[1], output));
        check(ccu::Load(address(args[1], 8), value));
        CCU_IF(value == 0) {
            // A finite faulty program deliberately omits done; the runtime must not release its consumer.
            store(output, 0, 2971);
        }
        CCU_ELSE {
            check(ccu::Store(output, value));
            check(ccu::NotifyRecord(binding->channel_handle, binding->ccu_id, 1));
            check(ccu::NotifyWait(binding->channel_handle, binding->ccu_id, 1));
            check(ccu::EventRecord(done, 1));
        }
        return CCU_SUCCESS;
    } catch (...) {
        return CCU_E_INTERNAL;
    }
}

struct Manifest {
    SimplerCcuModuleManifestV2 header;
    SimplerCcuKernelRecordV2 kernels[2];
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
            sizeof(SimplerCcuKernelRecordV2),
            i + 1,
            0,
            SIMPLER_CCU_TASK_ARGUMENT_LAYOUT_VERSION,
            16,
            64,
            SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS,
            static_cast<uint32_t>(offsetof(Manifest, symbols)) + i * 13,
            13,
            SIMPLER_CCU_REGISTRATION_BINDING_LAYOUT_VERSION,
            sizeof(SimplerCcuRegistrationBindingV2),
            SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT,
            SIMPLER_CCU_TASK_ARGUMENT_COUNT,
            {0, 0}
        };
    }
    return result;
}
}  // namespace

extern "C" int32_t ccu_mission_a(const SimplerCcuRegistrationBindingV2 *binding) noexcept {
    return native_mission(binding);
}
extern "C" int32_t ccu_mission_b(const SimplerCcuRegistrationBindingV2 *binding) noexcept {
    return native_mission(binding);
}
extern "C" const SimplerCcuModuleManifestV2 *simpler_ccu_module_manifest_v2() noexcept {
    static const Manifest manifest = make_manifest();
    return &manifest.header;
}

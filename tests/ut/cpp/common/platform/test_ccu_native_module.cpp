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

#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "platform_comm/ccu/ccu_native_module.h"

extern "C" int simpler_ccu_test_kernel_a(const SimplerCcuRegistrationBindingV1 *);
extern "C" {
int simpler_ccu_test_manifest_variant = 0;
}

namespace {

std::vector<uint8_t> module_bytes() {
    if (simpler_ccu_test_kernel_a(nullptr) != 3) throw std::runtime_error("invalid CCU test fixture");
    Dl_info info{};
    const void *entry = ::dlsym(RTLD_NEXT, "simpler_ccu_test_kernel_a");
    if (entry == nullptr || ::dladdr(entry, &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("cannot locate CCU test module");
    std::ifstream input(info.dli_fname, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

simpler::ccu::ExpectedBuildIdentity expected_identity() {
    simpler::ccu::ExpectedBuildIdentity identity{};
    identity.sdk.fill(0x11);
    identity.pto_isa.fill(0x22);
    return identity;
}

}  // namespace

TEST(CcuNativeModuleTest, LoadsExactBytesWithLocalEntriesAndStableCacheIdentity) {
    simpler_ccu_test_manifest_variant = 0;
    const auto bytes = module_bytes();
    simpler::ccu::NativeModuleCache cache;
    auto first = cache.load(bytes, expected_identity());
    auto second = cache.load(bytes, expected_identity());

    ASSERT_EQ(first, second);
    ASSERT_EQ(cache.cached_module_count(), 1U);
    EXPECT_EQ(first->module_bytes(), bytes);
    ASSERT_EQ(first->kernels().size(), 2U);
    EXPECT_EQ(first->kernels()[0].record.kernel_id, 3U);
    EXPECT_EQ(first->kernels()[0].entry_symbol, "simpler_ccu_test_kernel_a");
    EXPECT_NE(first->kernels()[0].entry, nullptr);
    EXPECT_FALSE(first->cache_identity().empty());
}

TEST(CcuNativeModuleTest, BuildIdentityParticipatesInValidation) {
    simpler_ccu_test_manifest_variant = 0;
    auto wrong = expected_identity();
    wrong.sdk[0] = 0;
    EXPECT_THROW(simpler::ccu::NativeModule::load(module_bytes(), wrong), std::invalid_argument);
}

TEST(CcuNativeModuleTest, RejectsVersionReservedKernelAndEntryViolations) {
    for (int variant = 1; variant <= 10; ++variant) {
        simpler_ccu_test_manifest_variant = variant;
        EXPECT_THROW(simpler::ccu::NativeModule::load(module_bytes(), expected_identity()), std::invalid_argument)
            << "variant=" << variant;
    }
    simpler_ccu_test_manifest_variant = 0;
}

TEST(CcuNativeModuleTest, RejectsNonElfBytesBeforeLoading) {
    EXPECT_THROW(
        simpler::ccu::NativeModule::load(std::vector<uint8_t>(64, 0), expected_identity()), std::invalid_argument
    );
}

TEST(CcuNativeAbiTest, PinsRegistrationAndLaunchWordContracts) {
    static_assert(offsetof(SimplerCcuRegistrationBindingV1, channel_handle) == 32);
    static_assert(offsetof(SimplerCcuRegistrationBindingV1, trigger_event_handle) == 40);
    static_assert(offsetof(SimplerCcuRegistrationBindingV1, stop_event_handle) == 48);
    static_assert(offsetof(SimplerCcuMissionDescriptorV1, trigger_gate_addr) == 64);
    static_assert(offsetof(SimplerCcuMissionDescriptorV1, stop_gate_addr) == 72);
    static_assert(offsetof(SimplerCcuMissionDescriptorV1, run_control_token) == 80);
    static_assert(offsetof(SimplerCcuMissionDescriptorV1, parameter_token) == 88);
    static_assert(offsetof(SimplerCcuMissionDescriptorV1, completion_slot_token) == 96);

    const SimplerCcuLaunchArgumentsV1 arguments{SIMPLER_CCU_TASK_ARGUMENT_LAYOUT_VERSION, 2, 3, 4, 5, 6, 7, 8, 9, 0};
    const auto *words = reinterpret_cast<const uint64_t *>(&arguments);
    ASSERT_EQ(sizeof(arguments) / sizeof(words[0]), SIMPLER_CCU_TASK_ARGUMENT_COUNT);
    for (uint64_t index = 0; index < 9; ++index)
        EXPECT_EQ(words[index], index + 1);
    EXPECT_EQ(words[9], 0U);
}

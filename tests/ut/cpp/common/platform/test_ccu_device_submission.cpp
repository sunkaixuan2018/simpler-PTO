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
#include "platform_comm/ccu/ccu_device_submission.h"
namespace {
struct Fixture {
    alignas(64) uint64_t parameters[2]{};
    uint64_t ready{}, trigger{}, done{};
    SimplerCcuMissionDescriptorV2 mission{};
    Fixture() {
        mission.abi_version = 2;
        mission.struct_size = sizeof(mission);
        mission.run_generation = 7;
        mission.kernel_id = 3;
        mission.parameter_alignment = 64;
        mission.parameter_addr = reinterpret_cast<uint64_t>(parameters);
        mission.parameter_size = sizeof(parameters);
        mission.ready = {reinterpret_cast<uint64_t>(&ready), 1};
        mission.trigger = {reinterpret_cast<uint64_t>(&trigger), 1};
        mission.done = {reinterpret_cast<uint64_t>(&done), 1};
    }
};
TEST(CcuDeviceSubmission, RegistersDoneBeforeWritingParametersOrTrigger) {
    Fixture f;
    const uint64_t input[]{19, 23};
    ASSERT_TRUE(simpler::ccu::submit_mission(&f.mission, input, sizeof(input), [&](uint64_t address, uint64_t mask) {
        EXPECT_EQ(address, f.mission.done.address);
        EXPECT_EQ(mask, 1U);
        EXPECT_EQ(f.parameters[0], 0U);
        EXPECT_EQ(f.trigger, 0U);
        return true;
    }));
    EXPECT_EQ(f.parameters[0], 19U);
    EXPECT_EQ(f.parameters[1], 23U);
    EXPECT_EQ(f.trigger, (uint64_t{1} << 63) | 1);
}
TEST(CcuDeviceSubmission, RegistrationFailureDoesNotPublishParametersOrTrigger) {
    Fixture f;
    const uint64_t input[]{19, 23};
    EXPECT_FALSE(simpler::ccu::submit_mission(&f.mission, input, sizeof(input), [](uint64_t, uint64_t) {
        return false;
    }));
    EXPECT_EQ(f.parameters[0], 0U);
    EXPECT_EQ(f.trigger, 0U);
}
TEST(CcuDeviceSubmission, ValidatesWorkspaceGenerationAndAmbiguousKernelIds) {
    Fixture f;
    SimplerCcuMissionDescriptorV2 missions[]{f.mission, f.mission};
    missions[1].mission_index = 1;
    missions[1].kernel_id = 4;
    SimplerCcuWorkspaceV2 workspace{
        2,     sizeof(SimplerCcuWorkspaceV2), 2, 2, 7, reinterpret_cast<uint64_t>(missions), sizeof(missions[0]), 0, 91,
        {0, 0}
    };
    EXPECT_EQ(simpler::ccu::find_mission(&workspace, 4), &missions[1]);
    missions[1].run_generation = 6;
    EXPECT_EQ(simpler::ccu::find_mission(&workspace, 4), nullptr);
    missions[1].run_generation = 7;
    missions[1].kernel_id = 3;
    EXPECT_EQ(simpler::ccu::find_mission(&workspace, 3), nullptr);
}
TEST(CcuDeviceSubmission, RejectsAliasedEventsBeforeRegisteringCompletion) {
    Fixture f;
    f.mission.done.address = f.mission.trigger.address;
    const uint64_t input[]{19, 23};
    EXPECT_FALSE(simpler::ccu::submit_mission(&f.mission, input, sizeof(input), [](uint64_t, uint64_t) {
        ADD_FAILURE();
        return true;
    }));
    EXPECT_EQ(f.trigger, 0U);
}
}  // namespace

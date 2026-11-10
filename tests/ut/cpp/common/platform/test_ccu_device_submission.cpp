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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "platform_comm/ccu/ccu_device_submission.h"

namespace {

struct FakeIo {
    static uint64_t load_u64(const volatile uint64_t *address) {
        events.push_back("load");
        return *address;
    }

    static void publish(volatile void *address, uint64_t) {
        if (address == slot_address) {
            events.push_back("slot");
        } else if (address == parameter_address) {
            events.push_back("parameters");
        } else if (address == trigger_address) {
            events.push_back("trigger");
        } else {
            events.push_back("unknown");
        }
    }

    static void
    reset(volatile SimplerCcuCompletionSlotV1 *slot, volatile void *parameters, volatile uint64_t *trigger) {
        events.clear();
        slot_address = slot;
        parameter_address = parameters;
        trigger_address = trigger;
    }

    static inline std::vector<std::string> events;
    static inline volatile void *slot_address{};
    static inline volatile void *parameter_address{};
    static inline volatile void *trigger_address{};
};

struct MissionImage {
    alignas(SIMPLER_CCU_COMPLETION_SLOT_ALIGNMENT) SimplerCcuCompletionSlotV1 slot{};
    std::array<uint8_t, 8> parameters{};
    uint64_t trigger{};
    SimplerCcuMissionDescriptorV1 mission{};

    explicit MissionImage(uint32_t mission_id) {
        slot.abi_version = SIMPLER_CCU_MODULE_ABI_VERSION;
        slot.struct_size = sizeof(slot);
        slot.mission_id = mission_id;
        slot.run_generation = 9;

        mission.abi_version = SIMPLER_CCU_MODULE_ABI_VERSION;
        mission.struct_size = sizeof(mission);
        mission.run_generation = slot.run_generation;
        mission.mission_id = mission_id;
        mission.kernel_id = mission_id + 10;
        mission.group_id = 7;
        mission.parameter_addr = reinterpret_cast<uint64_t>(parameters.data());
        mission.parameter_size = parameters.size();
        mission.completion_slot_addr = reinterpret_cast<uint64_t>(&slot);
        mission.trigger_gate_addr = reinterpret_cast<uint64_t>(&trigger);
    }

    void reset_io() { FakeIo::reset(&slot, parameters.data(), &trigger); }
};

TEST(CcuDeviceSubmission, RegistersBeforePublishingParametersAndTrigger) {
    MissionImage image(3);
    image.reset_io();
    const std::array<uint8_t, 8> parameters{1, 2, 3, 4, 5, 6, 7, 8};

    const bool submitted = simpler::ccu::submit_mission<FakeIo>(
        &image.mission, 1, parameters.data(), parameters.size(), [](uint64_t address, uint64_t sequence) {
            FakeIo::events.push_back("register");
            EXPECT_EQ(address, reinterpret_cast<uint64_t>(FakeIo::slot_address));
            EXPECT_EQ(sequence, 1U);
            return true;
        }
    );

    EXPECT_TRUE(submitted);
    EXPECT_EQ(FakeIo::events, (std::vector<std::string>{"load", "register", "slot", "parameters", "trigger"}));
    EXPECT_EQ(image.parameters, parameters);
    EXPECT_EQ(image.slot.operation_sequence, 1U);
    EXPECT_EQ(image.slot.status, SIMPLER_CCU_COMPLETION_PENDING);
    EXPECT_EQ(image.trigger, 1U);
}

TEST(CcuDeviceSubmission, RegistrationFailurePublishesNothingAndDoesNotTrigger) {
    MissionImage image(3);
    image.reset_io();
    const std::array<uint8_t, 8> parameters{1, 2, 3, 4, 5, 6, 7, 8};

    EXPECT_FALSE(
        simpler::ccu::submit_mission<FakeIo>(
            &image.mission, 1, parameters.data(), parameters.size(), [](uint64_t, uint64_t) {
                FakeIo::events.push_back("register");
                return false;
            }
        )
    );

    EXPECT_EQ(FakeIo::events, (std::vector<std::string>{"load", "register"}));
    EXPECT_EQ(image.parameters, (std::array<uint8_t, 8>{}));
    EXPECT_EQ(image.slot.operation_sequence, 0U);
    EXPECT_EQ(image.trigger, 0U);
}

TEST(CcuDeviceSubmission, RejectsOverlapSequenceJumpAndFailedPredecessor) {
    MissionImage image(3);
    image.reset_io();
    const std::array<uint8_t, 8> parameters{};
    uint32_t registrations = 0;
    const auto register_completion = [&](uint64_t, uint64_t) {
        ++registrations;
        return true;
    };

    EXPECT_FALSE(
        simpler::ccu::submit_mission<FakeIo>(
            &image.mission, 2, parameters.data(), parameters.size(), register_completion
        )
    );
    EXPECT_EQ(registrations, 0U);

    image.slot.completed_sequence = 1;
    image.slot.operation_sequence = 1;
    image.slot.status = SIMPLER_CCU_COMPLETION_FAILED;
    image.slot.native_error = -9;
    EXPECT_FALSE(
        simpler::ccu::submit_mission<FakeIo>(
            &image.mission, 2, parameters.data(), parameters.size(), register_completion
        )
    );
    EXPECT_EQ(registrations, 0U);
    EXPECT_EQ(image.trigger, 0U);
}

TEST(CcuDeviceSubmission, IndependentMissionsCanBeOutstandingTogether) {
    MissionImage mission_a(3);
    MissionImage mission_b(4);
    const std::array<uint8_t, 8> parameters_a{1};
    const std::array<uint8_t, 8> parameters_b{2};
    std::vector<uint64_t> registered_slots;
    const auto register_completion = [&](uint64_t address, uint64_t) {
        registered_slots.push_back(address);
        return true;
    };

    mission_a.reset_io();
    ASSERT_TRUE(
        simpler::ccu::submit_mission<FakeIo>(
            &mission_a.mission, 1, parameters_a.data(), parameters_a.size(), register_completion
        )
    );
    mission_b.reset_io();
    ASSERT_TRUE(
        simpler::ccu::submit_mission<FakeIo>(
            &mission_b.mission, 1, parameters_b.data(), parameters_b.size(), register_completion
        )
    );

    ASSERT_EQ(registered_slots.size(), 2U);
    EXPECT_EQ(registered_slots[0], reinterpret_cast<uint64_t>(&mission_a.slot));
    EXPECT_EQ(registered_slots[1], reinterpret_cast<uint64_t>(&mission_b.slot));
    EXPECT_EQ(mission_a.slot.completed_sequence, 0U);
    EXPECT_EQ(mission_b.slot.completed_sequence, 0U);
    EXPECT_EQ(mission_a.trigger, 1U);
    EXPECT_EQ(mission_b.trigger, 1U);
}

}  // namespace

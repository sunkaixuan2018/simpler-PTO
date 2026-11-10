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

#include <map>
#include <string>
#include <vector>

#include "platform_comm/ccu/ccu_run_lifecycle.h"

namespace {
using simpler::ccu::DomainRunLifecycle;
using simpler::ccu::MissionRunResources;
using simpler::ccu::RegisteredKernel;
using simpler::ccu::RunLifecycleSdk;
using simpler::ccu::StreamState;

class FakeSdk final : public RunLifecycleSdk {
public:
    int launch(uint64_t, const uint64_t *, uint32_t argument_count, uint64_t stream) override {
        observations.push_back("launch:" + std::to_string(stream));
        EXPECT_EQ(argument_count, SIMPLER_CCU_TASK_ARGUMENT_COUNT);
        return launch_calls++ == launch_error_at ? launch_error : 0;
    }
    int read_control(uint64_t address, SimplerCcuRunControlV1 *control) override {
        observations.push_back("read:" + std::to_string(address));
        *control = controls[address];
        return read_error;
    }
    int request_stop(uint64_t address) override {
        observations.push_back("stop:" + std::to_string(address));
        controls[address].stop_requested = 1;
        return stop_error;
    }
    int wake_stop(uint64_t address, uint64_t value) override {
        observations.push_back("wake:" + std::to_string(address));
        wake_values[address] = value;
        return wake_error;
    }
    int query_stream(uint64_t stream, StreamState *state) override {
        observations.push_back("query_stream:" + std::to_string(stream));
        *state = streams[stream];
        return query_error;
    }
    int abort_stream(uint64_t stream) override {
        observations.push_back("abort:" + std::to_string(stream));
        return abort_error;
    }

    int launch_error{};
    int launch_error_at{-1};
    int read_error{};
    int stop_error{};
    int wake_error{};
    int query_error{};
    int abort_error{};
    std::map<uint64_t, SimplerCcuRunControlV1> controls;
    std::map<uint64_t, StreamState> streams;
    std::map<uint64_t, uint64_t> wake_values;
    std::vector<std::string> observations;

private:
    int launch_calls{};
};

SimplerCcuRunControlV1 control(uint64_t generation, uint32_t mission, uint32_t state) {
    SimplerCcuRunControlV1 value{};
    value.abi_version = SIMPLER_CCU_MODULE_ABI_VERSION;
    value.struct_size = sizeof(value);
    value.run_generation = generation;
    value.mission_id = mission;
    value.state = state;
    value.stop_gate_addr = 0x9000 + mission;
    value.final_wake_value = (uint64_t{1} << 63) | 1;
    return value;
}

MissionRunResources mission(uint64_t generation, uint32_t mission_id, uint64_t program, uint64_t stream) {
    MissionRunResources value{};
    value.mission_id = mission_id;
    value.program_handle = program;
    value.stream_handle = stream;
    value.launch_arguments = {
        SIMPLER_CCU_TASK_ARGUMENT_LAYOUT_VERSION,
        generation,
        mission_id,
        0x1000 + mission_id,
        0x2000 + mission_id,
        0x3000 + mission_id,
        0x4000 + mission_id,
        0x5000 + mission_id,
        0x6000 + mission_id,
        0
    };
    return value;
}

std::vector<RegisteredKernel> programs() { return {{1, 7, 101}, {2, 7, 102}}; }

TEST(CcuRunLifecycle, LaunchReturnDoesNotProveReadyAndAllRanksGateAdmission) {
    FakeSdk sdk0;
    FakeSdk sdk1;
    sdk0.controls[0x1001] = control(8, 1, SIMPLER_CCU_RUN_STATE_READY);
    sdk0.controls[0x1002] = control(8, 2, SIMPLER_CCU_RUN_STATE_READY);
    sdk1.controls[0x1001] = control(8, 1, SIMPLER_CCU_RUN_STATE_READY);
    sdk1.controls[0x1002] = control(8, 2, SIMPLER_CCU_RUN_STATE_LAUNCHED);
    DomainRunLifecycle owner0;
    DomainRunLifecycle owner1;
    auto run0 = owner0.begin_run(8, programs(), {mission(8, 1, 101, 11), mission(8, 2, 102, 12)}, sdk0);
    auto run1 = owner1.begin_run(8, programs(), {mission(8, 1, 101, 21), mission(8, 2, 102, 22)}, sdk1);

    run0->launch_all();
    run1->launch_all();

    EXPECT_FALSE(run0->ready());
    EXPECT_FALSE(simpler::ccu::all_runs_ready({run0.get(), run1.get()}));
    sdk1.controls[0x1002].state = SIMPLER_CCU_RUN_STATE_READY;
    EXPECT_TRUE(simpler::ccu::all_runs_ready({run0.get(), run1.get()}));
}

TEST(CcuRunLifecycle, NormalStopRequestsFinalWakeAndRequiresExitedStreams) {
    FakeSdk sdk;
    sdk.controls[0x1001] = control(9, 1, SIMPLER_CCU_RUN_STATE_READY);
    sdk.controls[0x1002] = control(9, 2, SIMPLER_CCU_RUN_STATE_READY);
    sdk.streams[11] = StreamState::RUNNING;
    sdk.streams[12] = StreamState::RUNNING;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(9, programs(), {mission(9, 1, 101, 11), mission(9, 2, 102, 12)}, sdk);
    run->launch_all();
    ASSERT_TRUE(run->poll_ready());

    EXPECT_FALSE(run->stop());
    EXPECT_EQ(sdk.controls[0x1001].stop_requested, 1U);
    EXPECT_EQ(sdk.controls[0x1002].stop_requested, 1U);
    EXPECT_EQ(sdk.wake_values[0x9001], (uint64_t{1} << 63) | 1);
    EXPECT_EQ(sdk.wake_values[0x9002], (uint64_t{1} << 63) | 1);

    sdk.controls[0x1001].state = SIMPLER_CCU_RUN_STATE_EXITED;
    sdk.controls[0x1002].state = SIMPLER_CCU_RUN_STATE_EXITED;
    sdk.streams[11] = StreamState::COMPLETE;
    sdk.streams[12] = StreamState::COMPLETE;
    EXPECT_TRUE(run->poll_terminal());
    EXPECT_TRUE(run->confirmed_stopped());
    EXPECT_TRUE(owner.admission_open());
    EXPECT_FALSE(owner.resources_retained());
}

TEST(CcuRunLifecycle, MissionFailureIsPrimaryAndPoisonsTheWholeRun) {
    FakeSdk sdk;
    sdk.controls[0x1001] = control(10, 1, SIMPLER_CCU_RUN_STATE_FAILED);
    sdk.controls[0x1001].native_error = 77;
    sdk.controls[0x1002] = control(10, 2, SIMPLER_CCU_RUN_STATE_READY);
    sdk.abort_error = 88;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(10, programs(), {mission(10, 1, 101, 11), mission(10, 2, 102, 12)}, sdk);
    run->launch_all();

    EXPECT_FALSE(run->poll_ready());
    ASSERT_NE(run->primary_failure(), nullptr);
    EXPECT_EQ(run->primary_failure()->native_error, 77);
    EXPECT_EQ(run->primary_failure()->native_error, 77);
    EXPECT_EQ(sdk.observations[sdk.observations.size() - 2], "abort:11");
    EXPECT_EQ(sdk.observations.back(), "abort:12");
    EXPECT_FALSE(owner.admission_open());
    EXPECT_TRUE(owner.resources_retained());
    EXPECT_THROW((void)owner.begin_run(11, programs(), {mission(11, 1, 101, 11)}, sdk), std::runtime_error);
}

TEST(CcuRunLifecycle, MissingStopGateRetainsResourcesAndClosesAdmission) {
    FakeSdk sdk;
    sdk.controls[0x1001] = control(12, 1, SIMPLER_CCU_RUN_STATE_READY);
    sdk.controls[0x1001].stop_gate_addr = 0;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(12, programs(), {mission(12, 1, 101, 11)}, sdk);
    run->launch_all();

    EXPECT_FALSE(run->stop());
    ASSERT_NE(run->primary_failure(), nullptr);
    EXPECT_EQ(run->primary_failure()->operation, "validate CCU stop gate");
    EXPECT_FALSE(owner.admission_open());
    EXPECT_TRUE(owner.resources_retained());
}

TEST(CcuRunLifecycle, DestroyingUnconfirmedRunRetainsResources) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    {
        auto run = owner.begin_run(13, programs(), {mission(13, 1, 101, 11)}, sdk);
        run->launch_all();
    }
    EXPECT_FALSE(owner.admission_open());
    EXPECT_TRUE(owner.resources_retained());
}

TEST(CcuRunLifecycle, PartialLaunchFailureAbortsOnlyStartedMissions) {
    FakeSdk sdk;
    sdk.launch_error_at = 1;
    sdk.launch_error = 66;
    DomainRunLifecycle owner;
    auto run = owner.begin_run(14, programs(), {mission(14, 1, 101, 11), mission(14, 2, 102, 12)}, sdk);

    EXPECT_THROW(run->launch_all(), std::runtime_error);
    ASSERT_NE(run->primary_failure(), nullptr);
    EXPECT_EQ(run->primary_failure()->native_error, 66);
    EXPECT_EQ(sdk.observations.back(), "abort:11");
    EXPECT_FALSE(owner.admission_open());
    EXPECT_TRUE(owner.resources_retained());
}

TEST(CcuRunLifecycle, UnlaunchedRunCanBeCancelledWithoutPoisoningDomain) {
    FakeSdk sdk;
    DomainRunLifecycle owner;
    { auto run = owner.begin_run(15, programs(), {mission(15, 1, 101, 11)}, sdk); }
    EXPECT_TRUE(owner.admission_open());
    EXPECT_FALSE(owner.resources_retained());

    auto retry = owner.begin_run(16, programs(), {mission(16, 1, 101, 11)}, sdk);
    EXPECT_FALSE(retry->launched());
}

}  // namespace

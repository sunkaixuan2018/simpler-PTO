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

#include "ccu_run_lifecycle_hcomm.h"

#include <cstddef>
#include <utility>

#include "hcomm/ccu/ccu_launch.h"

namespace simpler::ccu {
namespace {

constexpr int kInvalidArgument = -1;

}  // namespace

HcommRunLifecycleSdk::HcommRunLifecycleSdk(
    std::vector<HcommMissionStream> streams, StopWakeLauncher stop_wake_launcher
) :
    streams_(std::move(streams)),
    stop_wake_launcher_(std::move(stop_wake_launcher)) {}

const HcommMissionStream *HcommRunLifecycleSdk::find_stream(uint64_t handle) const {
    for (const auto &stream : streams_) {
        if (stream.handle == handle) return &stream;
    }
    return nullptr;
}

int HcommRunLifecycleSdk::launch(
    uint64_t program_handle, const uint64_t *arguments, uint32_t argument_count, uint64_t stream_handle
) {
    const HcommMissionStream *mission = find_stream(stream_handle);
    if (mission == nullptr || mission->thread == 0 || program_handle == 0 || arguments == nullptr) {
        return kInvalidArgument;
    }
    if (!HcommCcuKernelLaunch) return CCU_E_UNAVAIL;
    return HcommCcuKernelLaunch(mission->thread, program_handle, arguments, argument_count);
}

int HcommRunLifecycleSdk::read_control(uint64_t control_address, SimplerCcuRunControlV1 *control) {
    if (control_address == 0 || control == nullptr) return kInvalidArgument;
    return aclrtMemcpy(
        control, sizeof(*control), reinterpret_cast<const void *>(control_address), sizeof(*control),
        ACL_MEMCPY_DEVICE_TO_HOST
    );
}

int HcommRunLifecycleSdk::request_stop(uint64_t control_address) {
    if (control_address == 0) return kInvalidArgument;
    constexpr uint32_t stop_requested = 1;
    return aclrtMemcpy(
        reinterpret_cast<void *>(control_address + offsetof(SimplerCcuRunControlV1, stop_requested)),
        sizeof(stop_requested), &stop_requested, sizeof(stop_requested), ACL_MEMCPY_HOST_TO_DEVICE
    );
}

int HcommRunLifecycleSdk::wake_stop(uint64_t stop_gate_address, uint64_t final_wake_value) {
    if (stop_gate_address == 0 || final_wake_value == 0 || !stop_wake_launcher_) return kInvalidArgument;
    return stop_wake_launcher_(stop_gate_address, final_wake_value);
}

int HcommRunLifecycleSdk::query_stream(uint64_t stream_handle, StreamState *state) {
    const HcommMissionStream *mission = find_stream(stream_handle);
    if (mission == nullptr || mission->stream == nullptr || state == nullptr) return kInvalidArgument;
    aclrtStreamStatus status{};
    const aclError rc = aclrtStreamQuery(mission->stream, &status);
    if (rc != ACL_SUCCESS) {
        *state = StreamState::UNKNOWN;
        return rc;
    }
    *state = status == ACL_STREAM_STATUS_COMPLETE ? StreamState::COMPLETE : StreamState::RUNNING;
    return ACL_SUCCESS;
}

int HcommRunLifecycleSdk::abort_stream(uint64_t stream_handle) {
    const HcommMissionStream *mission = find_stream(stream_handle);
    if (mission == nullptr || mission->stream == nullptr) return kInvalidArgument;
    return aclrtStreamAbort(mission->stream);
}

}  // namespace simpler::ccu

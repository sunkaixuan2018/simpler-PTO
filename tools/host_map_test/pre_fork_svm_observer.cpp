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

#include <cstdint>

#include "common/memory_barrier.h"
#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

namespace {

constexpr int kExpectedArgCount = 4;
constexpr uint64_t kModeNoop = 0;
constexpr uint64_t kPayloadOffset = 0;
constexpr uint64_t kTailOffset = 64;
constexpr uint64_t kCompletionOffset = 128;
constexpr uint32_t kTailTimeout = 0xFFFFFFFEU;
constexpr uint32_t kPayloadMismatch = 0xFFFFFFFDU;

void submit_completion_task() {
    uint32_t shape[1] = {1};
    TensorCreateInfo output_info(shape, 1, DataType::INT32);
    CoreTaskArgs task_args;
    task_args.add_output(output_info);
    rt_submit_aiv_task(0, task_args);
}

void publish_completion(uint64_t base, uint32_t value) {
    auto *completion = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(base + kCompletionOffset));
    *completion = value;
    wmb();
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = kExpectedArgCount};
}

__attribute__((visibility("default"))) void pre_fork_svm_observer(const ChipTaskArgs &orch_args) {
    const uint64_t mode = orch_args.scalar(0);
    if (mode == kModeNoop) {
        submit_completion_task();
        return;
    }
    const uint64_t base = orch_args.scalar(1);
    const uint64_t expected_payload = orch_args.scalar(2);
    const uint32_t expected_tail = static_cast<uint32_t>(orch_args.scalar(3));
    auto *tail = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(base + kTailOffset));
    const uint32_t observed_tail = *tail;
    if (observed_tail < expected_tail) {
        publish_completion(base, kTailTimeout);
        submit_completion_task();
        return;
    }

    auto *payload = reinterpret_cast<volatile uint64_t *>(static_cast<uintptr_t>(base + kPayloadOffset));
    if (*payload != expected_payload) {
        publish_completion(base, kPayloadMismatch);
        submit_completion_task();
        return;
    }
    publish_completion(base, expected_tail);
    submit_completion_task();
}

}  // extern "C"

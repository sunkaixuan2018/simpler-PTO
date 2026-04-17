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
/**
 * Demo orchestration for host-side mapped memory.
 *
 * Host passes:
 *   tensor(0): mapped_out
 *
 * Runtime appends:
 *   scalar(last): mapped_dev_ptr for the runtime-managed host/device shared buffer
 *                  (appended after framework scalars, so use scalar_count()-1)
 *
 * The shared buffer is wrapped as an external tensor so the AIV kernel can copy
 * the scheduler-updated values into a normal output tensor.
 */

#include <inttypes.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 2,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    const ContinuousTensor &out_arg = orch_args.tensor(0);
    Tensor mapped_out = from_tensor_arg(out_arg);

    int last_scalar = orch_args.scalar_count() - 1;
    uint64_t mapped_input_u64 = orch_args.scalar(last_scalar);
    Tensor mapped_host_buffer = make_tensor_external(
        reinterpret_cast<void *>(static_cast<uintptr_t>(mapped_input_u64)), out_arg.shapes, out_arg.ndims, out_arg.dtype
    );

    LOG_INFO(
        "host_register_mapped_demo: mapped_host_buffer=0x%" PRIx64 " mapped_out=0x%" PRIx64 " elements=%u",
        mapped_input_u64,
        out_arg.data,
        out_arg.shapes[0]
    );

    Arg params;
    params.add_inout(mapped_host_buffer);
    params.add_output(mapped_out);
    pto2_rt_submit_aiv_task(0, params);
}

}  // extern "C"

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
 *   tensor(1): device_out
 *
 * Runtime appends:
 *   scalar(last - 1): mapped_dev_ptr for the runtime-managed host/device shared buffer
 *   scalar(last): direct_dev_ptr for the runtime-managed direct device buffer
 *
 * Both buffers are wrapped as external tensors so the AIV kernel can update
 * them and copy the final values into normal output tensors.
 */

#include <inttypes.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

static inline void demo_dsb_ld() {
#if defined(__aarch64__)
    __asm__ __volatile__("dsb ld" ::: "memory");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    const ContinuousTensor &mapped_out_arg = orch_args.tensor(0);
    const ContinuousTensor &device_out_arg = orch_args.tensor(1);
    Tensor mapped_out = from_tensor_arg(mapped_out_arg);
    Tensor device_out = from_tensor_arg(device_out_arg);

    int last_scalar = orch_args.scalar_count() - 1;
    uint64_t mapped_input_u64 = orch_args.scalar(last_scalar - 1);
    uint64_t direct_input_u64 = orch_args.scalar(last_scalar);
    Tensor mapped_host_buffer = make_tensor_external(
        reinterpret_cast<void *>(static_cast<uintptr_t>(mapped_input_u64)), mapped_out_arg.shapes,
        mapped_out_arg.ndims, mapped_out_arg.dtype
    );
    Tensor direct_device_buffer = make_tensor_external(
        reinterpret_cast<void *>(static_cast<uintptr_t>(direct_input_u64)), device_out_arg.shapes,
        device_out_arg.ndims, device_out_arg.dtype
    );
    mapped_out.update_start_offset();
    device_out.update_start_offset();
    mapped_host_buffer.update_start_offset();
    direct_device_buffer.update_start_offset();

    LOG_INFO(
        "host_register_mapped_demo: mapped_host_buffer=0x%" PRIx64 " direct_device_buffer=0x%" PRIx64
        " mapped_out=0x%" PRIx64 " device_out=0x%" PRIx64 " elements=%u",
        mapped_input_u64,
        direct_input_u64,
        mapped_out_arg.data,
        device_out_arg.data,
        mapped_out_arg.shapes[0]
    );

    Arg params;
    params.add_inout(mapped_host_buffer);
    params.add_inout(direct_device_buffer);
    params.add_output(mapped_out);
    params.add_output(device_out);

    demo_dsb_ld();
    uint64_t mapped_data_addr = mapped_host_buffer.buffer.addr +
                                mapped_host_buffer.start_offset * get_element_size(mapped_host_buffer.dtype);
    uint64_t direct_data_addr = direct_device_buffer.buffer.addr +
                                direct_device_buffer.start_offset * get_element_size(direct_device_buffer.dtype);
    uint64_t element_count = (mapped_out_arg.shapes[0] > 0) ? static_cast<uint64_t>(mapped_out_arg.shapes[0]) : 0;
    LOG_INFO(
        "host_register_mapped_demo: before_submit_aiv mapped_buffer_addr=0x%" PRIx64
        " mapped_data_addr=0x%" PRIx64 " direct_buffer_addr=0x%" PRIx64
        " direct_data_addr=0x%" PRIx64 " elements=%" PRIu64,
        mapped_host_buffer.buffer.addr,
        mapped_data_addr,
        direct_device_buffer.buffer.addr,
        direct_data_addr,
        element_count
    );

    pto2_rt_submit_aiv_task(0, params);
}

}  // extern "C"

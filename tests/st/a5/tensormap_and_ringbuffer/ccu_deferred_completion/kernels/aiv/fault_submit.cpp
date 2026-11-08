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
#include "common.h"
#include "backend/ccu/ccu_completion_kernel.h"

inline __aicore__ void load_geometry(const __gm__ Tensor *source, Tensor &tensor) {
    tensor.buffer.addr = source->buffer.addr;
    tensor.buffer.size = source->buffer.size;
    tensor.start_offset = source->start_offset;
    tensor.ndims = source->ndims;
    tensor.dtype = source->dtype;
    tensor.address_space = source->address_space;
    for (uint32_t dim = 0; dim < MAX_TENSOR_DIMS; ++dim) {
        tensor.shapes[dim] = source->shapes[dim];
        tensor.strides[dim] = source->strides[dim];
    }
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    const auto *domain = reinterpret_cast<const __gm__ CommContext *>(args[2]);
    auto *control = reinterpret_cast<volatile __gm__ uint32_t *>(args[3]);
    const uint64_t mismatch = args[4];
    // A5's system counter runs at 1 GHz. The peer delay is below the 5 s CCU timeout.
    if (domain->rankId == 1) {
        const uint64_t start = get_sys_cnt();
        while (get_sys_cnt() - start < 2000000000ULL) {}
    }
    Tensor source;
    Tensor destination;
    load_geometry(reinterpret_cast<const __gm__ Tensor *>(args[0]), source);
    load_geometry(reinterpret_cast<const __gm__ Tensor *>(args[1]), destination);
    CcuCallArgs parameters;
    parameters.add_tensor(source);
    parameters.add_tensor(destination);
    parameters.add_scalar(uint64_t{64} + (mismatch ? domain->rankId : 0));
    AsyncCtx async = get_async_ctx(args);
    const bool accepted = send_request_entry(async, CcuCall(domain, 0, parameters));
    control[0] = accepted ? 1U : 2U;
    pto2::detail::defer_flush_range(control, 64);
    dsb(DSB_DDR);
}

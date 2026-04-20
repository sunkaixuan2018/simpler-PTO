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
 * Demo kernel:
 *   mapped_host_buffer[i] += 3
 *   out[i] = mapped_host_buffer[i]
 *
 * This verifies the AIV path can read and write the host-registered mapped
 * address, then mirrors the updated values into a regular output tensor.
 */

#include <cstdint>

#include "tensor.h"  // NOLINT(build/include_subdir)

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

constexpr int32_t kWordCount = 16;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *mapped_host_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ uint64_t *mapped_host =
        reinterpret_cast<__gm__ uint64_t *>(mapped_host_tensor->buffer.addr) + mapped_host_tensor->start_offset;
    __gm__ uint64_t *out = reinterpret_cast<__gm__ uint64_t *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    for (int32_t i = 0; i < kWordCount; ++i) {
        mapped_host[i] += 3;
        out[i] = mapped_host[i];
    }
}

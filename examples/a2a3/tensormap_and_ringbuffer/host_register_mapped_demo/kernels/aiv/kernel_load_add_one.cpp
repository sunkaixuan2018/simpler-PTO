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
 *   out[i] = mapped_host_buffer[i]
 *
 * The scheduler updates the shared host/device buffer from 0..15 to 1..16.
 * This kernel only copies that result into a regular output tensor.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"  // NOLINT(build/include_subdir)

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

constexpr int32_t kWordCount = 16;
constexpr int32_t kFloatCount = static_cast<int32_t>((kWordCount * sizeof(uint64_t)) / sizeof(float));

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *mapped_host_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *mapped_host = reinterpret_cast<__gm__ float *>(mapped_host_tensor->buffer.addr) +
                                (mapped_host_tensor->start_offset * static_cast<int32_t>(sizeof(uint64_t) / sizeof(float)));
    __gm__ float *out =
        reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) +
        (out_tensor->start_offset * static_cast<int32_t>(sizeof(uint64_t) / sizeof(float)));

    using DynShapeDim5 = Shape<1, 1, 1, 1, kFloatCount>;
    using DynStridDim5 = Stride<1, 1, 1, kFloatCount, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, 1, kFloatCount, BLayout::RowMajor, -1, -1>;

    TileData src_tile(1, kFloatCount);
    TASSIGN(src_tile, 0x0);

    GlobalData mapped_host_global(mapped_host);
    GlobalData dst_global(out);

    TLOAD(src_tile, mapped_host_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSTORE(dst_global, src_tile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

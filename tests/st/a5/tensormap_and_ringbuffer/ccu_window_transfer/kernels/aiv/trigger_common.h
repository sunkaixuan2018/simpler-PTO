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

#pragma once
#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"
#include "tensor.h"

inline __aicore__ void trigger_window(__gm__ int64_t *args, uint32_t offset, uint32_t ccu_id) {
    const auto *comm = reinterpret_cast<const __gm__ CommContext *>(args[offset + 1]);
    const uint64_t round = args[offset + 2] - (ccu_id == 1 ? 2 : 1);
    const uint64_t displacement = (round % 8) * 64;
    const uint64_t length = ((round % 16) + 1) * 64;
    using Arg = simpler::ccu::CcuCallArg;
    const Arg call_args[]{
        {Arg::Kind::Window, comm->rankId, ccu_id * 8192 + displacement, length},
        {Arg::Kind::Window, (comm->rankId + 1) % comm->rankNum, 16384 + ccu_id * 8192 + displacement, length},
        {Arg::Kind::Scalar, 0, length, 0}
    };
    AsyncCtx ctx = get_async_ctx(args);
    (void)submit_ccu_mission(ctx, comm, ccu_id, call_args);
}

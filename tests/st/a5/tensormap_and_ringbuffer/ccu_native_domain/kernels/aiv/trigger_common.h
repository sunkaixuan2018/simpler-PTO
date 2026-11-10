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

#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

inline __aicore__ void trigger_mission(__gm__ int64_t *args, uint32_t offset, uint32_t index) {
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[offset]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[offset + 1]);
    const uint64_t value = args[offset + 2];
    AsyncCtx ctx = get_async_ctx(args);
    if (!comm || comm->ccu_mission_count != 2 || !comm->ccu_missions_addr) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    auto *missions = reinterpret_cast<volatile __gm__ SimplerCcuMissionDescriptorV1 *>(comm->ccu_missions_addr);
    auto *a_slot = reinterpret_cast<volatile __gm__ SimplerCcuCompletionSlotV1 *>(missions[0].completion_slot_addr);
    dcci(const_cast<__gm__ SimplerCcuCompletionSlotV1 *>(a_slot), cache_line_t::ENTIRE_DATA_CACHE);
    if (a_slot->completed_sequence != 0) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    uint64_t parameters[]{output->buffer.addr + output->start_offset * sizeof(uint64_t), value};
    (void)submit_ccu_mission(ctx, &missions[index], 1, parameters, sizeof(parameters));
}

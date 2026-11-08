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

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    namespace wire = simpler::ccu::device;
    auto *results = data<uint32_t>(args, 0);
    const auto *domain = reinterpret_cast<const __gm__ CommContext *>(args[1]);
    const auto *control = reinterpret_cast<const volatile __gm__ uint32_t *>(args[2]);
    const uint32_t block = get_block_idx(args);
    const uint32_t blocks = get_block_num(args);
    auto *global = reinterpret_cast<__gm__ GlobalContext *>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    const auto &storage = comm_workspace<CommWorkspaceKind::CCU>(*domain);
    uint64_t bytes = 0;
    auto *workspace = wire::Protocol<>::instance(
        reinterpret_cast<__gm__ wire::DomainDirectory *>(storage.addr), storage.size, global->callable_id,
        domain->rankId, domain->rankNum, bytes
    );
    wire::Reference ref{};
    bool pending = false;
    if (workspace && wire::Protocol<>::resolve(workspace, bytes, global->callable_id, 0, ref)) {
        auto *context = wire::Protocol<>::platform(ref);
        if (context) {
            auto *record = ccu_detail::At<CcuRequestRecord>(context, context->record_offset);
            ccu_detail::Sync(record, sizeof(*record));
            pending = record->slot_state == CcuSlotState::SUBMITTED && record->sequence != 0 &&
                      record->completed_sequence < record->sequence;
        }
    }
    // Each cohort lane owns a complete cache line. No lane modifies a request record.
    const uint32_t offset = block * 16;
    results[offset] = pending ? 1U : 0U;
    results[offset + 1] = control[0];
    results[offset + 2] = control[16];
    results[offset + 3] = blocks;
    pto2::detail::defer_flush_range(results + offset, 64);
    dsb(DSB_DDR);
}

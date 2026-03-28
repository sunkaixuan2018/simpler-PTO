/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_EVENT_IMPL_HPP
#define PTO_COMM_ASYNC_EVENT_IMPL_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"

namespace pto {
namespace comm {

// ============================================================================
// AsyncSession: engine-agnostic session for async DMA operations.
// Users build via comm::BuildAsyncSession<engine>() and pass to
// TPUT_ASYNC / TGET_ASYNC / event.Wait() without knowing engine internals.
// ============================================================================

struct AsyncSession {
    DmaEngine engine{DmaEngine::SDMA};
    sdma::SdmaSession sdmaSession{};
    bool valid{false};
};

template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile, __gm__ uint8_t *workspace, AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {sdma::detail::kDefaultSdmaBlockBytes, 0,
                                                                              1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx)
{
    session.engine = engine;
    if constexpr (engine == DmaEngine::SDMA) {
        session.valid =
            sdma::BuildSdmaSession(scratchTile, workspace, session.sdmaSession, syncId, baseConfig, channelGroupIdx);
        return session.valid;
    } else {
        static_assert(engine == DmaEngine::SDMA, "Only SDMA engine is supported currently");
        return false;
    }
}

// ============================================================================
// AsyncEvent::Wait / WaitCounted / Test — AsyncSession overloads (primary user API)
// ============================================================================

PTO_INTERNAL bool AsyncEvent::Wait(const AsyncSession &session) const
{
    if (handle == 0) {
        return true;
    }
    switch (session.engine) {
        case DmaEngine::SDMA:
            return sdma::detail::SdmaWaitEvent(handle, session.sdmaSession.eventCtx);
        default:
            return false;
    }
}

PTO_INTERNAL uint32_t AsyncEvent::WaitCounted(const AsyncSession &session) const
{
    if (handle == 0) {
        return 0;
    }
    switch (session.engine) {
        case DmaEngine::SDMA:
            return sdma::detail::SdmaWaitEventCounted(handle, session.sdmaSession.eventCtx);
        default:
            return 0;
    }
}

PTO_INTERNAL bool AsyncEvent::Test(const AsyncSession &session) const
{
    if (handle == 0) {
        return true;
    }
    switch (session.engine) {
        case DmaEngine::SDMA:
            return sdma::detail::SdmaTestEvent(handle, session.sdmaSession.eventCtx);
        default:
            return false;
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_EVENT_IMPL_HPP

#pragma once

#include <cstdint>

#include "common/comm_context.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef AICORE
#define AICORE inline
#endif

template <typename T>
AICORE __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T*>(ctx->windowsIn[peer_rank] + offset);
}

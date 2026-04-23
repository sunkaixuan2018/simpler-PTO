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

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_ASYNC_CONTEXT_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_ASYNC_CONTEXT_H_

#include <stdint.h>

typedef enum PTO2AsyncEngine {
    PTO2_ASYNC_ENGINE_SDMA = 0,
    PTO2_ASYNC_ENGINE_ROCE = 1,
    PTO2_ASYNC_ENGINE_URMA = 2,
    PTO2_ASYNC_ENGINE_CCU = 3,
    PTO2_ASYNC_ENGINE_COUNT = 4
} PTO2AsyncEngine;

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_ASYNC_CONTEXT_H_

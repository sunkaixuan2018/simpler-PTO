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

#ifndef TOOLS_A3_HCCL_SMOKE_A3_FABRIC_WINDOW_H_
#define TOOLS_A3_HCCL_SMOKE_A3_FABRIC_WINDOW_H_

#include <acl/acl.h>

#include <cstddef>

class A3FabricWindow {
public:
    A3FabricWindow() = default;
    A3FabricWindow(const A3FabricWindow &) = delete;
    A3FabricWindow &operator=(const A3FabricWindow &) = delete;

    aclError CreateLocal(int device_id, size_t requested_size);
    aclError Export(aclrtMemFabricHandle *shareable_handle) const;
    aclError ImportPeer(const aclrtMemFabricHandle &shareable_handle, size_t peer_size);
    aclError Destroy();

    void *local_base() const { return local_base_; }
    void *peer_base() const { return peer_base_; }
    size_t size() const { return size_; }

private:
    int device_id_{-1};
    size_t size_{0};
    size_t peer_size_{0};
    aclrtDrvMemHandle local_handle_{nullptr};
    aclrtDrvMemHandle peer_handle_{nullptr};
    void *local_base_{nullptr};
    void *peer_base_{nullptr};
};

#endif  // TOOLS_A3_HCCL_SMOKE_A3_FABRIC_WINDOW_H_

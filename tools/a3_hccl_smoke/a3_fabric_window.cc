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

#include "a3_fabric_window.h"

namespace {

size_t AlignUp(size_t size, size_t alignment) { return ((size + alignment - 1U) / alignment) * alignment; }

aclError ReserveAndMap(int device_id, size_t size, aclrtDrvMemHandle handle, void **base) {
    aclError status = aclrtReserveMemAddress(base, size, 0, nullptr, HUGE_PAGE_TYPE);
    if (status != ACL_SUCCESS) {
        return status;
    }

    status = aclrtMapMem(*base, size, 0, handle, 0);
    if (status != ACL_SUCCESS) {
        (void)aclrtReleaseMemAddress(*base);
        *base = nullptr;
        return status;
    }

    aclrtMemAccessDesc access_desc{};
    access_desc.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    access_desc.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_id;
    status = aclrtMemSetAccess(*base, size, &access_desc, 1);
    if (status != ACL_SUCCESS) {
        (void)aclrtUnmapMem(*base);
        (void)aclrtReleaseMemAddress(*base);
        *base = nullptr;
    }
    return status;
}

void RecordFirstError(aclError status, aclError *first_error) {
    if (*first_error == ACL_SUCCESS && status != ACL_SUCCESS) {
        *first_error = status;
    }
}

void ReleaseMapping(void **base, aclrtDrvMemHandle *handle, aclError *first_error) {
    if (*base != nullptr) {
        RecordFirstError(aclrtUnmapMem(*base), first_error);
        RecordFirstError(aclrtReleaseMemAddress(*base), first_error);
        *base = nullptr;
    }
    if (*handle != nullptr) {
        RecordFirstError(aclrtFreePhysical(*handle), first_error);
        *handle = nullptr;
    }
}

}  // namespace

aclError A3FabricWindow::CreateLocal(int device_id, size_t requested_size) {
    if (device_id < 0 || requested_size == 0 || local_handle_ != nullptr || local_base_ != nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }

    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.memAttr = ACL_HBM_MEM_HUGE;

    size_t granularity = 0;
    aclError status = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity);
    if (status != ACL_SUCCESS) {
        return status;
    }
    if (granularity == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }

    device_id_ = device_id;
    size_ = AlignUp(requested_size, granularity);
    status = aclrtMallocPhysical(&local_handle_, size_, &prop, 0);
    if (status != ACL_SUCCESS) {
        local_handle_ = nullptr;
        size_ = 0;
        device_id_ = -1;
        return status;
    }

    status = ReserveAndMap(device_id_, size_, local_handle_, &local_base_);
    if (status != ACL_SUCCESS) {
        (void)aclrtFreePhysical(local_handle_);
        local_handle_ = nullptr;
        size_ = 0;
        device_id_ = -1;
    }
    return status;
}

aclError A3FabricWindow::Export(aclrtMemFabricHandle *shareable_handle) const {
    if (local_handle_ == nullptr || shareable_handle == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclrtMemExportToShareableHandleV2(
        local_handle_, ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC,
        shareable_handle
    );
}

aclError A3FabricWindow::ImportPeer(const aclrtMemFabricHandle &shareable_handle, size_t peer_size) {
    if (device_id_ < 0 || size_ == 0 || peer_size == 0 || peer_handle_ != nullptr || peer_base_ != nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }

    aclrtMemFabricHandle imported_shareable_handle = shareable_handle;
    aclError status = aclrtMemImportFromShareableHandleV2(
        &imported_shareable_handle, ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT, &peer_handle_
    );
    if (status != ACL_SUCCESS) {
        peer_handle_ = nullptr;
        return status;
    }

    peer_size_ = peer_size;
    status = ReserveAndMap(device_id_, peer_size_, peer_handle_, &peer_base_);
    if (status != ACL_SUCCESS) {
        (void)aclrtFreePhysical(peer_handle_);
        peer_handle_ = nullptr;
        peer_size_ = 0;
    }
    return status;
}

aclError A3FabricWindow::Destroy() {
    aclError first_error = ACL_SUCCESS;
    ReleaseMapping(&peer_base_, &peer_handle_, &first_error);
    ReleaseMapping(&local_base_, &local_handle_, &first_error);
    peer_size_ = 0;
    size_ = 0;
    device_id_ = -1;
    return first_error;
}

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
 * PTO Runtime C API - Implementation (On-board Hardware)
 *
 * Platform-specific implementation of the public C API declared in
 * src/common/worker/pto_runtime_c_api.h.  Uses real Ascend device execution.
 */

#include "pto_runtime_c_api.h"

#include "callable.h"
#include "task_args.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <pthread.h>
#include <sstream>
#include <vector>

#include "common/unified_log.h"
#include "device_runner.h"
#include "host/raii_scope_guard.h"
#include "runtime.h"

namespace {

using RtMallocHostFn = int (*)(void **, uint64_t, uint32_t);
using RtFreeHostFn = int (*)(void *);
using RtHostRegisterFn = int (*)(void *, uint64_t, uint32_t, void **);
using RtHostUnregisterFn = int (*)(void *);
using AclMallocHostFn = int (*)(void **, size_t);
using AclFreeHostFn = int (*)(void *);
using AclHostRegisterFn = int (*)(void *, uint64_t, uint32_t, void **);
using AclHostRegisterV2Fn = int (*)(void *, uint64_t, uint32_t);
using AclHostGetDevicePointerFn = int (*)(void *, void **, uint32_t);
using AclHostUnregisterFn = int (*)(void *);
using GetDeviceFn = int (*)(int32_t *);
using DirectSetDeviceFn = int (*)(int32_t);

constexpr const char *kDemoShareMemCountEnv = "PTO2_DEMO_SHARE_MEM_U64_COUNT";
constexpr uint64_t kDemoShareMemPreviewCount = 16;

template <typename Fn>
Fn resolve_symbol(const char **resolved_name, std::initializer_list<const char *> names) {
    for (const char *name : names) {
        dlerror();
        void *sym = dlsym(RTLD_DEFAULT, name);
        const char *err = dlerror();
        if (err == nullptr && sym != nullptr) {
            if (resolved_name != nullptr) {
                *resolved_name = name;
            }
            return reinterpret_cast<Fn>(sym);
        }
    }
    if (resolved_name != nullptr) {
        *resolved_name = nullptr;
    }
    return nullptr;
}

static constexpr uint32_t kHostRegisterMappedFlag =
#if defined(RT_HOST_REGISTER_MAPPED)
    RT_HOST_REGISTER_MAPPED;
#elif defined(ACL_HOST_REGISTER_MAPPED)
    ACL_HOST_REGISTER_MAPPED;
#else
    0U;
#endif

static constexpr uint32_t kAclHostRegisterV2MappedFlag =
#if defined(ACL_HOST_REG_MAPPED)
    ACL_HOST_REG_MAPPED;
#else
    0x2U;
#endif

int ensure_current_device_for_share_mem(uint32_t device_id) {
    const char *symbol_name = nullptr;
    if (GetDeviceFn get_device_fn = resolve_symbol<GetDeviceFn>(&symbol_name, {"aclrtGetDevice", "rtGetDevice"})) {
        int32_t current_device = -1;
        int rc = get_device_fn(&current_device);
        if (rc != 0) {
            LOG_INFO(
                "ensure_current_device_for_share_mem: %s failed rc=%d, trying to set device to %u",
                symbol_name, rc, device_id
            );
            if (DirectSetDeviceFn set_device_fn =
                    resolve_symbol<DirectSetDeviceFn>(&symbol_name, {"rtSetDevice", "aclrtSetDevice"})) {
                rc = set_device_fn(static_cast<int32_t>(device_id));
                if (rc != 0) {
                    LOG_ERROR(
                        "ensure_current_device_for_share_mem: %s(%u) failed: rc=%d", symbol_name, device_id, rc
                    );
                    return rc;
                }
                return 0;
            }
            LOG_ERROR("ensure_current_device_for_share_mem: missing symbols rtSetDevice / aclrtSetDevice");
            return rc;
        }

        if (current_device != static_cast<int32_t>(device_id)) {
            LOG_ERROR(
                "ensure_current_device_for_share_mem: current device %d does not match requested device %u",
                static_cast<int>(current_device), device_id
            );
            return -1;
        }
        return 0;
    }

    LOG_ERROR("ensure_current_device_for_share_mem: missing symbols aclrtGetDevice / rtGetDevice");
    return -1;
}

uint64_t parse_demo_share_mem_u64_count() {
    const char *env = std::getenv(kDemoShareMemCountEnv);
    if (env == nullptr || env[0] == '\0') {
        return 0;
    }

    char *endptr = nullptr;
    errno = 0;
    uint64_t count = strtoull(env, &endptr, 10);
    if (errno == ERANGE || endptr == env || *endptr != '\0') {
        LOG_WARN("%s=%s invalid, expected a positive integer", kDemoShareMemCountEnv, env);
        return 0;
    }

    return count;
}

void log_demo_share_mem_preview(const char *label, const uint64_t *values, uint64_t count) {
    if (values == nullptr || count == 0) {
        return;
    }

    uint64_t preview_count = std::min(count, kDemoShareMemPreviewCount);
    std::ostringstream os;
    os << label << " first_" << preview_count << "=[";
    for (uint64_t i = 0; i < preview_count; ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << values[i];
    }
    os << "] total=" << count;
    LOG_INFO("%s", os.str().c_str());
}

int setup_demo_share_mem(Runtime *runtime, int device_id) {
    if (runtime == nullptr) {
        return -1;
    }

    uint64_t word_count = parse_demo_share_mem_u64_count();
    if (word_count == 0) {
        runtime->clear_share_mem_registration();
        return 0;
    }

    uint64_t size_bytes = word_count * sizeof(uint64_t);
    void *host_ptr = nullptr;
    void *dev_ptr = nullptr;
    int rc = mallocHostDeviceShareMem(static_cast<uint32_t>(device_id), size_bytes, &host_ptr, &dev_ptr);
    if (rc != 0) {
        LOG_ERROR("host_register_mapped_demo: mallocHostDeviceShareMem failed rc=%d", rc);
        return rc;
    }
    auto mapped_mem_guard = RAIIScopeGuard([device_id, host_ptr]() {
        freeHostDeviceShareMem(static_cast<uint32_t>(device_id), host_ptr);
    });

    void *direct_host_ptr = std::malloc(static_cast<size_t>(size_bytes));
    if (direct_host_ptr == nullptr) {
        LOG_ERROR("host_register_mapped_demo: failed to allocate host staging for direct device buffer");
        return -1;
    }
    auto direct_host_guard = RAIIScopeGuard([direct_host_ptr]() {
        std::free(direct_host_ptr);
    });

    void *direct_dev_ptr = runtime->host_api.device_malloc(static_cast<size_t>(size_bytes));
    if (direct_dev_ptr == nullptr) {
        LOG_ERROR("host_register_mapped_demo: failed to allocate direct device buffer size=%" PRIu64, size_bytes);
        return -1;
    }
    auto direct_dev_guard = RAIIScopeGuard([runtime, direct_dev_ptr]() {
        runtime->host_api.device_free(direct_dev_ptr);
    });

    auto *host_words = static_cast<uint64_t *>(host_ptr);
    auto *direct_host_words = static_cast<uint64_t *>(direct_host_ptr);
    for (uint64_t i = 0; i < word_count; ++i) {
        host_words[i] = i;
        direct_host_words[i] = 100 + i;
    }

    rc = runtime->host_api.copy_to_device(direct_dev_ptr, direct_host_ptr, static_cast<size_t>(size_bytes));
    if (rc != 0) {
        LOG_ERROR("host_register_mapped_demo: failed to initialize direct device buffer rc=%d", rc);
        return rc;
    }

    runtime->set_share_mem_registration(
        static_cast<uint32_t>(device_id), host_ptr, dev_ptr, direct_host_ptr, direct_dev_ptr, size_bytes, word_count
    );
    mapped_mem_guard.dismiss();
    direct_host_guard.dismiss();
    direct_dev_guard.dismiss();

    log_demo_share_mem_preview("host_register_mapped_demo: mapped_host_init_data", host_words, word_count);
    log_demo_share_mem_preview("host_register_mapped_demo: direct_device_init_data", direct_host_words, word_count);
    LOG_INFO(
        "host_register_mapped_demo: mapped_host_ptr=%p mapped_dev_ptr=%p direct_dev_ptr=%p size=%" PRIu64,
        host_ptr,
        dev_ptr,
        direct_dev_ptr,
        size_bytes
    );
    return 0;
}

void release_demo_share_mem(Runtime *runtime) {
    if (runtime == nullptr || !runtime->get_share_mem_enabled()) {
        return;
    }

    uint64_t word_count = runtime->get_share_mem_u64_count();
    uint64_t size_bytes = runtime->get_share_mem_size_bytes();
    log_demo_share_mem_preview(
        "host_register_mapped_demo: mapped_host_data_after_run",
        static_cast<const uint64_t *>(runtime->get_share_mem_host_ptr()), word_count
    );

    if (runtime->get_share_mem_direct_host_ptr() != nullptr && runtime->get_share_mem_direct_dev_ptr() != nullptr) {
        int copy_rc = runtime->host_api.copy_from_device(
            runtime->get_share_mem_direct_host_ptr(), runtime->get_share_mem_direct_dev_ptr(),
            static_cast<size_t>(size_bytes)
        );
        if (copy_rc != 0) {
            LOG_ERROR(
                "host_register_mapped_demo: copy direct device buffer back failed rc=%d dev=%p",
                copy_rc,
                runtime->get_share_mem_direct_dev_ptr()
            );
        } else {
            log_demo_share_mem_preview(
                "host_register_mapped_demo: direct_device_data_after_run",
                static_cast<const uint64_t *>(runtime->get_share_mem_direct_host_ptr()), word_count
            );
        }
        runtime->host_api.device_free(runtime->get_share_mem_direct_dev_ptr());
    }
    if (runtime->get_share_mem_direct_host_ptr() != nullptr) {
        std::free(runtime->get_share_mem_direct_host_ptr());
    }

    int rc = freeHostDeviceShareMem(runtime->get_share_mem_device_id(), runtime->get_share_mem_host_ptr());
    if (rc != 0) {
        LOG_ERROR(
            "host_register_mapped_demo: freeHostDeviceShareMem failed rc=%d host=%p", rc,
            runtime->get_share_mem_host_ptr()
        );
    }
    runtime->clear_share_mem_registration();
}

}  // namespace

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in runtime_maker.cpp)
 * =========================================================================== */
int init_runtime_impl(Runtime *runtime, const ChipCallable *callable, const ChipStorageTaskArgs *orch_args);
int validate_runtime_impl(Runtime *runtime);

/* ===========================================================================
 * Per-thread DeviceRunner binding (set by run_runtime, read by HostApi wrappers)
 * =========================================================================== */

static pthread_key_t g_runner_key;
static pthread_once_t g_runner_key_once = PTHREAD_ONCE_INIT;
static void create_runner_key() { pthread_key_create(&g_runner_key, nullptr); }

static DeviceRunner *current_runner() { return static_cast<DeviceRunner *>(pthread_getspecific(g_runner_key)); }

/* ===========================================================================
 * Internal device-memory functions (used via Runtime.host_api, NOT dlsym'd)
 * =========================================================================== */

static void *device_malloc(size_t size) {
    try {
        return current_runner()->allocate_tensor(size);
    } catch (...) {
        return NULL;
    }
}

static void device_free(void *dev_ptr) {
    if (dev_ptr == NULL) return;
    try {
        current_runner()->free_tensor(dev_ptr);
    } catch (...) {}
}

static int copy_to_device(void *dev_ptr, const void *host_ptr, size_t size) {
    if (dev_ptr == NULL || host_ptr == NULL) return -1;
    try {
        return current_runner()->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

static int copy_from_device(void *host_ptr, const void *dev_ptr, size_t size) {
    if (host_ptr == NULL || dev_ptr == NULL) return -1;
    try {
        return current_runner()->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

static uint64_t upload_kernel_binary_wrapper(int func_id, const uint8_t *bin_data, size_t bin_size) {
    try {
        return current_runner()->upload_kernel_binary(func_id, bin_data, bin_size);
    } catch (...) {
        return 0;
    }
}

static void remove_kernel_binary_wrapper(int func_id) {
    try {
        current_runner()->remove_kernel_binary(func_id);
    } catch (...) {}
}

/* ===========================================================================
 * Public C API (resolved by ChipWorker via dlsym)
 * =========================================================================== */

DeviceContextHandle create_device_context(void) {
    try {
        return static_cast<DeviceContextHandle>(new DeviceRunner());
    } catch (...) {
        return NULL;
    }
}

void destroy_device_context(DeviceContextHandle ctx) { delete static_cast<DeviceRunner *>(ctx); }

size_t get_runtime_size(void) { return sizeof(Runtime); }

int set_device(DeviceContextHandle ctx, int device_id) {
    (void)ctx;
    (void)device_id;
    return 0;
}

int mallocHostDeviceShareMem(uint32_t deviceId, uint64_t size, void **hostPtr, void **devPtr) {
    if (hostPtr == NULL || devPtr == NULL || size == 0) {
        return -1;
    }

    *hostPtr = nullptr;
    *devPtr = nullptr;

    int rc = ensure_current_device_for_share_mem(deviceId);
    if (rc != 0) {
        return rc;
    }

    void *allocated_host_ptr = nullptr;
    const char *symbol_name = nullptr;
    if (RtMallocHostFn malloc_fn = resolve_symbol<RtMallocHostFn>(&symbol_name, {"rtMallocHost"})) {
        rc = malloc_fn(&allocated_host_ptr, size, 0U);
        if (rc != 0 || allocated_host_ptr == nullptr) {
            LOG_ERROR(
                "mallocHostDeviceShareMem via %s failed on rtMallocHost: rc=%d size=%" PRIu64, symbol_name, rc, size
            );
            return (rc != 0) ? rc : -1;
        }
    } else if (AclMallocHostFn malloc_fn = resolve_symbol<AclMallocHostFn>(&symbol_name, {"aclrtMallocHost"})) {
        rc = malloc_fn(&allocated_host_ptr, static_cast<size_t>(size));
        if (rc != 0 || allocated_host_ptr == nullptr) {
            LOG_ERROR(
                "mallocHostDeviceShareMem via %s failed on aclrtMallocHost: rc=%d size=%" PRIu64, symbol_name, rc,
                size
            );
            return (rc != 0) ? rc : -1;
        }
    } else {
        LOG_ERROR("mallocHostDeviceShareMem: missing symbols rtMallocHost / aclrtMallocHost");
        return -1;
    }

    auto free_allocated_host = [&allocated_host_ptr]() {
        if (RtFreeHostFn free_fn = resolve_symbol<RtFreeHostFn>(nullptr, {"rtFreeHost"})) {
            free_fn(allocated_host_ptr);
        } else if (AclFreeHostFn free_fn = resolve_symbol<AclFreeHostFn>(nullptr, {"aclrtFreeHost"})) {
            free_fn(allocated_host_ptr);
        }
    };

    bool registered = false;
    const char *register_backend = "unknown";
    if (AclHostRegisterV2Fn register_v2_fn =
            resolve_symbol<AclHostRegisterV2Fn>(&symbol_name, {"aclrtHostRegisterV2"})) {
        const char *register_symbol_name = symbol_name;
        const char *get_dev_symbol_name = nullptr;
        if (AclHostGetDevicePointerFn get_dev_fn = resolve_symbol<AclHostGetDevicePointerFn>(
                &get_dev_symbol_name, {"aclrtHostGetDevicePointer"}
            )) {
            bool v2_registered = false;
            rc = register_v2_fn(allocated_host_ptr, size, kAclHostRegisterV2MappedFlag);
            if (rc == 0) {
                v2_registered = true;
                rc = get_dev_fn(allocated_host_ptr, devPtr, 0U);
            }
            if (rc == 0 && *devPtr != nullptr) {
                registered = true;
                register_backend = "V2";
                LOG_INFO(
                    "mallocHostDeviceShareMem via %s + %s succeeded: host=%p dev=%p size=%" PRIu64 " flag=%u",
                    register_symbol_name,
                    get_dev_symbol_name,
                    allocated_host_ptr,
                    *devPtr,
                    size,
                    kAclHostRegisterV2MappedFlag
                );
            } else {
                LOG_WARN(
                    "mallocHostDeviceShareMem via %s + %s failed: rc=%d host=%p dev=%p size=%" PRIu64 " flag=%u, "
                    "falling back to HostRegister V1",
                    register_symbol_name,
                    get_dev_symbol_name,
                    rc,
                    allocated_host_ptr,
                    *devPtr,
                    size,
                    kAclHostRegisterV2MappedFlag
                );
                if (v2_registered) {
                    if (AclHostUnregisterFn unregister_fn = resolve_symbol<AclHostUnregisterFn>(
                            &symbol_name, {"aclrtHostUnregister"}
                        )) {
                        int unregister_rc = unregister_fn(allocated_host_ptr);
                        if (unregister_rc != 0) {
                            LOG_ERROR(
                                "mallocHostDeviceShareMem via %s failed to clean up V2 registration: rc=%d host=%p",
                                symbol_name,
                                unregister_rc,
                                allocated_host_ptr
                            );
                            free_allocated_host();
                            return unregister_rc;
                        }
                    } else {
                        LOG_ERROR(
                            "mallocHostDeviceShareMem: aclrtHostUnregister missing after V2 registration failure"
                        );
                        free_allocated_host();
                        return -1;
                    }
                }
                *devPtr = nullptr;
            }
        } else {
            LOG_WARN(
                "mallocHostDeviceShareMem: %s found but aclrtHostGetDevicePointer missing, falling back to "
                "HostRegister V1",
                register_symbol_name
            );
        }
    }

    if (!registered) {
        RtHostRegisterFn register_fn =
            resolve_symbol<RtHostRegisterFn>(&symbol_name, {"rtsHostRegister", "rtHostRegister"});
        if (register_fn != nullptr) {
            rc = register_fn(allocated_host_ptr, size, kHostRegisterMappedFlag, devPtr);
            if (rc != 0 || *devPtr == nullptr) {
                LOG_ERROR(
                    "mallocHostDeviceShareMem via %s failed on host register: rc=%d host=%p size=%" PRIu64
                    " flag=%u",
                    symbol_name,
                    rc,
                    allocated_host_ptr,
                    size,
                    kHostRegisterMappedFlag
                );
                free_allocated_host();
                return (rc != 0) ? rc : -1;
            }
            registered = true;
            register_backend = "V1_RT";
        }
    }

    if (!registered) {
        AclHostRegisterFn register_fn = resolve_symbol<AclHostRegisterFn>(&symbol_name, {"aclrtHostRegister"});
        if (register_fn != nullptr) {
            rc = register_fn(allocated_host_ptr, size, kHostRegisterMappedFlag, devPtr);
            if (rc != 0 || *devPtr == nullptr) {
                LOG_ERROR(
                    "mallocHostDeviceShareMem via %s failed on host register: rc=%d host=%p size=%" PRIu64
                    " flag=%u",
                    symbol_name,
                    rc,
                    allocated_host_ptr,
                    size,
                    kHostRegisterMappedFlag
                );
                free_allocated_host();
                return (rc != 0) ? rc : -1;
            }
            registered = true;
            register_backend = "V1_ACL";
        }
    }

    if (!registered) {
        LOG_ERROR(
            "mallocHostDeviceShareMem: missing usable host register symbols "
            "aclrtHostRegisterV2 / rtsHostRegister / rtHostRegister / aclrtHostRegister"
        );
        free_allocated_host();
        return -1;
    }

    *hostPtr = allocated_host_ptr;
    LOG_INFO(
        "mallocHostDeviceShareMem: device=%u backend=%s host=%p dev=%p size=%" PRIu64,
        deviceId,
        register_backend,
        *hostPtr,
        *devPtr,
        size
    );
    return 0;
}

int freeHostDeviceShareMem(uint32_t deviceId, void *hostPtr) {
    if (hostPtr == NULL) {
        return 0;
    }

    int rc = ensure_current_device_for_share_mem(deviceId);
    if (rc != 0) {
        return rc;
    }

    const char *symbol_name = nullptr;
    int unregister_rc = -1;
    if (AclHostUnregisterFn unregister_fn =
            resolve_symbol<AclHostUnregisterFn>(&symbol_name, {"aclrtHostUnregister"})) {
        rc = unregister_fn(hostPtr);
        if (rc == 0) {
            unregister_rc = 0;
        } else {
            unregister_rc = rc;
            LOG_WARN(
                "freeHostDeviceShareMem via %s failed on unregister: rc=%d host=%p, trying RT unregister",
                symbol_name,
                rc,
                hostPtr
            );
        }
    }

    if (unregister_rc != 0) {
        if (RtHostUnregisterFn unregister_fn =
                resolve_symbol<RtHostUnregisterFn>(&symbol_name, {"rtsHostUnregister", "rtHostUnregister"})) {
            rc = unregister_fn(hostPtr);
            if (rc == 0) {
                unregister_rc = 0;
            } else {
                unregister_rc = rc;
                LOG_ERROR(
                    "freeHostDeviceShareMem via %s failed on unregister: rc=%d host=%p", symbol_name, rc, hostPtr
                );
            }
        }
    }

    if (unregister_rc != 0) {
        LOG_ERROR(
            "freeHostDeviceShareMem: no unregister method succeeded "
            "(aclrtHostUnregister / rtsHostUnregister / rtHostUnregister)"
        );
        return unregister_rc;
    }

    if (RtFreeHostFn free_fn = resolve_symbol<RtFreeHostFn>(&symbol_name, {"rtFreeHost"})) {
        rc = free_fn(hostPtr);
        if (rc != 0) {
            LOG_ERROR("freeHostDeviceShareMem via %s failed on free: rc=%d host=%p", symbol_name, rc, hostPtr);
            return rc;
        }
    } else if (AclFreeHostFn free_fn = resolve_symbol<AclFreeHostFn>(&symbol_name, {"aclrtFreeHost"})) {
        rc = free_fn(hostPtr);
        if (rc != 0) {
            LOG_ERROR("freeHostDeviceShareMem via %s failed on free: rc=%d host=%p", symbol_name, rc, hostPtr);
            return rc;
        }
    } else {
        LOG_ERROR("freeHostDeviceShareMem: missing symbols rtFreeHost / aclrtFreeHost");
        return -1;
    }

    LOG_INFO("freeHostDeviceShareMem: device=%u host=%p", deviceId, hostPtr);
    return 0;
}

int run_runtime(
    DeviceContextHandle ctx, RuntimeHandle runtime, const void *callable, const void *args, int block_dim,
    int aicpu_thread_num, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size, const uint8_t *aicore_binary,
    size_t aicore_size, int enable_profiling, int enable_dump_tensor
) {
    if (ctx == NULL || runtime == NULL) return -1;
    if (aicpu_binary == NULL || aicpu_size == 0 || aicore_binary == NULL || aicore_size == 0) return -1;

    DeviceRunner *runner = static_cast<DeviceRunner *>(ctx);

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    try {
        int rc = runner->prepare_run_context(device_id);
        if (rc != 0) return rc;
        auto run_context_guard = RAIIScopeGuard([runner]() {
            runner->release_run_context();
        });

        Runtime *r = new (runtime) Runtime();
        r->host_api.device_malloc = device_malloc;
        r->host_api.device_free = device_free;
        r->host_api.copy_to_device = copy_to_device;
        r->host_api.copy_from_device = copy_from_device;
        r->host_api.upload_kernel_binary = upload_kernel_binary_wrapper;
        r->host_api.remove_kernel_binary = remove_kernel_binary_wrapper;
        auto demo_share_mem_guard = RAIIScopeGuard([r]() {
            release_demo_share_mem(r);
        });
        auto release_and_destroy_runtime = [&demo_share_mem_guard, r]() {
            release_demo_share_mem(r);
            demo_share_mem_guard.dismiss();
            r->~Runtime();
        };

        rc = setup_demo_share_mem(r, device_id);
        if (rc != 0) {
            release_and_destroy_runtime();
            return rc;
        }

        LOG_DEBUG("About to call init_runtime_impl, r=%p", (void *)r);
        rc = init_runtime_impl(
            r, reinterpret_cast<const ChipCallable *>(callable), reinterpret_cast<const ChipStorageTaskArgs *>(args)
        );
        LOG_DEBUG("init_runtime_impl returned: %d", rc);
        if (rc != 0) {
            r->set_pto2_gm_sm_ptr(nullptr);
            validate_runtime_impl(r);
            release_and_destroy_runtime();
            return rc;
        }

        if (enable_profiling) {
            r->enable_profiling = true;
        }

        std::vector<uint8_t> aicpu_vec(aicpu_binary, aicpu_binary + aicpu_size);
        std::vector<uint8_t> aicore_vec(aicore_binary, aicore_binary + aicore_size);
        rc = runner->run(*r, block_dim, device_id, aicpu_vec, aicore_vec, aicpu_thread_num, enable_dump_tensor != 0);
        if (rc != 0) {
            validate_runtime_impl(r);
            release_and_destroy_runtime();
            return rc;
        }

        rc = validate_runtime_impl(r);
        release_and_destroy_runtime();
        return rc;
    } catch (...) {
        return -1;
    }
}

int finalize_device(DeviceContextHandle ctx) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->finalize();
    } catch (...) {
        return -1;
    }
}

/* ===========================================================================
 * Internal helpers called from runtime_maker.cpp via Runtime.host_api
 * =========================================================================== */

void record_tensor_pair(RuntimeHandle runtime, void *host_ptr, void *dev_ptr, size_t size) {
    if (runtime == NULL) return;
    Runtime *r = static_cast<Runtime *>(runtime);
    r->record_tensor_pair(host_ptr, dev_ptr, size);
}

}  // extern "C"

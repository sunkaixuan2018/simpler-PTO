/**
 * @file host_prefetch_setup.cpp
 * @brief Host-side SDMA prefetch channel setup
 *
 * Sets up STARS SDMA channel for AICPU prefetch by:
 * 1. Creating a device-only stream (gets SQ metadata)
 * 2. Allocating device workspace
 * 3. Launching AclnnShmemSdmaStarsQuery to populate stars_channel_info_t
 *
 * All APIs are resolved via dlsym to avoid hard dependencies on specific
 * CANN/shmem versions. If any step fails, prefetch is silently disabled.
 */

#include "host/host_prefetch_setup.h"
#include "device_runner.h"

#include <acl/acl.h>
#include <cstdlib>
#include <dlfcn.h>
#include <cstring>
#include <cstdio>
#include <strings.h>
#include <vector>
#include <cerrno>
#include <string>

// shmem host types (must match shmem's layout)
struct host_stream_info_t {
    uint64_t stream_;
    uint64_t ctx_;
    int32_t stream_id;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t cqe_addr;
    int32_t dev_id;
    uint8_t reserved[20];
};

struct sdma_op_res_info_t {
    uint64_t size;
    uint64_t streams_addr;
    uint64_t workspace_addr;
    uint8_t reserved[40];
};

// dlsym function pointer types
using RtStreamGetSqidFn = int (*)(const void* stream, uint32_t* sqId);
using RtStreamGetCqidFn = int (*)(const void* stream, uint32_t* cqId, uint32_t* logicCqId);

// aclnn two-phase API (aclTensor-based)
using AclCreateTensorFn = void* (*)(const int64_t* shape, uint64_t dimNum, int dataType,
                                     const int64_t* strides, int64_t offset, int format,
                                     const int64_t* storageShape, uint64_t storageDimNum, void* data);
using AclDestroyTensorFn = void (*)(void* tensor);
using AclnnGetWsFn = int (*)(const void* input, const void* output, uint64_t* wsSize, void** executor);
using AclnnExecFn = int (*)(void* ws, uint64_t wsSize, void* executor, void* stream);

static constexpr size_t SDMA_WORKSPACE_SIZE = 16 * 1024;

// State for cleanup
static std::vector<void*> g_prefetch_streams;
static void* g_streams_device_ptr = nullptr;
static void* g_op_res_device_ptr = nullptr;
static void* g_opapi_handle = nullptr;
static void* g_workspace_device_ptr = nullptr;
static int g_cached_device_id = -1;
static int g_cached_channel_count = 0;

static void* try_dlopen_opapi_from_provider_root()
{
    const char* provider_root = std::getenv("PTO_SDMA_PROVIDER_ROOT");
    if (provider_root == nullptr || *provider_root == '\0') {
        return nullptr;
    }

    std::string lib_path(provider_root);
    lib_path += "/x86_64-linux/lib64/libopapi.so";
    void* handle = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (handle == nullptr) {
        LOG_WARN("SDMA prefetch: failed to dlopen provider libopapi.so from %s (%s)",
                 lib_path.c_str(), dlerror());
    }
    return handle;
}

static bool sdma_prefetch_enabled_by_env()
{
    const char* mode = std::getenv("PTO_SDMA_PREFETCH_MODE");
    if (mode != nullptr && *mode != '\0') {
        if (strcasecmp(mode, "baseline") == 0 || strcmp(mode, "0") == 0) {
            return false;
        }
        if (strcasecmp(mode, "twoslot") == 0 || strcmp(mode, "1") == 0) {
            return false;
        }
        if (strcasecmp(mode, "sdma") == 0 || strcmp(mode, "2") == 0) {
            return true;
        }
        if (strcasecmp(mode, "sdma_fake") == 0 || strcasecmp(mode, "fake") == 0 ||
            strcmp(mode, "3") == 0) {
            return false;
        }
    }

    const char* value = std::getenv("PTO_ENABLE_SDMA_PREFETCH");
    if (value == nullptr || *value == '\0') {
        return true;
    }
    if (std::strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "off") == 0 ||
        strcasecmp(value, "no") == 0) {
        return false;
    }
    return true;
}

static int resolve_channel_count(int requested_count)
{
    if (requested_count <= 0) {
        return requested_count;
    }

    const char* env = std::getenv("PTO_SDMA_PREFETCH_CHANNELS");
    if (env == nullptr || *env == '\0') {
        return requested_count;
    }

    char* end = nullptr;
    errno = 0;
    long parsed = std::strtol(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || parsed <= 0) {
        LOG_INFO("SDMA prefetch: ignore invalid PTO_SDMA_PREFETCH_CHANNELS=%s", env);
        return requested_count;
    }

    int override_count = static_cast<int>(parsed);
    int final_count = override_count < requested_count ? override_count : requested_count;
    if (final_count < 1) {
        final_count = 1;
    }
    LOG_INFO("SDMA prefetch: channel override requested=%d env=%d final=%d",
             requested_count, override_count, final_count);
    return final_count;
}

void* host_prefetch_setup(int channel_count)
{
    if (!sdma_prefetch_enabled_by_env()) {
        LOG_INFO("SDMA prefetch: disabled by PTO_ENABLE_SDMA_PREFETCH");
        return nullptr;
    }

    channel_count = resolve_channel_count(channel_count);
    if (channel_count <= 0) {
        LOG_INFO("SDMA prefetch: disabled (invalid channel_count=%d)", channel_count);
        return nullptr;
    }

    // Resolve all required APIs via dlsym
    auto rtStreamGetSqid = reinterpret_cast<RtStreamGetSqidFn>(dlsym(RTLD_DEFAULT, "rtStreamGetSqid"));
    auto rtStreamGetCqid = reinterpret_cast<RtStreamGetCqidFn>(dlsym(RTLD_DEFAULT, "rtStreamGetCqid"));
    auto aclCreateTensor = reinterpret_cast<AclCreateTensorFn>(dlsym(RTLD_DEFAULT, "aclCreateTensor"));
    auto aclDestroyTensor = reinterpret_cast<AclDestroyTensorFn>(dlsym(RTLD_DEFAULT, "aclDestroyTensor"));

    // aclnn ops are in libopapi.so which may not be loaded yet — try dlopen
    void* opapi_handle = nullptr;
    auto aclnnGetWs = reinterpret_cast<AclnnGetWsFn>(dlsym(RTLD_DEFAULT, "aclnnShmemSdmaStarsQueryGetWorkspaceSize"));
    auto aclnnExec = reinterpret_cast<AclnnExecFn>(dlsym(RTLD_DEFAULT, "aclnnShmemSdmaStarsQuery"));
    if (!aclnnGetWs || !aclnnExec) {
        opapi_handle = try_dlopen_opapi_from_provider_root();
        if (!opapi_handle) {
            opapi_handle = dlopen("libopapi.so", RTLD_LAZY | RTLD_GLOBAL);
        }
        if (opapi_handle) {
            aclnnGetWs = reinterpret_cast<AclnnGetWsFn>(dlsym(opapi_handle, "aclnnShmemSdmaStarsQueryGetWorkspaceSize"));
            aclnnExec = reinterpret_cast<AclnnExecFn>(dlsym(opapi_handle, "aclnnShmemSdmaStarsQuery"));
            if (!aclCreateTensor) aclCreateTensor = reinterpret_cast<AclCreateTensorFn>(dlsym(opapi_handle, "aclCreateTensor"));
            if (!aclDestroyTensor) aclDestroyTensor = reinterpret_cast<AclDestroyTensorFn>(dlsym(opapi_handle, "aclDestroyTensor"));
            g_opapi_handle = opapi_handle;
        }
    }

    if (!rtStreamGetSqid || !rtStreamGetCqid) {
        LOG_INFO("SDMA prefetch: rtStreamGetSqid/Cqid not found, skipping");
        return nullptr;
    }
    if (!aclCreateTensor || !aclDestroyTensor || !aclnnGetWs || !aclnnExec) {
        LOG_INFO("SDMA prefetch: shmem aclnn ops not found, skipping");
        return nullptr;
    }

    int rc;

    int32_t dev_id = -1;
    aclrtGetDevice(&dev_id);
    void* ctx = nullptr;
    aclrtGetCurrentContext(&ctx);

    // Fast path: reuse existing channels/workspace when device and count match.
    if (g_workspace_device_ptr != nullptr &&
        g_cached_device_id == dev_id &&
        g_cached_channel_count == channel_count &&
        static_cast<int>(g_prefetch_streams.size()) == channel_count) {
        LOG_INFO("SDMA prefetch: reusing cached STARS workspace (device=%d channels=%d)", dev_id, channel_count);
        return g_workspace_device_ptr;
    }

    // If something is already initialized but doesn't match the requested config, tear it down first.
    if (!g_prefetch_streams.empty() || g_streams_device_ptr != nullptr ||
        g_op_res_device_ptr != nullptr || g_workspace_device_ptr != nullptr) {
        host_prefetch_teardown(nullptr);
    }

    void* workspace = nullptr;
    std::vector<host_stream_info_t> stream_infos(static_cast<size_t>(channel_count));
    g_prefetch_streams.reserve(static_cast<size_t>(channel_count));
    for (int i = 0; i < channel_count; ++i) {
        void* stream = nullptr;
        rc = aclrtCreateStreamWithConfig(reinterpret_cast<aclrtStream*>(&stream), 0, 0x20);  // ACL_STREAM_DEVICE_USE_ONLY
        if (rc != 0 || !stream) {
            LOG_INFO("SDMA prefetch: create device stream %d/%d failed (rc=%d)", i, channel_count, rc);
            goto fail_streams;
        }
        g_prefetch_streams.push_back(stream);

        host_stream_info_t& si = stream_infos[static_cast<size_t>(i)];
        si.stream_ = reinterpret_cast<uint64_t>(stream);
        si.ctx_ = reinterpret_cast<uint64_t>(ctx);
        si.dev_id = dev_id;

        int32_t stream_id = 0;
        aclrtStreamGetId(reinterpret_cast<aclrtStream>(stream), &stream_id);
        si.stream_id = stream_id;
        rtStreamGetSqid(stream, &si.sq_id);
        rtStreamGetCqid(stream, &si.cq_id, &si.logic_cq_id);

        if (i < 4 || i == channel_count - 1) {
            LOG_INFO("SDMA prefetch: stream[%d/%d] created (sid=%d sq=%u cq=%u)",
                     i, channel_count, stream_id, si.sq_id, si.cq_id);
        }
    }

    // 2. Allocate workspace
    rc = aclrtMalloc(&workspace, SDMA_WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != 0) {
        LOG_ERROR("SDMA prefetch: workspace malloc failed");
        goto fail_streams;
    }
    aclrtMemset(workspace, SDMA_WORKSPACE_SIZE, 0, SDMA_WORKSPACE_SIZE);

    // 3. Copy stream info to device
    {
        void* si_dev = nullptr;
        size_t stream_infos_size = stream_infos.size() * sizeof(host_stream_info_t);
        rc = aclrtMalloc(&si_dev, stream_infos_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != 0) goto fail_workspace;
        aclrtMemcpy(si_dev, stream_infos_size, stream_infos.data(), stream_infos_size, ACL_MEMCPY_HOST_TO_DEVICE);
        g_streams_device_ptr = si_dev;

        // 4. Prepare op_res_info
        sdma_op_res_info_t op_res = {};
        op_res.size = static_cast<uint64_t>(stream_infos.size());
        op_res.streams_addr = reinterpret_cast<uint64_t>(si_dev);
        op_res.workspace_addr = reinterpret_cast<uint64_t>(workspace);

        void* op_dev = nullptr;
        rc = aclrtMalloc(&op_dev, sizeof(op_res), ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != 0) goto fail_si_dev;
        aclrtMemcpy(op_dev, sizeof(op_res), &op_res, sizeof(op_res), ACL_MEMCPY_HOST_TO_DEVICE);
        g_op_res_device_ptr = op_dev;

        // 5. Create input/output tensors and launch op
        uint64_t in_data[2] = {reinterpret_cast<uint64_t>(op_dev), reinterpret_cast<uint64_t>(workspace)};
        uint64_t out_data[1] = {0};
        void* in_dev = nullptr;
        void* out_dev = nullptr;
        aclrtMalloc(&in_dev, sizeof(in_data), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(in_dev, sizeof(in_data), in_data, sizeof(in_data), ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMalloc(&out_dev, sizeof(out_data), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(out_dev, sizeof(out_data), 0, sizeof(out_data));

        int64_t in_shape[] = {2};
        int64_t out_shape[] = {1};
        int64_t strides[] = {1};

        // ACL_UINT64 = 10, ACL_FORMAT_ND = 2
        void* input = aclCreateTensor(in_shape, 1, 10, strides, 0, 2, in_shape, 1, in_dev);
        void* output = aclCreateTensor(out_shape, 1, 10, strides, 0, 2, out_shape, 1, out_dev);

        if (!input || !output) {
            LOG_ERROR("SDMA prefetch: aclCreateTensor failed");
            if (input) aclDestroyTensor(input);
            if (output) aclDestroyTensor(output);
            aclrtFree(in_dev);
            aclrtFree(out_dev);
            goto fail_op_dev;
        }

        // Launch AICPU op
        void* aicpu_stream = nullptr;
        aclrtCreateStreamWithConfig(reinterpret_cast<aclrtStream*>(&aicpu_stream), 0, 0x3);

        uint64_t ws_size = 0;
        void* executor = nullptr;
        rc = aclnnGetWs(input, output, &ws_size, &executor);

        void* ws = nullptr;
        if (rc == 0) {
            if (ws_size > 0) aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
            rc = aclnnExec(ws, ws_size, executor, aicpu_stream);
            if (rc == 0) {
                aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(aicpu_stream));
                LOG_INFO("SDMA prefetch: STARS channel initialized");
            } else {
                LOG_ERROR("SDMA prefetch: aclnnShmemSdmaStarsQuery exec failed (rc=%d)", rc);
            }
            if (ws) aclrtFree(ws);
        } else {
            LOG_ERROR("SDMA prefetch: aclnnShmemSdmaStarsQuery getWs failed (rc=%d)", rc);
        }

        aclrtDestroyStream(reinterpret_cast<aclrtStream>(aicpu_stream));
        aclDestroyTensor(input);
        aclDestroyTensor(output);
        aclrtFree(in_dev);
        aclrtFree(out_dev);

        if (rc != 0) goto fail_op_dev;
    }

    g_workspace_device_ptr = workspace;
    g_cached_device_id = dev_id;
    g_cached_channel_count = channel_count;
    return workspace;

fail_op_dev:
    aclrtFree(g_op_res_device_ptr);
    g_op_res_device_ptr = nullptr;
fail_si_dev:
    aclrtFree(g_streams_device_ptr);
    g_streams_device_ptr = nullptr;
fail_workspace:
    aclrtFree(workspace);
fail_streams:
    for (void* stream : g_prefetch_streams) {
        aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream));
    }
    g_prefetch_streams.clear();
    return nullptr;
}

void host_prefetch_teardown(void* workspace)
{
    void* workspace_to_free = workspace != nullptr ? workspace : g_workspace_device_ptr;
    if (workspace_to_free) {
        aclrtFree(workspace_to_free);
    }
    g_workspace_device_ptr = nullptr;
    g_cached_device_id = -1;
    g_cached_channel_count = 0;
    if (g_op_res_device_ptr) { aclrtFree(g_op_res_device_ptr); g_op_res_device_ptr = nullptr; }
    if (g_streams_device_ptr) { aclrtFree(g_streams_device_ptr); g_streams_device_ptr = nullptr; }
    for (void* stream : g_prefetch_streams) {
        aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream));
    }
    g_prefetch_streams.clear();
    // Don't dlclose opapi — other code may still reference its symbols
    g_opapi_handle = nullptr;
}

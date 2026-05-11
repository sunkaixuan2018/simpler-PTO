/**
 * @file host_prefetch_setup.cpp
 * @brief Host-side SDMA prefetch channel setup
 *
 * Sets up STARS SDMA channel for AICPU prefetch by:
 * 1. Creating a device-only stream (gets SQ metadata)
 * 2. Allocating device workspace
 * 3. Writing stream SQ/CQ metadata into stars_channel_info_t.
 *
 * The AICPU side uses halSqCqQuery to resolve SQ base/register/depth from
 * these IDs, avoiding a dependency on ops-legacy/libopapi.so.
 *
 * Only stream SQ/CQ query symbols are resolved via dlsym. If any step fails,
 * prefetch is silently disabled.
 */

#include "host/host_prefetch_setup.h"
#include "device_runner.h"

#include <acl/acl.h>
#include <driver/ascend_hal.h>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <cstring>
#include <cstdio>
#include <strings.h>
#include <vector>
#include <cerrno>
#include <string>

// Device workspace layout shared with the AICPU side.
struct stars_channel_flag_info_t {
    uint32_t flag;
    uint32_t totalQueueNum;
    uint8_t reserved[56];
};

struct stars_channel_info_t {
    uint32_t sq_head;
    uint32_t sq_tail;
    uint64_t sq_base;
    uint64_t sq_reg_base;
    uint32_t sq_depth;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t cqe_addr;
    uint32_t report_cqe_num;
    uint32_t stream_id;
    uint32_t dev_id;
    uint8_t reserved[4];
};

static_assert(sizeof(stars_channel_flag_info_t) == 64, "Flag info must be 64 bytes");
static_assert(sizeof(stars_channel_info_t) == 64, "Channel info must be 64 bytes");

// dlsym function pointer types
using RtStreamGetSqidFn = int (*)(const void* stream, uint32_t* sqId);
using RtStreamGetCqidFn = int (*)(const void* stream, uint32_t* cqId, uint32_t* logicCqId);
using HalSqCqQueryFn = drvError_t (*)(uint32_t devId, halSqCqQueryInfo* info);

static constexpr size_t SDMA_WORKSPACE_SIZE = 16 * 1024;

// State for cleanup
static std::vector<void*> g_prefetch_streams;
static void* g_workspace_device_ptr = nullptr;
static int g_cached_device_id = -1;
static int g_cached_channel_count = 0;
static HalSqCqQueryFn g_hal_sq_cq_query = nullptr;
static bool g_hal_sq_cq_query_resolved = false;

struct HostPrefetchSetupTimer {
    std::chrono::steady_clock::time_point start{std::chrono::steady_clock::now()};
    const char* outcome{"unknown"};
    int channel_count{0};

    ~HostPrefetchSetupTimer()
    {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::steady_clock::now() - start);
        LOG_INFO(
            "SDMA prefetch: host_prefetch_setup outcome=%s channels=%d total=%.3fms",
            outcome, channel_count, elapsed.count()
        );
    }
};

static void resolve_hal_sq_cq_query()
{
    if (g_hal_sq_cq_query_resolved) {
        return;
    }
    g_hal_sq_cq_query = reinterpret_cast<HalSqCqQueryFn>(dlsym(RTLD_DEFAULT, "halSqCqQuery"));
    if (g_hal_sq_cq_query == nullptr) {
        LOG_INFO("SDMA prefetch: halSqCqQuery not found: %s", dlerror());
    }
    g_hal_sq_cq_query_resolved = true;
}

static uint64_t query_value_to_u64(const uint32_t value[SQCQ_QUERY_INFO_LENGTH])
{
    return static_cast<uint64_t>(value[0]) | (static_cast<uint64_t>(value[1]) << 32);
}

static uint64_t query_value_to_u64_reversed(const uint32_t value[SQCQ_QUERY_INFO_LENGTH])
{
    return static_cast<uint64_t>(value[1]) | (static_cast<uint64_t>(value[0]) << 32);
}

static bool query_sqcq_u64(
    uint32_t dev_id, uint32_t ts_id, drvSqCqType_t type, uint32_t sq_id, uint32_t cq_id, drvSqCqPropType_t prop,
    uint64_t* value
)
{
    if (g_hal_sq_cq_query == nullptr) {
        return false;
    }
    halSqCqQueryInfo query = {};
    query.type = type;
    query.tsId = ts_id;
    query.sqId = sq_id;
    query.cqId = cq_id;
    query.prop = prop;
    drvError_t rc = g_hal_sq_cq_query(dev_id, &query);
    if (rc != 0) {
        return false;
    }
    if (prop == DRV_SQCQ_PROP_SQ_REG_BASE) {
        *value = query_value_to_u64_reversed(query.value);
    } else {
        *value = query_value_to_u64(query.value);
    }
    return true;
}

static int resolve_forced_ts_id()
{
    const char* env = std::getenv("PTO_SDMA_HAL_TS_ID");
    if (env == nullptr || *env == '\0') {
        return -1;
    }

    char* end = nullptr;
    errno = 0;
    long parsed = std::strtol(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0' || (parsed != 0 && parsed != 1)) {
        LOG_INFO("SDMA prefetch: ignore invalid PTO_SDMA_HAL_TS_ID=%s", env);
        return -1;
    }
    LOG_INFO("SDMA prefetch: forcing HAL ts_id=%ld", parsed);
    return static_cast<int>(parsed);
}

static bool populate_channel_with_hal(stars_channel_info_t& ch, uint32_t channel_idx, uint32_t channel_count)
{
    resolve_hal_sq_cq_query();
    if (g_hal_sq_cq_query == nullptr) {
        return false;
    }

    static constexpr uint32_t ts_ids[] = {0, 1};
    int forced_ts = resolve_forced_ts_id();
    auto try_one = [&](uint32_t ts_id) -> bool {
        uint64_t sq_base = 0;
        uint64_t sq_reg_base = 0;
        uint64_t sq_depth = 0;
        bool ok =
            query_sqcq_u64(ch.dev_id, ts_id, DRV_NORMAL_TYPE, ch.sq_id, ch.cq_id, DRV_SQCQ_PROP_SQ_BASE, &sq_base) &&
            query_sqcq_u64(ch.dev_id, ts_id, DRV_NORMAL_TYPE, ch.sq_id, ch.cq_id, DRV_SQCQ_PROP_SQ_REG_BASE, &sq_reg_base) &&
            query_sqcq_u64(ch.dev_id, ts_id, DRV_NORMAL_TYPE, ch.sq_id, ch.cq_id, DRV_SQCQ_PROP_SQ_DEPTH, &sq_depth);
        if (!ok || sq_base == 0 || sq_reg_base == 0 || sq_depth == 0) {
            return false;
        }

        ch.sq_base = sq_base;
        ch.sq_reg_base = sq_reg_base;
        ch.sq_depth = static_cast<uint32_t>(sq_depth);
        if (channel_idx < 4 || channel_idx + 1 == channel_count) {
            LOG_INFO(
                "SDMA prefetch: host HAL channel[%u] sid=%u sq=%u cq=%u dev=%u ts=%u type=%d base=0x%llx reg=0x%llx depth=%u",
                channel_idx, ch.stream_id, ch.sq_id, ch.cq_id, ch.dev_id, ts_id, static_cast<int>(DRV_NORMAL_TYPE),
                static_cast<unsigned long long>(ch.sq_base), static_cast<unsigned long long>(ch.sq_reg_base), ch.sq_depth
            );
        }
        return true;
    };

    if (forced_ts >= 0) {
        uint32_t ts_id = static_cast<uint32_t>(forced_ts);
        if (try_one(ts_id)) {
            return true;
        }
        LOG_INFO(
            "SDMA prefetch: host HAL forced ts=%u miss for channel[%u] sid=%u sq=%u cq=%u dev=%u",
            ts_id, channel_idx, ch.stream_id, ch.sq_id, ch.cq_id, ch.dev_id
        );
        return false;
    }

    for (uint32_t ts_id : ts_ids) {
        if (try_one(ts_id)) {
            return true;
        }
    }

    LOG_INFO(
        "SDMA prefetch: host HAL query failed for channel[%u] sid=%u sq=%u cq=%u dev=%u", channel_idx, ch.stream_id,
        ch.sq_id, ch.cq_id, ch.dev_id
    );
    return false;
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
    HostPrefetchSetupTimer setup_timer;
    if (!sdma_prefetch_enabled_by_env()) {
        setup_timer.outcome = "disabled_by_env";
        LOG_INFO("SDMA prefetch: disabled by PTO_ENABLE_SDMA_PREFETCH");
        return nullptr;
    }

    channel_count = resolve_channel_count(channel_count);
    setup_timer.channel_count = channel_count;
    if (channel_count <= 0) {
        setup_timer.outcome = "invalid_channel_count";
        LOG_INFO("SDMA prefetch: disabled (invalid channel_count=%d)", channel_count);
        return nullptr;
    }

    // Resolve all required APIs via dlsym
    auto rtStreamGetSqid =
        reinterpret_cast<RtStreamGetSqidFn>(dlsym(RTLD_DEFAULT, "rtStreamGetSqid"));
    auto rtStreamGetCqid =
        reinterpret_cast<RtStreamGetCqidFn>(dlsym(RTLD_DEFAULT, "rtStreamGetCqid"));
    if (!rtStreamGetSqid || !rtStreamGetCqid) {
        setup_timer.outcome = "missing_rt_symbols";
        LOG_INFO("SDMA prefetch: rtStreamGetSqid/Cqid not found, skipping");
        return nullptr;
    }

    int rc;

    int32_t dev_id = -1;
    aclrtGetDevice(&dev_id);
    void* ctx = nullptr;
    aclrtGetCurrentContext(&ctx);
    LOG_INFO(
        "SDMA prefetch: setup start (device=%d ctx=%p channels=%d provider_root=%s opp=%s)", dev_id, ctx, channel_count,
        std::getenv("PTO_SDMA_PROVIDER_ROOT") ? std::getenv("PTO_SDMA_PROVIDER_ROOT") : "(unset)",
        std::getenv("ASCEND_OPP_PATH") ? std::getenv("ASCEND_OPP_PATH") : "(unset)"
    );

    // Fast path: reuse existing channels/workspace when device and count match.
    if (g_workspace_device_ptr != nullptr &&
        g_cached_device_id == dev_id &&
        g_cached_channel_count == channel_count &&
        static_cast<int>(g_prefetch_streams.size()) == channel_count) {
        setup_timer.outcome = "cached_reuse";
        LOG_INFO("SDMA prefetch: reusing cached STARS workspace (device=%d channels=%d)", dev_id, channel_count);
        return g_workspace_device_ptr;
    }

    // If something is already initialized but doesn't match the requested config, tear it down first.
    if (!g_prefetch_streams.empty() || g_workspace_device_ptr != nullptr) {
        host_prefetch_teardown(nullptr);
    }

    void* workspace = nullptr;
    std::vector<stars_channel_info_t> channel_infos(static_cast<size_t>(channel_count));
    g_prefetch_streams.reserve(static_cast<size_t>(channel_count));
    for (int i = 0; i < channel_count; ++i) {
        void* stream = nullptr;
        rc = aclrtCreateStreamWithConfig(reinterpret_cast<aclrtStream*>(&stream), 0, 0x20);  // ACL_STREAM_DEVICE_USE_ONLY
        if (rc != 0 || !stream) {
            setup_timer.outcome = "stream_create_failed";
            LOG_INFO("SDMA prefetch: create device stream %d/%d failed (rc=%d)", i, channel_count, rc);
            goto fail_streams;
        }
        g_prefetch_streams.push_back(stream);

        stars_channel_info_t& ch = channel_infos[static_cast<size_t>(i)];
        ch.dev_id = static_cast<uint32_t>(dev_id);

        int32_t stream_id = 0;
        aclrtStreamGetId(reinterpret_cast<aclrtStream>(stream), &stream_id);
        ch.stream_id = static_cast<uint32_t>(stream_id);
        rc = rtStreamGetSqid(stream, &ch.sq_id);
        if (rc != 0) {
            setup_timer.outcome = "get_sqid_failed";
            LOG_INFO("SDMA prefetch: get sqid failed for stream %d/%d (rc=%d)", i, channel_count, rc);
            goto fail_streams;
        }
        rc = rtStreamGetCqid(stream, &ch.cq_id, &ch.logic_cq_id);
        if (rc != 0) {
            setup_timer.outcome = "get_cqid_failed";
            LOG_INFO("SDMA prefetch: get cqid failed for stream %d/%d (rc=%d)", i, channel_count, rc);
            goto fail_streams;
        }
        if (!populate_channel_with_hal(ch, static_cast<uint32_t>(i), static_cast<uint32_t>(channel_count))) {
            setup_timer.outcome = "host_hal_query_failed";
            goto fail_streams;
        }

        if (i < 4 || i == channel_count - 1) {
            LOG_INFO(
                "SDMA prefetch: stream[%d/%d] created (sid=%d sq=%u cq=%u base=0x%llx reg=0x%llx depth=%u)",
                i, channel_count, stream_id, ch.sq_id, ch.cq_id,
                static_cast<unsigned long long>(ch.sq_base),
                static_cast<unsigned long long>(ch.sq_reg_base), ch.sq_depth
            );
        }
    }
    (void)ctx;

    rc = aclrtMalloc(&workspace, SDMA_WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != 0) {
        setup_timer.outcome = "workspace_malloc_failed";
        LOG_ERROR("SDMA prefetch: workspace malloc failed");
        goto fail_streams;
    }
    aclrtMemset(workspace, SDMA_WORKSPACE_SIZE, 0, SDMA_WORKSPACE_SIZE);
    LOG_INFO("SDMA prefetch: workspace allocated at %p size=%zu", workspace, SDMA_WORKSPACE_SIZE);

    {
        stars_channel_flag_info_t flag_info = {};
        flag_info.totalQueueNum = static_cast<uint32_t>(channel_infos.size());

        rc = aclrtMemcpy(workspace, sizeof(flag_info), &flag_info, sizeof(flag_info), ACL_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            setup_timer.outcome = "flag_info_copy_failed";
            LOG_ERROR("SDMA prefetch: copy flag info failed (rc=%d)", rc);
            goto fail_workspace;
        }

        size_t channel_infos_size = channel_infos.size() * sizeof(stars_channel_info_t);
        void* channel_info_dev = static_cast<uint8_t*>(workspace) + sizeof(stars_channel_flag_info_t);
        rc = aclrtMemcpy(channel_info_dev, channel_infos_size, channel_infos.data(), channel_infos_size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (rc != 0) {
            setup_timer.outcome = "channel_info_copy_failed";
            LOG_ERROR("SDMA prefetch: copy channel info failed (rc=%d)", rc);
            goto fail_workspace;
        }
    }

    g_workspace_device_ptr = workspace;
    g_cached_device_id = dev_id;
    g_cached_channel_count = channel_count;
    setup_timer.outcome = "initialized";
    LOG_INFO("SDMA prefetch: STARS channel IDs initialized for AICPU HAL query");
    return workspace;

fail_workspace:
    if (setup_timer.outcome == std::string("unknown")) {
        setup_timer.outcome = "fail_workspace";
    }
    aclrtFree(workspace);
fail_streams:
    if (setup_timer.outcome == std::string("unknown")) {
        setup_timer.outcome = "fail_streams";
    }
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
    for (void* stream : g_prefetch_streams) {
        aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream));
    }
    g_prefetch_streams.clear();
}

/**
 * @file device_prefetch.cpp
 * @brief SDMA Prefetch for Real Hardware (a2a3) via STARS SQ
 *
 * Implements AICPU-initiated SDMA prefetch by writing CMO PREFETCH SQEs
 * directly to the STARS submission queue and ringing the hardware doorbell.
 *
 * The host sets up STARS channels in a device workspace, and AICPU selects
 * one fixed channel per target core. This matches shmem's "stable executor
 * identity -> stable channel" model, but keeps a conservative single
 * outstanding SQE per channel because completion handling is not yet modeled.
 */

#include "aicpu/device_prefetch.h"
#include "aicpu/device_log.h"
#include "common/platform_config.h"

#include <cstring>

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

struct stars_sdma_cmo_sqe_t {
    uint8_t type : 6;
    uint8_t l1_lock : 1;
    uint8_t l1_unlock : 1;

    uint8_t ie : 2;
    uint8_t pre_p : 2;
    uint8_t post_p : 2;
    uint8_t wr_cqe : 1;
    uint8_t reserved_hdr : 1;

    uint16_t block_dim;
    uint16_t rt_streamid;
    uint16_t task_id;

    uint32_t res3;

    uint16_t res4;
    uint8_t kernel_credit;
    uint8_t ptr_mode : 1;
    uint8_t res5 : 7;

    uint32_t opcode : 8;
    uint32_t ie2 : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t d2d_offset_flag : 1;
    uint32_t res6 : 3;

    uint16_t src_streamid;
    uint16_t src_sub_streamid;
    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;

    uint32_t length;
    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;

    uint32_t src_offset_low;
    uint32_t dst_offset_low;
    uint16_t src_offset_high;
    uint16_t dst_offset_high;
    uint32_t res_last[1];
};

static_assert(sizeof(stars_sdma_cmo_sqe_t) == 64, "CMO SQE must be 64 bytes");
static_assert(sizeof(stars_channel_flag_info_t) == 64, "Flag info must be 64 bytes");

static constexpr uint8_t STARS_SQE_TYPE_SDMA = 11;
static constexpr uint8_t CMO_OPCODE_PREFETCH = 6;
static constexpr uint8_t DEFAULT_KERNEL_CREDIT = 254;

static bool g_prefetch_enabled = false;
static volatile stars_channel_info_t* g_channel_info = nullptr;
static uint32_t g_channel_count = 0;
static uint32_t g_prefetch_suppress_window = 0;
static volatile uint32_t g_prefetch_submit_lock[PLATFORM_MAX_CORES] = {0};
static volatile uint32_t g_prefetch_skip_queue_full[PLATFORM_MAX_CORES] = {0};
static volatile uint32_t g_prefetch_channel_suppress_remaining[PLATFORM_MAX_CORES] = {0};
static volatile uint32_t g_prefetch_skip_suppressed[PLATFORM_MAX_CORES] = {0};
static volatile uint32_t g_prefetch_attempt_count = 0;
static volatile uint32_t g_prefetch_issue_count = 0;
static volatile uint64_t g_prefetch_attempt_bytes = 0;
static volatile uint64_t g_prefetch_issue_bytes = 0;

static inline void prefetch_lock(int channel_idx)
{
    while (__atomic_test_and_set(&g_prefetch_submit_lock[channel_idx], __ATOMIC_ACQUIRE)) {
    }
}

static inline void prefetch_unlock(int channel_idx)
{
    __atomic_clear(&g_prefetch_submit_lock[channel_idx], __ATOMIC_RELEASE);
}

static inline uint32_t consume_prefetch_suppress_window(int channel_idx)
{
    while (true) {
        uint32_t remaining = __atomic_load_n(&g_prefetch_channel_suppress_remaining[channel_idx], __ATOMIC_RELAXED);
        if (remaining == 0) {
            return 0;
        }
        uint32_t desired = remaining - 1;
        if (__atomic_compare_exchange_n(
                &g_prefetch_channel_suppress_remaining[channel_idx], &remaining, desired, false,
                __ATOMIC_ACQ_REL, __ATOMIC_RELAXED
            )) {
            return remaining;
        }
    }
}

static void fill_cmo_prefetch_sqe(volatile stars_sdma_cmo_sqe_t* sqe,
                                  uint64_t src_addr, uint32_t length,
                                  uint16_t stream_id, uint16_t task_id)
{
    memset(const_cast<stars_sdma_cmo_sqe_t*>(sqe), 0, sizeof(stars_sdma_cmo_sqe_t));

    sqe->type = STARS_SQE_TYPE_SDMA;
    sqe->rt_streamid = stream_id;
    sqe->task_id = task_id;
    sqe->kernel_credit = DEFAULT_KERNEL_CREDIT;

    sqe->opcode = CMO_OPCODE_PREFETCH;
    sqe->length = length;

    sqe->src_addr_low = static_cast<uint32_t>(src_addr & 0xFFFFFFFFu);
    sqe->src_addr_high = static_cast<uint32_t>((src_addr >> 32) & 0xFFFFFFFFu);

    sqe->qos = 6;
    sqe->partid = 63;
    sqe->sssv = 1;
    sqe->dssv = 1;
    sqe->sns = 1;
    sqe->dns = 1;
}

void aicpu_prefetch_init(void* sdma_workspace, uint32_t suppress_window)
{
    g_prefetch_enabled = false;
    g_channel_info = nullptr;
    g_channel_count = 0;
    g_prefetch_suppress_window = suppress_window;
    g_prefetch_attempt_count = 0;
    g_prefetch_issue_count = 0;
    g_prefetch_attempt_bytes = 0;
    g_prefetch_issue_bytes = 0;
    for (int i = 0; i < PLATFORM_MAX_CORES; ++i) {
        g_prefetch_channel_suppress_remaining[i] = 0;
        g_prefetch_skip_suppressed[i] = 0;
    }

    if (sdma_workspace == nullptr) {
        DEV_ALWAYS("SDMA prefetch: disabled (no workspace provided)");
        return;
    }

    uint8_t* base = static_cast<uint8_t*>(sdma_workspace);
    auto* flag_info = reinterpret_cast<volatile stars_channel_flag_info_t*>(base);
    g_channel_info = reinterpret_cast<volatile stars_channel_info_t*>(base + sizeof(stars_channel_flag_info_t));
    DEV_ALWAYS(
        "SDMA prefetch: workspace=%p flag_info=%p channel_info=%p flag=0x%x totalQueueNum=%u", sdma_workspace,
        const_cast<stars_channel_flag_info_t*>(flag_info), const_cast<stars_channel_info_t*>(g_channel_info),
        flag_info->flag, flag_info->totalQueueNum
    );
    g_channel_count = flag_info->totalQueueNum;
    if (g_channel_count == 0 || g_channel_count > PLATFORM_MAX_CORES) {
        DEV_ALWAYS("SDMA prefetch: invalid channel count %u", g_channel_count);
        g_channel_info = nullptr;
        g_channel_count = 0;
        return;
    }

    if (g_channel_info->sq_base == 0 || g_channel_info->sq_reg_base == 0 || g_channel_info->sq_depth == 0) {
        DEV_ALWAYS("SDMA prefetch: disabled (channel info not initialized: sq_base=%llu sq_reg=%llu depth=%u)",
                   (unsigned long long)g_channel_info->sq_base,
                   (unsigned long long)g_channel_info->sq_reg_base,
                   g_channel_info->sq_depth);
        g_channel_info = nullptr;
        return;
    }

    g_prefetch_enabled = true;
    for (uint32_t i = 0; i < g_channel_count && i < 4; ++i) {
        volatile stars_channel_info_t* ch = g_channel_info + i;
        DEV_ALWAYS(
            "SDMA prefetch: channel[%u] sq_head=%u sq_tail=%u sq_base=0x%llx sq_reg=0x%llx depth=%u sq_id=%u cq_id=%u stream=%u",
            i, ch->sq_head, ch->sq_tail, (unsigned long long)ch->sq_base, (unsigned long long)ch->sq_reg_base,
            ch->sq_depth, ch->sq_id, ch->cq_id, ch->stream_id
        );
    }
    DEV_ALWAYS("SDMA prefetch: enabled (channels=%u sq_base=0x%llx sq_reg=0x%llx depth=%u stream=%u)",
               g_channel_count,
               (unsigned long long)g_channel_info->sq_base,
               (unsigned long long)g_channel_info->sq_reg_base,
               g_channel_info->sq_depth,
               g_channel_info->stream_id);
}

void aicpu_prefetch_deinit()
{
    g_prefetch_enabled = false;
    g_channel_info = nullptr;
    g_channel_count = 0;
    for (int i = 0; i < PLATFORM_MAX_CORES; ++i) {
        __atomic_store_n(&g_prefetch_submit_lock[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_skip_queue_full[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_channel_suppress_remaining[i], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_prefetch_skip_suppressed[i], 0, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&g_prefetch_attempt_count, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_issue_count, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_attempt_bytes, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_issue_bytes, 0, __ATOMIC_RELEASE);
}

void aicpu_prefetch_tensor(void* addr, size_t size, int channel_idx)
{
    if (!g_prefetch_enabled || addr == nullptr || size == 0) {
        return;
    }

    if (g_channel_info == nullptr || g_channel_count == 0 || channel_idx < 0) {
        return;
    }

    channel_idx %= static_cast<int>(g_channel_count);
    __atomic_add_fetch(&g_prefetch_attempt_count, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_prefetch_attempt_bytes, static_cast<uint64_t>(size), __ATOMIC_RELAXED);

    uint32_t suppress_remaining = consume_prefetch_suppress_window(channel_idx);
    if (suppress_remaining != 0) {
        uint32_t skipped = __atomic_add_fetch(&g_prefetch_skip_suppressed[channel_idx], 1, __ATOMIC_RELAXED);
        if (skipped <= 4 || (skipped % 64) == 0) {
            DEV_ALWAYS(
                "SDMA prefetch: suppressed, skip issue (channel=%d skipped=%u remaining_after=%u)",
                channel_idx, skipped, suppress_remaining - 1
            );
        }
        return;
    }

    volatile stars_channel_info_t* ch = g_channel_info + channel_idx;
    prefetch_lock(channel_idx);

    uint32_t sq_head = __atomic_load_n(&ch->sq_head, __ATOMIC_ACQUIRE);
    uint32_t sq_tail = __atomic_load_n(&ch->sq_tail, __ATOMIC_ACQUIRE);
    uint32_t sq_depth = ch->sq_depth;
    if (sq_depth == 0) {
        DEV_ALWAYS("SDMA prefetch: channel %d has zero depth", channel_idx);
        prefetch_unlock(channel_idx);
        return;
    }

    uint32_t new_tail = (sq_tail + 1) % sq_depth;
    if (new_tail == sq_head) {
        uint32_t skipped = __atomic_add_fetch(&g_prefetch_skip_queue_full[channel_idx], 1, __ATOMIC_RELAXED);
        if (skipped <= 4 || (skipped % 64) == 0) {
            DEV_ALWAYS("SDMA prefetch: queue full, skip issue (channel=%d head=%u tail=%u depth=%u skipped=%u)",
                       channel_idx, sq_head, sq_tail, sq_depth, skipped);
        }
        prefetch_unlock(channel_idx);
        return;
    }

    volatile stars_sdma_cmo_sqe_t* sqe = reinterpret_cast<volatile stars_sdma_cmo_sqe_t*>(ch->sq_base);
    sqe += sq_tail;

    uint16_t task_id = static_cast<uint16_t>(sq_tail - ch->sq_head);
    fill_cmo_prefetch_sqe(sqe,
                          reinterpret_cast<uint64_t>(addr),
                          static_cast<uint32_t>(size),
                          static_cast<uint16_t>(ch->stream_id),
                          task_id);
    uint32_t issue_idx = __atomic_add_fetch(&g_prefetch_issue_count, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_prefetch_issue_bytes, static_cast<uint64_t>(size), __ATOMIC_RELAXED);
    if (issue_idx <= 8) {
        DEV_ALWAYS(
            "SDMA prefetch: issue[%u] channel=%d addr=0x%llx size=%zu sq_head=%u sq_tail=%u new_tail=%u depth=%u stream=%u doorbell=0x%llx",
            issue_idx, channel_idx, (unsigned long long)reinterpret_cast<uint64_t>(addr), size, sq_head, sq_tail,
            new_tail, sq_depth, ch->stream_id, (unsigned long long)(ch->sq_reg_base + 8)
        );
    }

    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&ch->sq_tail, new_tail, __ATOMIC_RELEASE);
    __atomic_store_n(&g_prefetch_channel_suppress_remaining[channel_idx], g_prefetch_suppress_window, __ATOMIC_RELEASE);

    volatile uint32_t* doorbell = reinterpret_cast<volatile uint32_t*>(ch->sq_reg_base + 8);
    *doorbell = new_tail;

    prefetch_unlock(channel_idx);
}

bool aicpu_prefetch_available()
{
    return g_prefetch_enabled;
}

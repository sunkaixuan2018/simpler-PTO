/**
 * Paged Attention + AllGather (Manual) orchestration for tensormap_and_ringbuffer.
 *
 * Flow: Phase 1 - Paged Attention (QK->SF->PV->UP)
 *       Phase 2 - WindowMemCopyIn -> CommBarrier(pre) -> AllGatherManual
 *                 -> WindowMemCopyOut -> CommBarrier(post)
 *
 * TODO: 完整实现需合并 paged_attention_orch 与 allgather_orch 逻辑。
 * 当前为占位实现，需根据 runtime 的 args 布局补充 Phase 1 和 Phase 2。
 * 参考：examples/tensormap_and_ringbuffer/paged_attention 与 allgather_Manual。
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5
#define FUNC_WIN_MEMCOPY_IN  6
#define FUNC_ALLGATHER       7
#define FUNC_WIN_MEMCOPY_OUT 8
#define FUNC_COMM_BARRIER    9

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 22,  /* query, key, value, block_table, context_lens,
                                      attn_out, allgather_out, config, 7 sizes,
                                      device_ctx_ptr, win_in_base, win_out_base,
                                      n_ranks, root, rank_id */
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)args;
    (void)arg_count;
    pto2_rt_init_tensor_pool(rt);
    /* TODO: 实现 Phase 1 (Paged Attention) + Phase 2 (AllGather) */
    LOG_INFO(rt, "paged_attention_allgather_Manual: placeholder - full impl needed");
}

}  // extern "C"

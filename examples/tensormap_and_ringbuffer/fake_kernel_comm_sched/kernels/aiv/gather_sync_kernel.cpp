/**
 * GatherSync: synchronous TGATHER collective kernel.
 * Interchangeable with GatherAsync — same args, same result.
 *
 * Args (8):
 *   args[0] = dst (TensorData*)
 *   args[1] = src (TensorData*)
 *   args[2] = sync_done (TensorData*, dependency only)
 *   args[3] = device_ctx_ptr (scalar)
 *   args[4] = nranks (scalar)
 *   args[5] = root (scalar)
 *   args[6] = sdma_workspace_ptr (scalar, unused by sync variant)
 *   args[7] = debug_poll_counts (TensorData*, unused by sync variant)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include "hccl_context.h"
#include "hccl_helpers.h"

#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// MAX_GATHER_COUNT covers the largest benchmark size (256K total / 4 ranks / 4 bytes = 16384).
// TileData is allocated at this maximum size; actual gather count is derived from the dst
// tensor at runtime so the same compiled kernel handles all benchmark data sizes.
static constexpr size_t MAX_GATHER_COUNT = 16384;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* dst_td = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* src_td = reinterpret_cast<__gm__ TensorData*>(args[1]);
    (void)args[2];  // sync_done dependency
    __gm__ HcclDeviceContext* hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext*>(args[3]);
    int nranks = static_cast<int>(args[4]);
    int root = static_cast<int>(args[5]);
    (void)args[6];  // sdma_workspace_ptr unused in sync variant
    (void)args[7];  // debug_poll_counts unused in sync variant

    __gm__ float* dst = reinterpret_cast<__gm__ float*>(dst_td->buffer.addr);
    __gm__ float* src = reinterpret_cast<__gm__ float*>(src_td->buffer.addr);

    // Derive actual gather_count from dst tensor size at runtime
    size_t gather_count = dst_td->buffer.size / (static_cast<size_t>(nranks) * sizeof(float));

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, MAX_GATHER_COUNT, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn srcShape(1, 1, 1, 1, gather_count);
    StrideDyn srcStride(gather_count, gather_count, gather_count, gather_count, 1);

    ShapeDyn dstShape(1, 1, 1, nranks, gather_count);
    StrideDyn dstStride(nranks * gather_count, nranks * gather_count, nranks * gather_count, gather_count, 1);
    Global dstG(dst, dstShape, dstStride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ float* remoteSrc = HcclRemotePtr(hcclCtx, src, i);
        tensors[i] = Global(remoteSrc, srcShape, srcStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, root);

    TileData ubTile(1, MAX_GATHER_COUNT);
    TASSIGN(ubTile, 0x0);

    if (my_rank == root) {
        pto::comm::TGATHER(pg, dstG, ubTile);
    }
    pipe_barrier(PIPE_ALL);
}

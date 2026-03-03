/**
 * Multi-card TGATHER: device kernel + host RunGatherKernel / RunGather.
 * Adapted from pto-comm-isa tests/npu/a2a3/comm/st/testcase/tgather/tgather_kernel.cpp
 * Build with CANN toolchain on a2a3 (host + device in one compile unit).
 */

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "comm_common.hpp"

#define COMM_GATHER_COUNT 512

// ============================================================================
// Device kernel: root gathers data from all ranks
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TGatherKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ HcclDeviceContext *hcclCtx, int nranks,
                                         int root)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn srcShape(1, 1, 1, 1, count);
    StrideDyn srcStride(count, count, count, count, 1);
    ShapeDyn dstShape(1, 1, 1, nranks, count);
    StrideDyn dstStride(nranks * count, nranks * count, nranks * count, count, 1);
    Global dstG(dst, dstShape, dstStride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteSrc = HcclRemotePtr(hcclCtx, src, i);
        tensors[i] = Global(remoteSrc, srcShape, srcStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, root);

    TileData ubTile(1, count);
    TASSIGN(ubTile, 0x0);

    if (my_rank == root) {
        pto::comm::TGATHER(pg, dstG, ubTile);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host: RunGatherKernel for one rank
// ============================================================================
template <typename T, size_t count>
bool RunGatherKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo,
                    int root)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;

    size_t src_size = count * sizeof(T);
    size_t dst_size = n_ranks * count * sizeof(T);
    void *src_ptr = WindowAlloc(localWinBase, winOffset, src_size);
    void *dst_ptr = WindowAlloc(localWinBase, winOffset, dst_size);

    T *src_host = nullptr;
    T *dst_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&src_host), src_size) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&dst_host), dst_size) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        src_host[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < n_ranks * count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }

    aclrtMemcpy(src_ptr, src_size, src_host, src_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rank_id == root) {
        aclrtMemcpy(dst_ptr, dst_size, dst_host, dst_size, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    TGatherKernelImpl<T, count><<<1, nullptr, ctx.stream>>>((T *)dst_ptr, (T *)src_ptr, ctx.deviceCtx, n_ranks, root);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;
    if (rank_id == root) {
        aclrtMemcpy(dst_host, dst_size, dst_ptr, dst_size, ACL_MEMCPY_DEVICE_TO_HOST);

        for (int r = 0; r < n_ranks; ++r) {
            for (size_t i = 0; i < count; ++i) {
                T expected = static_cast<T>(i + r * 10000);
                T actual = dst_host[r * count + i];
                if (actual != expected) {
                    std::cout << "Rank " << rank_id << " validation failed at rank " << r << " index " << i
                              << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok)
                break;
        }

        if (is_ok) {
            std::cout << "[comm_gather] Rank " << root << ": TGATHER OK (" << n_ranks << " ranks, " << count
                      << " elements/rank)." << std::endl;
        }
    }

    aclrtFreeHost(src_host);
    aclrtFreeHost(dst_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunGather(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunGatherKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo, 0);
        });
}

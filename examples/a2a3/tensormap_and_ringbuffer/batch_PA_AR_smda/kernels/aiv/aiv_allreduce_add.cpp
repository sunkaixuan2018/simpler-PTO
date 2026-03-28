/**
 * func_id 8 — 本地 chunk allreduce add：out_chunk[i] += peer_out_chunk[i]。
 *
 * args[0] out_chunk Tensor* (inout), args[1] peer_out_chunk Tensor* (input), args[2] total_elems (int64)
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"

#include "tensor.h"

using namespace pto;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* peer_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    int total_elems = static_cast<int>(args[2]);

    __gm__ float* out_data =
        reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;
    __gm__ float* peer_data =
        reinterpret_cast<__gm__ float*>(peer_tensor->buffer.addr) + peer_tensor->start_offset;

    constexpr int kChunk = 16;
    using FlatShape = Shape<1, 1, 1, 1, kChunk>;
    using FlatStride = Stride<kChunk, kChunk, kChunk, kChunk, 1>;
    using FlatGlobal = GlobalTensor<float, FlatShape, FlatStride>;
    using TileData = Tile<TileType::Vec, float, 1, kChunk, BLayout::RowMajor, -1, -1>;

    TileData aTile(1, kChunk);
    TileData bTile(1, kChunk);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x10000);

    for (int base = 0; base < total_elems; base += kChunk) {
        FlatGlobal outG(out_data + base);
        FlatGlobal peerG(peer_data + base);

        TLOAD(aTile, outG);
        TLOAD(bTile, peerG);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TADD(aTile, aTile, bTile);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TSTORE(outG, aTile);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }
}

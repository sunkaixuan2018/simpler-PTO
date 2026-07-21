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

#include <acl/acl.h>
#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

extern "C" void LaunchFabricTload(float *remote_src, float *local_dst, void *stream);

namespace {

constexpr size_t kElementCount = 64;
constexpr size_t kWindowBytes = 2U * 1024U * 1024U;

[[noreturn]] void AbortJob(int rank, const std::string &message, int error_code = 1) {
    std::cerr << "[rank " << rank << "] ERROR: " << message << std::endl;
    MPI_Abort(MPI_COMM_WORLD, error_code == 0 ? 1 : error_code);
    std::abort();
}

void CheckAcl(int rank, aclError status, const char *operation) {
    if (status != ACL_SUCCESS) {
        std::ostringstream os;
        os << operation << " failed, acl status=" << status;
        AbortJob(rank, os.str(), status);
    }
}

void CheckMpi(int rank, int status, const char *operation) {
    if (status != MPI_SUCCESS) {
        std::ostringstream os;
        os << operation << " failed, MPI status=" << status;
        AbortJob(rank, os.str(), status);
    }
}

struct Topology {
    int local_rank{-1};
    int local_size{0};
    int peer_rank{-1};
    std::string hostname;
    std::string peer_hostname;
};

Topology DiscoverTwoHostPeer(int rank, int world_size, MPI_Comm *local_comm) {
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, local_comm) != MPI_SUCCESS) {
        AbortJob(rank, "MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) failed");
    }

    Topology topo;
    MPI_Comm_rank(*local_comm, &topo.local_rank);
    MPI_Comm_size(*local_comm, &topo.local_size);

    // Use MPI's shared-memory communicator, not hostname text, as the node
    // identity.  This remains correct even when two machines have a stale or
    // identical hostname configuration.
    int node_leader = rank;
    MPI_Allreduce(&rank, &node_leader, 1, MPI_INT, MPI_MIN, *local_comm);

    char local_name[MPI_MAX_PROCESSOR_NAME] = {};
    int name_length = 0;
    MPI_Get_processor_name(local_name, &name_length);
    topo.hostname.assign(local_name, static_cast<size_t>(name_length));

    std::vector<char> all_names(static_cast<size_t>(world_size) * MPI_MAX_PROCESSOR_NAME, '\0');
    std::vector<int> all_local_ranks(static_cast<size_t>(world_size), -1);
    std::vector<int> all_node_leaders(static_cast<size_t>(world_size), -1);
    MPI_Allgather(
        local_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD
    );
    MPI_Allgather(&topo.local_rank, 1, MPI_INT, all_local_ranks.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&node_leader, 1, MPI_INT, all_node_leaders.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::set<int> nodes;
    for (int candidate = 0; candidate < world_size; ++candidate) {
        nodes.emplace(all_node_leaders[static_cast<size_t>(candidate)]);
    }
    if (nodes.size() != 2) {
        std::ostringstream os;
        os << "this smoke requires exactly two shared-memory nodes, discovered " << nodes.size();
        AbortJob(rank, os.str());
    }

    for (int candidate = 0; candidate < world_size; ++candidate) {
        const std::string candidate_host(all_names.data() + static_cast<size_t>(candidate) * MPI_MAX_PROCESSOR_NAME);
        if (all_node_leaders[static_cast<size_t>(candidate)] != node_leader &&
            all_local_ranks[static_cast<size_t>(candidate)] == topo.local_rank) {
            if (topo.peer_rank >= 0) {
                AbortJob(rank, "found more than one cross-host peer with the same local rank");
            }
            topo.peer_rank = candidate;
            topo.peer_hostname = candidate_host;
        }
    }
    if (topo.peer_rank < 0) {
        AbortJob(rank, "cannot find a cross-host peer with the same local rank");
    }
    if (world_size != topo.local_size * 2) {
        AbortJob(rank, "the two hosts must run the same number of ranks");
    }
    return topo;
}

float Pattern(int rank, size_t index) { return static_cast<float>(rank * 1000 + static_cast<int>(index)); }

}  // namespace

int main(int argc, char **argv) {
    const int mpi_status = MPI_Init(&argc, &argv);
    if (mpi_status != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed, status=" << mpi_status << std::endl;
        return 1;
    }

    int rank = -1;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm local_comm = MPI_COMM_NULL;
    const Topology topo = DiscoverTwoHostPeer(rank, world_size, &local_comm);

    CheckAcl(rank, aclInit(nullptr), "aclInit");
    uint32_t device_count = 0;
    CheckAcl(rank, aclrtGetDeviceCount(&device_count), "aclrtGetDeviceCount");
    if (topo.local_rank < 0 || static_cast<uint32_t>(topo.local_rank) >= device_count) {
        std::ostringstream os;
        os << "local rank " << topo.local_rank << " has no matching device; device_count=" << device_count;
        AbortJob(rank, os.str());
    }
    CheckAcl(rank, aclrtSetDevice(topo.local_rank), "aclrtSetDevice");

    std::cout << "[rank " << rank << "] host=" << topo.hostname << " local_rank=" << topo.local_rank
              << " device=" << topo.local_rank << " peer_rank=" << topo.peer_rank << " peer_host=" << topo.peer_hostname
              << std::endl;

    A3FabricWindow window;
    CheckAcl(rank, window.CreateLocal(topo.local_rank, kWindowBytes), "A3FabricWindow::CreateLocal");

    aclrtMemFabricHandle local_shareable_handle{};
    CheckAcl(rank, window.Export(&local_shareable_handle), "aclrtMemExportToShareableHandleV2(FABRIC)");

    const int local_fabric_handle_bytes = static_cast<int>(sizeof(aclrtMemFabricHandle));
    int min_fabric_handle_bytes = 0;
    int max_fabric_handle_bytes = 0;
    CheckMpi(
        rank, MPI_Allreduce(&local_fabric_handle_bytes, &min_fabric_handle_bytes, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce(min Fabric handle size)"
    );
    CheckMpi(
        rank, MPI_Allreduce(&local_fabric_handle_bytes, &max_fabric_handle_bytes, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(max Fabric handle size)"
    );
    if (min_fabric_handle_bytes != max_fabric_handle_bytes) {
        AbortJob(rank, "CANN Fabric handle size differs between hosts");
    }

    std::vector<aclrtMemFabricHandle> all_shareable_handles(static_cast<size_t>(world_size));
    CheckMpi(
        rank,
        MPI_Allgather(
            &local_shareable_handle, local_fabric_handle_bytes, MPI_BYTE, all_shareable_handles.data(),
            local_fabric_handle_bytes, MPI_BYTE, MPI_COMM_WORLD
        ),
        "MPI_Allgather(Fabric handles)"
    );

    const uint64_t local_window_bytes = window.size();
    std::vector<uint64_t> all_window_bytes(static_cast<size_t>(world_size), 0);
    CheckMpi(
        rank,
        MPI_Allgather(&local_window_bytes, 1, MPI_UINT64_T, all_window_bytes.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD),
        "MPI_Allgather(Fabric window sizes)"
    );
    const uint64_t peer_window_bytes = all_window_bytes[static_cast<size_t>(topo.peer_rank)];
    if (peer_window_bytes == 0 || peer_window_bytes > std::numeric_limits<size_t>::max()) {
        AbortJob(rank, "peer Fabric window size is invalid");
    }
    CheckAcl(
        rank,
        window.ImportPeer(
            all_shareable_handles[static_cast<size_t>(topo.peer_rank)], static_cast<size_t>(peer_window_bytes)
        ),
        "aclrtMemImportFromShareableHandleV2(FABRIC)"
    );
    CheckMpi(rank, MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(Fabric mappings ready)");
    std::cout << "[rank " << rank << "] CANN Fabric peer mapping OK" << std::endl;

    const size_t payload_bytes = kElementCount * sizeof(float);
    if (window.size() < 2 * payload_bytes) {
        AbortJob(rank, "CANN Fabric window is smaller than the smoke payload");
    }
    auto *local_source = static_cast<float *>(window.local_base());
    auto *local_output = local_source + kElementCount;

    std::vector<float> source(kElementCount);
    std::vector<float> zeros(kElementCount, 0.0F);
    std::vector<float> result(kElementCount, 0.0F);
    for (size_t i = 0; i < kElementCount; ++i) {
        source[i] = Pattern(rank, i);
    }
    CheckAcl(
        rank, aclrtMemcpy(local_source, payload_bytes, source.data(), payload_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy(source H2D)"
    );
    CheckAcl(
        rank, aclrtMemcpy(local_output, payload_bytes, zeros.data(), payload_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy(output H2D)"
    );

    // All source buffers must be initialized before any rank issues a remote TLOAD.
    CheckMpi(rank, MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(source ready)");
    std::cout << "[rank " << rank << "] Fabric source ready" << std::endl;

    auto *remote_source = static_cast<float *>(window.peer_base());
    if (remote_source == nullptr) {
        AbortJob(rank, "CANN Fabric peer address is null");
    }
    std::cout << "[rank " << rank << "] Fabric peer GVA(peer=" << topo.peer_rank << ") OK" << std::endl;

    aclrtStream stream = nullptr;
    CheckAcl(rank, aclrtCreateStream(&stream), "aclrtCreateStream");

    LaunchFabricTload(remote_source, local_output, stream);
    CheckAcl(rank, aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    std::cout << "[rank " << rank << "] PTO TLOAD/TSTORE stream sync OK" << std::endl;
    CheckAcl(
        rank, aclrtMemcpy(result.data(), payload_bytes, local_output, payload_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy(result D2H)"
    );

    int local_ok = 1;
    for (size_t i = 0; i < kElementCount; ++i) {
        const float expected = Pattern(topo.peer_rank, i);
        if (result[i] != expected) {
            std::cerr << "[rank " << rank << "] verify mismatch at index " << i << ": got=" << result[i]
                      << " expected=" << expected << " from peer=" << topo.peer_rank << std::endl;
            local_ok = 0;
            break;
        }
    }

    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (local_ok != 0) {
        std::cout << "[rank " << rank << "] cross-host CANN Fabric + PTO TLOAD/TSTORE verify OK" << std::endl;
    }

    // Keep all peer source buffers alive until every remote read has completed.
    CheckMpi(rank, MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(remote reads complete)");
    int cleanup_ok = 1;
    if (aclrtDestroyStream(stream) != ACL_SUCCESS) {
        std::cerr << "[rank " << rank << "] aclrtDestroyStream failed" << std::endl;
        cleanup_ok = 0;
    }
    if (window.Destroy() != ACL_SUCCESS) {
        std::cerr << "[rank " << rank << "] CANN Fabric window cleanup failed" << std::endl;
        cleanup_ok = 0;
    }
    if (aclrtResetDevice(topo.local_rank) != ACL_SUCCESS) {
        std::cerr << "[rank " << rank << "] aclrtResetDevice failed" << std::endl;
        cleanup_ok = 0;
    }
    if (aclFinalize() != ACL_SUCCESS) {
        std::cerr << "[rank " << rank << "] aclFinalize failed" << std::endl;
        cleanup_ok = 0;
    }

    int global_cleanup_ok = 0;
    MPI_Allreduce(&cleanup_ok, &global_cleanup_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        if (global_ok != 0 && global_cleanup_ok != 0) {
            std::cout << "A3 CANN Fabric cross-host PTO TLOAD smoke PASS" << std::endl;
        } else {
            std::cerr << "A3 CANN Fabric cross-host PTO TLOAD smoke FAIL" << std::endl;
        }
    }

    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return global_ok != 0 && global_cleanup_ok != 0 ? 0 : 1;
}

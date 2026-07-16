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

#include <acl/acl.h>
#include <hccl/hccl.h>
#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void AbortJob(int rank, const std::string &message, int error_code = 1) {
    std::cerr << "[rank " << rank << "] ERROR: " << message << std::endl;
    MPI_Abort(MPI_COMM_WORLD, error_code == 0 ? 1 : error_code);
    std::abort();
}

void CheckMpi(int rank, int status, const char *operation) {
    if (status != MPI_SUCCESS) {
        std::ostringstream os;
        os << operation << " failed, MPI status=" << status;
        AbortJob(rank, os.str(), status);
    }
}

void CheckAcl(int rank, aclError status, const char *operation) {
    if (status != ACL_SUCCESS) {
        std::ostringstream os;
        os << operation << " failed, ACL status=" << status;
        AbortJob(rank, os.str(), status);
    }
}

void CheckHccl(int rank, HcclResult status, const char *operation) {
    if (status != HCCL_SUCCESS) {
        std::ostringstream os;
        os << operation << " failed, HCCL status=" << status;
        AbortJob(rank, os.str(), status);
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed" << std::endl;
        return 1;
    }

    int rank = -1;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    uint64_t count = 16;
    try {
        if (argc > 1) count = std::stoull(argv[1]);
    } catch (const std::exception &error) {
        AbortJob(rank, std::string("invalid count: ") + error.what());
    }
    if (count == 0) {
        AbortJob(rank, "count must be greater than zero");
    }
    if (world_size <= 0) {
        AbortJob(rank, "MPI world size must be greater than zero");
    }
    const uint64_t bytes_per_world_element = static_cast<uint64_t>(world_size) * sizeof(int32_t);
    const uint64_t max_count = std::numeric_limits<size_t>::max() / bytes_per_world_element;
    if (count > max_count) {
        AbortJob(rank, "count is too large for the AllGather receive buffer");
    }

    MPI_Comm local_comm = MPI_COMM_NULL;
    CheckMpi(
        rank, MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm),
        "MPI_Comm_split_type"
    );
    int local_rank = -1;
    MPI_Comm_rank(local_comm, &local_rank);

    CheckAcl(rank, aclInit(nullptr), "aclInit");
    uint32_t device_count = 0;
    CheckAcl(rank, aclrtGetDeviceCount(&device_count), "aclrtGetDeviceCount");
    if (local_rank < 0 || static_cast<uint32_t>(local_rank) >= device_count) {
        AbortJob(rank, "local MPI rank has no matching device");
    }
    CheckAcl(rank, aclrtSetDevice(local_rank), "aclrtSetDevice");

    HcclRootInfo root_info{};
    if (rank == 0) {
        CheckHccl(rank, HcclGetRootInfo(&root_info), "HcclGetRootInfo");
    }
    CheckMpi(rank, MPI_Bcast(&root_info, sizeof(root_info), MPI_BYTE, 0, MPI_COMM_WORLD), "MPI_Bcast(HcclRootInfo)");

    HcclComm comm = nullptr;
    CheckHccl(
        rank, HcclCommInitRootInfo(static_cast<uint32_t>(world_size), &root_info, static_cast<uint32_t>(rank), &comm),
        "HcclCommInitRootInfo"
    );
    std::cout << "[rank " << rank << "] HcclCommInitRootInfo OK" << std::endl;

    aclrtStream stream = nullptr;
    CheckAcl(rank, aclrtCreateStream(&stream), "aclrtCreateStream");

    const size_t send_bytes = count * sizeof(int32_t);
    const size_t recv_bytes = count * static_cast<uint64_t>(world_size) * sizeof(int32_t);
    void *send_buffer = nullptr;
    void *recv_buffer = nullptr;
    CheckAcl(rank, aclrtMalloc(&send_buffer, send_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(send)");
    CheckAcl(rank, aclrtMalloc(&recv_buffer, recv_bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(recv)");

    std::vector<int32_t> host_send(count, rank);
    std::vector<int32_t> host_recv(count * static_cast<uint64_t>(world_size), -1);
    CheckAcl(
        rank, aclrtMemcpy(send_buffer, send_bytes, host_send.data(), send_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy(send H2D)"
    );
    CheckAcl(rank, aclrtMemset(recv_buffer, recv_bytes, 0, recv_bytes), "aclrtMemset(recv)");

    CheckHccl(
        rank, HcclAllGather(send_buffer, recv_buffer, count, HCCL_DATA_TYPE_INT32, comm, stream), "HcclAllGather"
    );
    CheckAcl(rank, aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    std::cout << "[rank " << rank << "] HcclAllGather stream sync OK" << std::endl;
    CheckAcl(
        rank, aclrtMemcpy(host_recv.data(), recv_bytes, recv_buffer, recv_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy(recv D2H)"
    );

    int local_ok = 1;
    for (int source_rank = 0; source_rank < world_size && local_ok != 0; ++source_rank) {
        for (uint64_t i = 0; i < count; ++i) {
            const int32_t actual = host_recv[static_cast<uint64_t>(source_rank) * count + i];
            if (actual != source_rank) {
                std::cerr << "[rank " << rank << "] mismatch source_rank=" << source_rank << " index=" << i
                          << " actual=" << actual << " expected=" << source_rank << std::endl;
                local_ok = 0;
                break;
            }
        }
    }
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (local_ok != 0) {
        std::cout << "[rank " << rank << "] HCCL RootInfo AllGather verify OK" << std::endl;
    }

    int cleanup_ok = 1;
    if (aclrtFree(recv_buffer) != ACL_SUCCESS) cleanup_ok = 0;
    if (aclrtFree(send_buffer) != ACL_SUCCESS) cleanup_ok = 0;
    if (aclrtDestroyStream(stream) != ACL_SUCCESS) cleanup_ok = 0;
    if (HcclCommDestroy(comm) != HCCL_SUCCESS) cleanup_ok = 0;
    if (aclrtResetDevice(local_rank) != ACL_SUCCESS) cleanup_ok = 0;
    if (aclFinalize() != ACL_SUCCESS) cleanup_ok = 0;

    int global_cleanup_ok = 0;
    MPI_Allreduce(&cleanup_ok, &global_cleanup_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        if (global_ok != 0 && global_cleanup_ok != 0) {
            std::cout << "A3 HCCL RootInfo AllGather smoke PASS" << std::endl;
        } else {
            std::cerr << "A3 HCCL RootInfo AllGather smoke FAIL" << std::endl;
        }
    }

    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return global_ok != 0 && global_cleanup_ok != 0 ? 0 : 1;
}

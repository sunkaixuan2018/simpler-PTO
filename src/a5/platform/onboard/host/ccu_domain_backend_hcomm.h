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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hccl/hccl_comm.h"
#include "platform_comm/ccu/ccu_domain_owner.h"

namespace simpler::ccu {

struct HcommPreparedMission {
    uint32_t mission_id{};
    uint32_t kernel_id{};
    uint64_t program_handle{};
    uint64_t trigger_event_handle{};
    uint64_t stop_event_handle{};
    uint64_t trigger_gate_address{};
    uint64_t stop_gate_address{};
    uint64_t stream_handle{};
};

struct HcommDomainPreparation {
    std::unique_ptr<DomainCcuOwner> owner;
    std::vector<HcommPreparedMission> missions;
};

HcommDomainPreparation prepare_hcomm_ccu_domain(
    HcclComm base_communicator, uint64_t domain_id, const std::vector<uint32_t> &rank_ids, uint32_t domain_rank,
    uint32_t group_id, const ExpectedBuildIdentity &expected_identity,
    const std::vector<std::vector<uint8_t>> &module_bytes, const std::string &program_namespace
);

}  // namespace simpler::ccu

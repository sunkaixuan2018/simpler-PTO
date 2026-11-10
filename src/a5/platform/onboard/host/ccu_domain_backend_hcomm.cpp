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

#include "ccu_domain_backend_hcomm.h"

#include <array>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <utility>

#include "acl/acl.h"
#include "hccl/hccl_ccu_res.h"
#include "hccl/hccl_channel.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hcomm/ccu/ccu_launch.h"
#include "hcomm/ccu/ccu_primitives.hpp"
#include "hcomm/ccu/ccu_res.h"

#include "ccu_domain_registration_hcomm.h"
#include "ccu_run_lifecycle_hcomm.h"

namespace simpler::ccu {
namespace {

[[noreturn]] void fail(const char *operation, int code) {
    throw std::runtime_error(std::string(operation) + " failed with code " + std::to_string(code));
}

void check(int code, const char *operation) {
    if (code != 0) fail(operation, code);
}

CcuResult probe_channel_die(const ChannelHandle *channel) { return AscendC::ccu::NotifyWait(*channel, 0); }

std::string sibling_trigger_binary() {
    Dl_info info{};
    if (::dladdr(reinterpret_cast<const void *>(&prepare_hcomm_ccu_domain), &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("cannot locate the A5 host runtime for the CCU trigger binary");
    std::string path(info.dli_fname);
    const size_t slash = path.find_last_of('/');
    return (slash == std::string::npos ? std::string(".") : path.substr(0, slash)) + "/ccu_event_trigger.o";
}

ChannelHandle acquire_ring_channel(HcclComm communicator, uint32_t rank, uint32_t rank_count) {
    const uint32_t peer = (rank + 1) % rank_count;
    CommLink *links = nullptr;
    uint32_t link_count = 0;
    check(HcclRankGraphGetLinks(communicator, 0, rank, peer, &links, &link_count), "HcclRankGraphGetLinks");
    for (uint32_t i = 0; i < link_count; ++i) {
        if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) continue;
        HcclChannelDesc descriptor{};
        check(HcclChannelDescInit(&descriptor, 1), "HcclChannelDescInit");
        descriptor.remoteRank = peer;
        descriptor.notifyNum = 4;
        descriptor.channelProtocol = COMM_PROTOCOL_UBC_CTP;
        descriptor.localEndpoint = links[i].srcEndpointDesc;
        descriptor.remoteEndpoint = links[i].dstEndpointDesc;
        ChannelHandle channel{};
        check(HcclChannelAcquire(communicator, COMM_ENGINE_CCU, &descriptor, 1, &channel), "HcclChannelAcquire");
        if (channel == 0) throw std::runtime_error("HcclChannelAcquire returned a null channel");
        return channel;
    }
    throw std::runtime_error("the CCU ring peer has no UBC_CTP link");
}

uint32_t find_channel_die(ChannelHandle channel) {
    int32_t selected = -1;
    const void *arguments[] = {&channel};
    for (uint32_t die = 0; die < 2; ++die) {
        HcommCcuResDescHandle descriptor{};
        check(HcommCcuInsResDescCreate(die, &descriptor), "HcommCcuInsResDescCreate(probe)");
        const int query =
            HcommCcuKernelQueryResReq(reinterpret_cast<const void *>(probe_channel_die), arguments, 1, descriptor);
        check(HcommCcuInsResDescDestroy(descriptor), "HcommCcuInsResDescDestroy(probe)");
        if (query == CCU_SUCCESS) {
            if (selected >= 0) throw std::runtime_error("CCU channel maps to multiple IO dies");
            selected = static_cast<int32_t>(die);
        } else if (query != CCU_E_PARA && query != CCU_E_UNAVAIL) {
            fail("HcommCcuKernelQueryResReq(channel probe)", query);
        }
    }
    if (selected < 0) throw std::runtime_error("CCU channel has no usable IO die");
    return static_cast<uint32_t>(selected);
}

SimplerCcuRegistrationBindingV1 binding_for(
    const NativeKernelDefinition &kernel, uint32_t rank, uint32_t rank_count, ChannelHandle channel,
    CcuEventHandle trigger, CcuEventHandle stop
) {
    return {
        SIMPLER_CCU_MODULE_ABI_VERSION,
        sizeof(SimplerCcuRegistrationBindingV1),
        SIMPLER_CCU_REGISTRATION_BINDING_LAYOUT_VERSION,
        kernel.record.kernel_id,
        kernel.record.group_id,
        rank,
        rank_count,
        0,
        channel,
        trigger,
        stop,
        {0, 0, 0, 0, 0}
    };
}

class HcommDomainResourceBackend final : public DomainResourceBackend {
public:
    HcclComm communicator{};
    CcuInsHandle unassigned_instance{};
    bool instance_assigned{false};
    ChannelHandle channel{};
    std::vector<HcommMissionStream> streams;
    aclrtBinHandle trigger_binary{};
    aclrtFuncHandle trigger_function{};
    aclrtStream trigger_stream{};

    bool drained() const override {
        if (closed_) return false;
        for (const auto &mission : streams) {
            if (mission.stream == nullptr) return false;
            aclrtStreamStatus state{};
            if (aclrtStreamQuery(mission.stream, &state) != ACL_SUCCESS || state != ACL_STREAM_STATUS_COMPLETE)
                return false;
        }
        if (trigger_stream != nullptr) {
            aclrtStreamStatus state{};
            if (aclrtStreamQuery(trigger_stream, &state) != ACL_SUCCESS || state != ACL_STREAM_STATUS_COMPLETE)
                return false;
        }
        return true;
    }

    bool close() override {
        if (closed_) return true;
        if (!drained()) return false;
        if (!instance_assigned && unassigned_instance != 0) {
            if (HcommCcuInsDestroy(unassigned_instance) != CCU_SUCCESS) return false;
            unassigned_instance = 0;
        }
        // The communicator owns an assigned instance, the channel and acquired CCU threads.
        if (communicator != nullptr) {
            if (HcclCommDestroy(communicator) != HCCL_SUCCESS) return false;
            communicator = nullptr;
            instance_assigned = false;
            unassigned_instance = 0;
            channel = 0;
        }
        for (auto it = streams.rbegin(); it != streams.rend(); ++it) {
            if (it->stream != nullptr && aclrtDestroyStream(it->stream) != ACL_SUCCESS) return false;
            it->stream = nullptr;
            it->thread = 0;
        }
        streams.clear();
        if (trigger_stream != nullptr && aclrtDestroyStream(trigger_stream) != ACL_SUCCESS) return false;
        trigger_stream = nullptr;
        trigger_function = nullptr;
        if (trigger_binary != nullptr && aclrtBinaryUnLoad(trigger_binary) != ACL_SUCCESS) return false;
        trigger_binary = nullptr;
        closed_ = true;
        return true;
    }

private:
    bool closed_{false};
};

}  // namespace

HcommDomainPreparation prepare_hcomm_ccu_domain(
    HcclComm base_communicator, uint64_t domain_id, const std::vector<uint32_t> &rank_ids, uint32_t domain_rank,
    uint32_t group_id, const ExpectedBuildIdentity &expected_identity,
    const std::vector<std::vector<uint8_t>> &module_bytes, const std::string &program_namespace
) {
    if (base_communicator == nullptr || domain_id == 0 || rank_ids.size() < 2 || domain_rank >= rank_ids.size() ||
        group_id == SIMPLER_CCU_INVALID_GROUP_ID || module_bytes.empty() || program_namespace.empty())
        throw std::invalid_argument("invalid A5 CCU domain preparation request");

    static NativeModuleCache module_cache;
    std::vector<std::shared_ptr<NativeModule>> modules;
    std::vector<NativeKernelDefinition> kernels;
    for (const auto &bytes : module_bytes) {
        auto module = module_cache.load(bytes, expected_identity);
        for (const auto &kernel : module->kernels())
            if (kernel.record.group_id == group_id) kernels.push_back(kernel);
        modules.push_back(std::move(module));
    }
    if (kernels.empty()) throw std::invalid_argument("the callable has no CCU kernel in the requested group");

    auto backend = std::make_unique<HcommDomainResourceBackend>();
    auto rollback = [&] {
        if (!backend->close()) throw std::runtime_error("A5 CCU prepare rollback could not confirm resource release");
    };
    try {
        HcclCommConfig config{};
        HcclCommConfigInit(&config);
        check(
            HcclCreateSubCommConfig(
                &base_communicator, static_cast<uint32_t>(rank_ids.size()), rank_ids.data(), domain_id, domain_rank,
                &config, &backend->communicator
            ),
            "HcclCreateSubCommConfig(CCU)"
        );
        backend->channel = acquire_ring_channel(backend->communicator, domain_rank, rank_ids.size());
        const uint32_t die = find_channel_die(backend->channel);

        CcuInsHandle staging{};
        check(HcommCcuInsCreateDefault(nullptr, 0, &staging), "HcommCcuInsCreateDefault(staging)");
        std::vector<std::pair<CcuEventHandle, CcuEventHandle>> staging_events;
        staging_events.reserve(kernels.size());
        try {
            HcommDomainRegistrationSdk sizing(staging, die);
            for (const auto &kernel : kernels) {
                CcuEventHandle trigger{}, stop{};
                check(HcommCcuEventAlloc(staging, die, 1, &trigger), "HcommCcuEventAlloc(staging trigger)");
                check(HcommCcuEventAlloc(staging, die, 1, &stop), "HcommCcuEventAlloc(staging stop)");
                staging_events.emplace_back(trigger, stop);
                const auto binding = binding_for(
                    kernel, domain_rank, static_cast<uint32_t>(rank_ids.size()), backend->channel, trigger, stop
                );
                const void *arguments[] = {&binding};
                check(
                    sizing.query_resources(kernel.entry, arguments, SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT),
                    "HcommCcuKernelQueryResReq(sizing)"
                );
            }

            HcommCcuResDescHandle descriptor{};
            check(HcommCcuInsResDescCreate(die, &descriptor), "HcommCcuInsResDescCreate(sized)");
            try {
                for (size_t type = 0; type < sizing.required_resources().size(); ++type) {
                    uint64_t count = sizing.required_resources()[type];
                    if (type == HCOMM_CCU_RES_TYPE_EVENT) count += kernels.size() * 2;
                    if (count > std::numeric_limits<uint32_t>::max())
                        throw std::overflow_error("CCU resource requirement exceeds SDK width");
                    check(
                        HcommCcuInsResDescSetNum(
                            descriptor, static_cast<HcommCcuResType>(type), static_cast<uint32_t>(count)
                        ),
                        "HcommCcuInsResDescSetNum"
                    );
                }
                check(HcommCcuInsDestroy(staging), "HcommCcuInsDestroy(staging)");
                staging = 0;
                const HcommCcuResDescHandle descriptors[] = {descriptor};
                check(HcommCcuInsCreate(descriptors, 1, &backend->unassigned_instance), "HcommCcuInsCreate(sized)");
            } catch (...) {
                (void)HcommCcuInsResDescDestroy(descriptor);
                throw;
            }
            check(HcommCcuInsResDescDestroy(descriptor), "HcommCcuInsResDescDestroy(sized)");
        } catch (...) {
            if (staging != 0) (void)HcommCcuInsDestroy(staging);
            throw;
        }

        DomainRegistrationRequest request{};
        request.domain_id = domain_id;
        request.group_id = group_id;
        request.rank_id = domain_rank;
        request.rank_count = static_cast<uint32_t>(rank_ids.size());
        request.channel_handle = backend->channel;
        request.program_namespace = program_namespace;
        std::vector<HcommPreparedMission> missions;
        missions.reserve(kernels.size());
        for (const auto &kernel : kernels) {
            CcuEventHandle trigger{}, stop{};
            check(HcommCcuEventAlloc(backend->unassigned_instance, die, 1, &trigger), "HcommCcuEventAlloc(trigger)");
            check(HcommCcuEventAlloc(backend->unassigned_instance, die, 1, &stop), "HcommCcuEventAlloc(stop)");
            uint64_t trigger_address{}, stop_address{};
            check(HcommCcuEventGetAddr(trigger, 0, &trigger_address), "HcommCcuEventGetAddr(trigger)");
            check(HcommCcuEventGetAddr(stop, 0, &stop_address), "HcommCcuEventGetAddr(stop)");
            request.missions.push_back({kernel.record.kernel_id, trigger, stop});
            missions.push_back(
                {kernel.record.kernel_id, kernel.record.kernel_id, 0, trigger, stop, trigger_address, stop_address, 0}
            );
        }

        HcommDomainRegistrationSdk registration(backend->unassigned_instance, die);
        const auto registered = register_domain_kernels(kernels, request, registration);
        for (auto &mission : missions) {
            for (const auto &kernel : registered)
                if (kernel.kernel_id == mission.kernel_id) mission.program_handle = kernel.program_handle;
            if (mission.program_handle == 0) throw std::runtime_error("missing registered CCU mission program");
            HcommMissionStream stream{};
            stream.handle = static_cast<uint64_t>(backend->streams.size() + 1);
            check(aclrtCreateStream(&stream.stream), "aclrtCreateStream(CCU mission)");
            backend->streams.push_back(stream);
            check(
                HcclThreadAcquireWithStream(
                    backend->communicator, COMM_ENGINE_CCU, backend->streams.back().stream, 1,
                    &backend->streams.back().thread
                ),
                "HcclThreadAcquireWithStream(CCU mission)"
            );
            mission.stream_handle = stream.handle;
        }

        check(aclrtCreateStream(&backend->trigger_stream), "aclrtCreateStream(CCU trigger)");
        const std::string binary_path = sibling_trigger_binary();
        check(
            aclrtBinaryLoadFromFile(binary_path.c_str(), nullptr, &backend->trigger_binary),
            "aclrtBinaryLoadFromFile(CCU trigger)"
        );
        check(
            aclrtBinaryGetFunction(backend->trigger_binary, "simpler_ccu_event_trigger", &backend->trigger_function),
            "aclrtBinaryGetFunction(CCU trigger)"
        );
        check(HcclCommAssignCcuIns(backend->communicator, backend->unassigned_instance), "HcclCommAssignCcuIns");
        backend->instance_assigned = true;
        CcuInsHandle assigned{};
        uint32_t assigned_count{};
        check(
            HcclCommQueryAssignedCcuIns(backend->communicator, &assigned, &assigned_count),
            "HcclCommQueryAssignedCcuIns"
        );
        if (assigned_count != 1 || assigned != backend->unassigned_instance)
            throw std::runtime_error("assigned CCU instance does not match the prepared instance");

        auto owner = std::make_unique<DomainCcuOwner>(std::move(backend));
        for (auto &module : modules)
            owner->retain_module(std::move(module));
        owner->set_registered_kernels(registered);
        return {std::move(owner), std::move(missions)};
    } catch (...) {
        rollback();
        throw;
    }
}

}  // namespace simpler::ccu

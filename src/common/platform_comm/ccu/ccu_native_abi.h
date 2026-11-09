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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <type_traits>
extern "C" {
#endif

#define SIMPLER_CCU_MODULE_ABI_VERSION 1U
#define SIMPLER_CCU_REGISTRATION_BINDING_LAYOUT_VERSION 1U
#define SIMPLER_CCU_TASK_ARGUMENT_LAYOUT_VERSION 1U
#define SIMPLER_CCU_MODULE_MANIFEST_SYMBOL "simpler_ccu_module_manifest_v1"
#define SIMPLER_CCU_BUILD_ID_SIZE 32U
#define SIMPLER_CCU_COMPLETION_SLOT_ALIGNMENT 64U
#define SIMPLER_CCU_INVALID_GROUP_ID UINT32_MAX
#define SIMPLER_CCU_REGISTRATION_ARGUMENT_COUNT 1U
#define SIMPLER_CCU_TASK_ARGUMENT_COUNT 10U

#define SIMPLER_CCU_KERNEL_CAP_COMPLETION_SLOT (1U << 0)
#define SIMPLER_CCU_KERNEL_CAP_STOP_GATE (1U << 1)
#define SIMPLER_CCU_KERNEL_CAP_MASK (SIMPLER_CCU_KERNEL_CAP_COMPLETION_SLOT | SIMPLER_CCU_KERNEL_CAP_STOP_GATE)

#define SIMPLER_CCU_RUN_STATE_INIT 0U
#define SIMPLER_CCU_RUN_STATE_LAUNCHED 1U
#define SIMPLER_CCU_RUN_STATE_READY 2U
#define SIMPLER_CCU_RUN_STATE_EXITING 3U
#define SIMPLER_CCU_RUN_STATE_EXITED 4U
#define SIMPLER_CCU_RUN_STATE_FAILED 5U

#define SIMPLER_CCU_COMPLETION_PENDING 0U
#define SIMPLER_CCU_COMPLETION_SUCCEEDED 1U
#define SIMPLER_CCU_COMPLETION_FAILED 2U

typedef struct SimplerCcuModuleManifestV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t manifest_size;
    uint32_t kernel_record_size;
    uint32_t kernel_count;
    uint32_t kernel_records_offset;
    uint32_t string_table_offset;
    uint32_t string_table_size;
    uint32_t capability_flags;
    uint32_t reserved0;
    uint8_t sdk_build_id[SIMPLER_CCU_BUILD_ID_SIZE];
    uint8_t pto_isa_build_id[SIMPLER_CCU_BUILD_ID_SIZE];
    uint64_t reserved[4];
} SimplerCcuModuleManifestV1;

typedef struct SimplerCcuKernelRecordV1 {
    uint32_t struct_size;
    uint32_t kernel_id;
    uint32_t group_id;
    uint32_t task_argument_layout_version;
    uint64_t parameter_size;
    uint32_t parameter_alignment;
    uint32_t capability_flags;
    uint32_t entry_symbol_offset;
    uint32_t entry_symbol_size;
    uint32_t registration_binding_layout_version;
    uint32_t registration_binding_size;
    uint32_t registration_argument_count;
    uint32_t task_argument_count;
    uint32_t reserved[2];
} SimplerCcuKernelRecordV1;

typedef struct SimplerCcuRegistrationBindingV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t layout_version;
    uint32_t kernel_id;
    uint32_t group_id;
    uint32_t rank_id;
    uint32_t rank_count;
    uint32_t flags;
    uint64_t channel_handle;
    uint64_t trigger_event_handle;
    uint64_t stop_event_handle;
    uint64_t reserved[5];
} SimplerCcuRegistrationBindingV1;

typedef struct SimplerCcuLaunchArgumentsV1 {
    uint64_t layout_version;
    uint64_t run_generation;
    uint64_t mission_id;
    uint64_t run_control_addr;
    uint64_t run_control_token;
    uint64_t parameter_addr;
    uint64_t parameter_token;
    uint64_t completion_slot_addr;
    uint64_t completion_slot_token;
    uint64_t reserved;
} SimplerCcuLaunchArgumentsV1;

typedef struct SimplerCcuRunControlV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t run_generation;
    uint32_t mission_id;
    uint32_t state;
    uint32_t stop_requested;
    uint32_t flags;
    uint64_t stop_gate_addr;
    uint64_t final_wake_value;
    uint64_t last_completed_sequence;
    int64_t native_error;
    uint64_t reserved[4];
} SimplerCcuRunControlV1;

typedef struct SimplerCcuMissionDescriptorV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t run_generation;
    uint32_t mission_id;
    uint32_t kernel_id;
    uint32_t group_id;
    uint32_t flags;
    uint64_t parameter_addr;
    uint64_t parameter_size;
    uint64_t run_control_addr;
    uint64_t completion_slot_addr;
    uint64_t trigger_gate_addr;
    uint64_t stop_gate_addr;
    uint64_t run_control_token;
    uint64_t parameter_token;
    uint64_t completion_slot_token;
    uint64_t reserved;
} SimplerCcuMissionDescriptorV1;

typedef struct SimplerCcuCompletionSlotV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t mission_id;
    uint32_t status;
    uint64_t run_generation;
    uint64_t operation_sequence;
    int64_t native_error;
    uint64_t reserved[4];
    uint64_t completed_sequence;
} SimplerCcuCompletionSlotV1;

typedef const SimplerCcuModuleManifestV1 *(*SimplerCcuModuleManifestEntryV1)(void);
typedef int32_t (*SimplerCcuNativeKernelEntryV1)(const SimplerCcuRegistrationBindingV1 *binding);

#ifdef __cplusplus
}

#define SIMPLER_CCU_ASSERT_POD(type) \
    static_assert(std::is_trivially_copyable_v<type> && std::is_standard_layout_v<type>, #type " must be C/POD")

SIMPLER_CCU_ASSERT_POD(SimplerCcuModuleManifestV1);
SIMPLER_CCU_ASSERT_POD(SimplerCcuKernelRecordV1);
SIMPLER_CCU_ASSERT_POD(SimplerCcuRegistrationBindingV1);
SIMPLER_CCU_ASSERT_POD(SimplerCcuLaunchArgumentsV1);
SIMPLER_CCU_ASSERT_POD(SimplerCcuRunControlV1);
SIMPLER_CCU_ASSERT_POD(SimplerCcuMissionDescriptorV1);
SIMPLER_CCU_ASSERT_POD(SimplerCcuCompletionSlotV1);

static_assert(sizeof(SimplerCcuModuleManifestV1) == 136, "CCU manifest ABI size changed");
static_assert(sizeof(SimplerCcuKernelRecordV1) == 64, "CCU kernel record ABI size changed");
static_assert(sizeof(SimplerCcuRegistrationBindingV1) == 96, "CCU registration binding ABI size changed");
static_assert(sizeof(SimplerCcuLaunchArgumentsV1) == 80, "CCU launch arguments ABI size changed");
static_assert(sizeof(SimplerCcuRunControlV1) == 96, "CCU run-control ABI size changed");
static_assert(sizeof(SimplerCcuMissionDescriptorV1) == 112, "CCU mission descriptor ABI size changed");
static_assert(sizeof(SimplerCcuCompletionSlotV1) == 80, "CCU completion-slot ABI size changed");
static_assert(
    offsetof(SimplerCcuCompletionSlotV1, completed_sequence) == 72,
    "completed_sequence must remain the final published completion field"
);

#undef SIMPLER_CCU_ASSERT_POD
#endif

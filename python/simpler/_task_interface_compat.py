# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compatibility imports for the _task_interface extension module."""

import _task_interface as _binding  # pyright: ignore[reportMissingImports]


def _require(name: str):
    return getattr(_binding, name)


CHIP_BOOTSTRAP_MAILBOX_SIZE = _require("CHIP_BOOTSTRAP_MAILBOX_SIZE")
CONTINUOUS_TENSOR_MAX_DIMS = _require("CONTINUOUS_TENSOR_MAX_DIMS")
MAILBOX_ERROR_MSG_SIZE = _require("MAILBOX_ERROR_MSG_SIZE")
MAILBOX_OFF_ERROR_MSG = _require("MAILBOX_OFF_ERROR_MSG")
MAILBOX_SIZE = _require("MAILBOX_SIZE")
ArgDirection = _require("ArgDirection")
ChipBootstrapChannel = _require("ChipBootstrapChannel")
ChipBootstrapMailboxState = _require("ChipBootstrapMailboxState")
ChipCallable = _require("ChipCallable")
ChipStorageTaskArgs = _require("ChipStorageTaskArgs")
ContinuousTensor = _require("ContinuousTensor")
CoreCallable = _require("CoreCallable")
DataType = _require("DataType")
SubmitResult = _require("SubmitResult")
TaskArgs = _require("TaskArgs")
TaskState = _require("TaskState")
TensorArgType = _require("TensorArgType")
WorkerType = _require("WorkerType")
_ChipWorker = _require("_ChipWorker")
_Orchestrator = _require("_Orchestrator")
_Worker = _require("_Worker")
arg_direction_name = _require("arg_direction_name")
get_dtype_name = _require("get_dtype_name")
get_element_size = _require("get_element_size")
read_args_from_blob = _require("read_args_from_blob")

# #687 renamed ChipCallConfig to CallConfig. Keep Python imports working when
# the package code is newer than the installed extension in an existing venv.
CallConfig = getattr(_binding, "CallConfig", None)
if CallConfig is None:
    CallConfig = _require("ChipCallConfig")

__all__ = [
    "CHIP_BOOTSTRAP_MAILBOX_SIZE",
    "CONTINUOUS_TENSOR_MAX_DIMS",
    "MAILBOX_ERROR_MSG_SIZE",
    "MAILBOX_OFF_ERROR_MSG",
    "MAILBOX_SIZE",
    "ArgDirection",
    "CallConfig",
    "ChipBootstrapChannel",
    "ChipBootstrapMailboxState",
    "ChipCallable",
    "ChipStorageTaskArgs",
    "ContinuousTensor",
    "CoreCallable",
    "DataType",
    "SubmitResult",
    "TaskArgs",
    "TaskState",
    "TensorArgType",
    "WorkerType",
    "_ChipWorker",
    "_Orchestrator",
    "_Worker",
    "arg_direction_name",
    "get_dtype_name",
    "get_element_size",
    "read_args_from_blob",
]

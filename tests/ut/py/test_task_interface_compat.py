# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compatibility tests for Python wrappers around _task_interface."""

import importlib
import sys
import types
import unittest


_MISSING = object()


def _load_compat_with_binding(binding):
    previous_binding = sys.modules.get("_task_interface", _MISSING)
    previous_compat = sys.modules.pop("simpler._task_interface_compat", _MISSING)
    sys.modules["_task_interface"] = binding
    try:
        return importlib.import_module("simpler._task_interface_compat")
    finally:
        sys.modules.pop("simpler._task_interface_compat", None)
        if previous_compat is not _MISSING:
            sys.modules["simpler._task_interface_compat"] = previous_compat
        if previous_binding is _MISSING:
            sys.modules.pop("_task_interface", None)
        else:
            sys.modules["_task_interface"] = previous_binding


def _fake_binding(**overrides):
    binding = types.ModuleType("_task_interface")
    names = [
        "CHIP_BOOTSTRAP_MAILBOX_SIZE",
        "CONTINUOUS_TENSOR_MAX_DIMS",
        "MAILBOX_ERROR_MSG_SIZE",
        "MAILBOX_OFF_ERROR_MSG",
        "MAILBOX_SIZE",
        "ArgDirection",
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
    for name in names:
        setattr(binding, name, object())
    for name, value in overrides.items():
        setattr(binding, name, value)
    return binding


class TestTaskInterfaceCompat(unittest.TestCase):
    def test_call_config_falls_back_to_legacy_chip_call_config(self):
        class LegacyChipCallConfig:
            pass

        compat = _load_compat_with_binding(_fake_binding(ChipCallConfig=LegacyChipCallConfig))

        self.assertIs(compat.CallConfig, LegacyChipCallConfig)

    def test_call_config_prefers_new_binding_name(self):
        class NewCallConfig:
            pass

        class LegacyChipCallConfig:
            pass

        compat = _load_compat_with_binding(
            _fake_binding(CallConfig=NewCallConfig, ChipCallConfig=LegacyChipCallConfig)
        )

        self.assertIs(compat.CallConfig, NewCallConfig)

    def test_task_interface_import_uses_legacy_binding_fallback(self):
        class LegacyChipCallConfig:
            pass

        previous_binding = sys.modules.get("_task_interface", _MISSING)
        previous_compat = sys.modules.pop("simpler._task_interface_compat", _MISSING)
        previous_task_interface = sys.modules.pop("simpler.task_interface", _MISSING)
        sys.modules["_task_interface"] = _fake_binding(ChipCallConfig=LegacyChipCallConfig)
        try:
            task_interface = importlib.import_module("simpler.task_interface")
        finally:
            sys.modules.pop("simpler.task_interface", None)
            sys.modules.pop("simpler._task_interface_compat", None)
            if previous_task_interface is not _MISSING:
                sys.modules["simpler.task_interface"] = previous_task_interface
            if previous_compat is not _MISSING:
                sys.modules["simpler._task_interface_compat"] = previous_compat
            if previous_binding is _MISSING:
                sys.modules.pop("_task_interface", None)
            else:
                sys.modules["_task_interface"] = previous_binding

        self.assertIs(task_interface.CallConfig, LegacyChipCallConfig)


if __name__ == "__main__":
    unittest.main()

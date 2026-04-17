# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Golden script for the host_register_mapped demo."""

import torch

__outputs__ = ["mapped_out"]
TENSOR_ORDER = ["mapped_out"]

RTOL = 0.0
ATOL = 0.0
SIZE = 16


def generate_inputs(params: dict) -> dict:
    del params
    return {
        "mapped_out": torch.zeros(SIZE, dtype=torch.int64),
    }


def compute_golden(tensors: dict, params: dict) -> None:
    del params
    tensors["mapped_out"][:] = torch.arange(1, SIZE + 1, dtype=torch.int64)

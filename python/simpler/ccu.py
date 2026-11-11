# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Internal CCU registration and execution ownership for a chip process."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path


def runtime_build_identity() -> tuple[bytes, bytes]:
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415 -- CCU-only toolchain discovery

    sdk_header = Path(os.environ["ASCEND_HOME_PATH"]) / "aarch64-linux/include/hcomm/ccu/ccu_res.h"
    isa_header = Path(ensure_pto_isa_root()) / "include/pto/pto-inst.hpp"
    return hashlib.sha256(sdk_header.read_bytes()).digest(), hashlib.sha256(isa_header.read_bytes()).digest()


class ChipCcuBinding:
    """Retain one domain registration and serialize its L2 executions."""

    def __init__(self, worker, communicator: int, allocation: int, timeout: float):
        self.worker = worker
        self.communicator = communicator
        self.allocation = allocation
        self.timeout = timeout
        self.generation = 0
        self.active = False

    def _control(self, operation):
        return self.worker.control_domain_ccu(self.communicator, self.allocation, operation, self.generation)

    def _wait(self, operation):
        deadline = time.monotonic() + self.timeout
        while self._control(operation):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"CCU domain {self.allocation} operation {operation} timed out")

    def begin(self):
        if self.active:
            raise RuntimeError("overlapping L2 runs on one CCU domain are unsupported")
        self.generation += 1
        self.active = True
        try:
            self._control(1)
            self._wait(2)
        except BaseException:
            self.abort()
            raise

    def finish(self):
        try:
            self._wait(3)
        except BaseException:
            self.abort()
            raise
        self.active = False

    def abort(self):
        # Keep active set until a confirmed stop; a successful abort call alone is insufficient.
        try:
            self._control(4)
        except Exception:
            pass

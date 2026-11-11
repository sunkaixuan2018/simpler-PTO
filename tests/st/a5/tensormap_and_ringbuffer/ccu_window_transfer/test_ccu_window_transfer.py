# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CCU writes real allocated W bytes to the peer, with dynamic ranges and tokens."""

from pathlib import Path

import pytest
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from tests.st.a5.tensormap_and_ringbuffer.ccu_native_domain.test_ccu_native_domain import _build_callable


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime("tensormap_and_ringbuffer")
@pytest.mark.device_count(2)
def test_ccu_real_window_transfer(st_platform, st_device_ids, tmp_path, request):
    rounds = request.config.getoption("--rounds")
    assert rounds > 0
    rounds = max(2, rounds)
    target = _build_callable(tmp_path, source_root=Path(__file__).resolve().parent)
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=[int(d) for d in st_device_ids[:2]],
        num_sub_workers=0,
    )
    chip = worker.register(target)
    retained = []
    allocations = []
    try:
        worker.init()
        outputs = [[worker.create_buffer(64), worker.create_buffer(64), worker.create_buffer(32768)] for _ in range(2)]
        for generation in range(1, rounds + 1):
            payloads = [
                bytes((i * 17 + rank * 73 + generation * 11) % 256 for i in range(16384)) + bytes([0xA5]) * 16384
                for rank in range(2)
            ]

            def run(orch, _args, config):
                domain = orch.get_or_allocate_domain(
                    name="ccu_w",
                    members=[0, 1],
                    window_size=4 * 1024 * 1024,
                    buffers=[CommBufferSpec("data", "uint64", 4096, 32768)],
                    workspace_engines=("ccu",),
                    retain_after_run=True,
                )
                if retained:
                    assert retained[0] is domain
                else:
                    retained.append(domain)
                    allocations.append(domain.allocation_id)
                for rank in range(2):
                    orch.copy_to_domain(domain, rank, payloads[rank], buffer="data")
                for rank in range(2):
                    args = TaskArgs()
                    for index, output in enumerate(outputs[rank]):
                        args.add_tensor(
                            output.tensor((4096 if index == 2 else 8,), DataType.UINT64), TensorArgType.OUTPUT_EXISTING
                        )
                    args.add_scalar(domain.device_context(rank, workspace_engines=("ccu",)))
                    args.add_scalar(generation)
                    orch.submit_next_level(chip, args, config, worker=rank)

            worker.run(run)
            displacement = (generation % 8) * 64
            length = ((generation % 16) + 1) * 64
            for rank in range(2):
                expected = bytearray(payloads[rank])
                for ccu_id in range(2):
                    src = ccu_id * 8192 + displacement
                    dst = 16384 + src
                    expected[dst : dst + length] = payloads[1 - rank][src : src + length]
                assert bytes(outputs[rank][2].shm.buf) == expected, f"W mismatch round={generation} rank={rank}"
            if generation in (1, rounds) or generation % 100 == 0:
                print(f"CCU W round={generation}/{rounds}: bidirectional bytes and untouched ranges passed", flush=True)
            if generation == rounds // 2:
                retained.pop().release()
        retained.pop().release()
        assert len(allocations) == 2 and allocations[0] != allocations[1]
        print("CCU W retained reuse, release and fresh window/token allocation passed", flush=True)
    finally:
        worker.close()

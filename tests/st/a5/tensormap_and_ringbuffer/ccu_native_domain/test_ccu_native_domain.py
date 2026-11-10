# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real native modules, Worker registration, READY, AIV/TMR and confirmed stop."""

import hashlib
import os
import struct
import subprocess
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipCallable, CoreCallable, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_ROOT = Path(__file__).resolve().parent
_RUNTIME = "tensormap_and_ringbuffer"


def _build_callable(tmp_path):
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    arch = sdk / "aarch64-linux"
    common = Path(__file__).resolve().parents[5] / "src/common"
    sdk_identity = hashlib.sha256((arch / "include/hcomm/ccu/ccu_res.h").read_bytes()).digest()
    isa_root = Path(ensure_pto_isa_root())
    isa_identity = hashlib.sha256((isa_root / "include/pto/pto-inst.hpp").read_bytes()).digest()
    identities = tmp_path / "identities.cpp"
    identities.write_text(
        "#include <cstdint>\n"
        f"extern const uint8_t ccu_test_sdk_identity[32] = {{{','.join(map(str, sdk_identity))}}};\n"
        f"extern const uint8_t ccu_test_isa_identity[32] = {{{','.join(map(str, isa_identity))}}};\n"
    )
    module = tmp_path / "native_module.so"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            str(_ROOT / "native_module.cpp"),
            str(identities),
            f"-I{common}",
            f"-I{arch / 'include'}",
            f"-I{arch / 'pkg_inc/hcomm/ccu'}",
            f"-I{arch / 'include/hcomm/ccu'}",
            f"-L{sdk / 'lib64'}",
            "-lasccomm_ccu",
            "-lhcomm",
            "-o",
            str(module),
        ],
        check=True,
    )
    compiler = KernelCompiler(platform="a5")
    orchestration = compiler.compile_orchestration(_RUNTIME, str(_ROOT / "kernels/orchestration/ccu_orch.cpp"))
    signatures = [[D.OUT, D.IN, D.IN, D.IN], [D.IN, D.OUT, D.IN, D.IN, D.IN], [D.IN, D.IN, D.OUT, D.IN, D.IN]]
    children = []
    for index, name in enumerate(("trigger_b", "trigger_a", "verify")):
        binary = compiler.compile_incore(
            str(_ROOT / f"kernels/aiv/{name}.cpp"),
            core_type="aiv",
            pto_isa_root=str(isa_root),
            extra_include_dirs=compiler.get_orchestration_include_dirs(_RUNTIME),
        )
        children.append((index, CoreCallable.build(signature=signatures[index], binary=extract_text_section(binary))))
    return (
        ChipCallable.build(
            signature=[D.OUT, D.OUT, D.OUT, D.IN, D.IN],
            func_name="ccu_orchestration",
            config_name="ccu_orchestration_config",
            binary=orchestration,
            children=children,
            ccu_modules=[module.read_bytes()],
        ),
        sdk_identity,
        isa_identity,
    )


@pytest.mark.platforms(["a5"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
def test_two_missions_b_before_a(st_platform, st_device_ids, tmp_path, request):
    rounds = request.config.getoption("--ccu-rounds")
    assert rounds > 0
    target, sdk_id, isa_id = _build_callable(tmp_path)
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=[int(device) for device in st_device_ids[:2]],
        num_sub_workers=0,
    )
    callable_handle = worker.register(target)
    retained = []
    try:
        worker.init()
        outputs = [[worker.create_buffer(64) for _ in range(3)] for _ in range(2)]

        def run_round(orch, _args, config):
            domain = orch.get_or_allocate_domain(
                name="native_ccu",
                members=[0, 1],
                window_size=4 * 1024 * 1024,
                retain_after_run=True,
                ccu_callable=callable_handle,
                ccu_group_id=7,
                ccu_sdk_build_id=sdk_id,
                ccu_pto_isa_build_id=isa_id,
            )
            if retained:
                assert domain is retained[0]
            else:
                retained.append(domain)
            for rank in range(2):
                args = TaskArgs()
                for output in outputs[rank]:
                    args.add_tensor(output.tensor((8,), DataType.UINT64), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(domain.device_context(rank))
                args.add_scalar(round_index * 100 + rank * 10)
                orch.submit_next_level(callable_handle, args, config, worker=rank)

        for round_index in range(1, rounds + 1):
            worker.run(run_round)
            for rank in range(2):
                value = round_index * 100 + rank * 10
                assert [struct.unpack_from("<Q", output.shm.buf)[0] for output in outputs[rank]] == [
                    value + 2,
                    value + 1,
                    value,
                ]
            if round_index in (1, rounds) or round_index % 100 == 0:
                print(
                    f"CCU round={round_index}/{rounds}: B-before-A, dynamic parameters and confirmed stop passed",
                    flush=True,
                )
        # No mission is triggered: normal stop must wake both EventWait loops.
        worker.run(lambda orch, _args, _cfg: orch.get_domain("native_ccu"))
        retained[0].release()
        assert retained[0].freed
    finally:
        worker.close()

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import contextlib
import ctypes
from types import SimpleNamespace
from typing import Any

import pytest
from simpler.buffer import mint_owner_instance_id, wrap_fork_inherited
from simpler.ccu_runtime import discover_ccu_domains, require_ccu_execution_backend
from simpler.comm_domain import DomainConfig
from simpler.orchestrator import Orchestrator
from simpler.task_interface import CcuFunction, ChipCallable, DataType, TaskArgs
from simpler.worker import Worker


def _domain(address=0x1000, *, worker=0, domain_id=1, engines=("ccu",), **changes) -> Any:
    domain = SimpleNamespace(
        name=f"domain_{domain_id}",
        contexts={worker: SimpleNamespace(device_ctx=address)},
        domain_id=domain_id,
        allocation_id=domain_id,
        generation=1,
        released=False,
        freed=False,
        _config=DomainConfig.capture([0, 1], 1024, [], True, (), engines),
    )
    for key, value in changes.items():
        setattr(domain, key, value)
    return domain


def _args(*values):
    args = TaskArgs()
    for value in values:
        args.add_scalar(value)
    return args


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim"])
@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_native_registration_refuses_ccu_without_provider(platform, runtime):
    worker = Worker(level=2, device_id=0, platform=platform, runtime=runtime)
    try:
        worker.init()
        native = worker._chip_worker
        assert native is not None
        target = _callable()
        with pytest.raises(RuntimeError, match="-1001"):
            native.register_callable(target)
        with pytest.raises(RuntimeError, match="-1001"):
            native.register_callable(0, target)
    finally:
        worker.close()


def _callable(*, ccu=True):
    binary = bytearray(64)
    binary[:7] = b"\x7fELF\x02\x01\x01"
    binary[16] = 3
    binary[20] = 1
    return ChipCallable.build(
        signature=[],
        func_name="orch",
        binary=b"orch",
        children=[],
        ccu_functions=[CcuFunction.build(bytes(binary), "copy")] if ccu else [],
    )


def test_context_discovery_selects_live_domains_once_and_ignores_unreferenced_domains():
    first = _domain()
    second = _domain(0x2000, domain_id=2)
    unused = _domain(0x3000, domain_id=3, released=True)
    args = _args(123, 0x2000, 0x1000, 0x1000)
    assert discover_ccu_domains(args, 0, [first, second, unused]) == (first, second)


@pytest.mark.parametrize("changes", [{"released": True}, {"freed": True}])
def test_context_discovery_rejects_retired_domain_handles(changes):
    with pytest.raises(ValueError, match="released communication domain"):
        discover_ccu_domains(_args(0x1000), 0, [_domain(**changes)])


@pytest.mark.parametrize("engines", [(), ("sdma",), ("urma",), ("rdma",)])
def test_context_discovery_rejects_wrong_engine(engines):
    with pytest.raises(ValueError, match="workspace_engines"):
        discover_ccu_domains(_args(0x1000), 0, [_domain(engines=engines)])


@pytest.mark.parametrize("domain_id", [0, (1 << 64) - 1])
def test_context_discovery_preserves_zero_based_and_maximum_domain_ids(domain_id):
    domain = _domain(domain_id=domain_id)
    assert discover_ccu_domains(_args(0x1000), 0, [domain]) == (domain,)
    assert domain.domain_id == domain_id


@pytest.mark.parametrize("changes", [{"generation": 0}, {"domain_id": -1}, {"domain_id": 1 << 64}])
def test_context_discovery_rejects_invalid_identity(changes):
    with pytest.raises(ValueError, match="invalid domain identity"):
        discover_ccu_domains(_args(0x1000), 0, [_domain(**changes)])


@pytest.mark.parametrize("address,worker", [(0x9999, 0), (0x1000, 1), (0, 0)])
def test_context_discovery_rejects_unknown_or_wrong_worker_context(address, worker):
    with pytest.raises(ValueError, match="live domain device_context scalar"):
        discover_ccu_domains(_args(address), worker, [_domain()])


def test_context_discovery_rejects_ambiguous_context_addresses():
    with pytest.raises(ValueError, match="ambiguously"):
        discover_ccu_domains(_args(0x1000), 0, [_domain(), _domain(domain_id=2)])


def test_context_discovery_does_not_treat_all_task_tensors_as_ccu_arguments():
    storage = ctypes.create_string_buffer(64)
    ordinary_buffer = wrap_fork_inherited(ctypes.addressof(storage), 64, mint_owner_instance_id(), 1, "L3")
    args = TaskArgs()
    args.add_tensor(ordinary_buffer.tensor((16,), int(DataType.FLOAT32.value)))
    args.add_scalar(0x1000)
    domain = _domain()
    assert discover_ccu_domains(args, 0, [domain]) == (domain,)
    assert args.tensor_count() == 1


@pytest.mark.parametrize("target", [None, SimpleNamespace(), SimpleNamespace(ccu_function_count=0)])
def test_ordinary_callables_need_no_ccu_backend(target):
    require_ccu_execution_backend(target)


class _NativeOrchestrator:
    def __init__(self):
        self.calls = []

    def submit_next_level(self, *args):
        self.calls.append(args)
        return "submitted"

    def submit_next_level_group(self, *args):
        self.calls.append(args)
        return "submitted"


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("ccu", [False, True])
def test_normal_registered_submission_refuses_missing_ccu_backend_before_native_launch(grouped, ccu):
    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[0, 1])
    try:
        handle = worker.register(_callable(ccu=ccu))
        assert worker._live_domains == {}
        worker._live_domains["domain_1"] = _domain()
        native = _NativeOrchestrator()
        orch = Orchestrator(native, worker=worker)

        def submit():
            if grouped:
                return orch.submit_next_level_group(handle, [_args(0x1000)], workers=[0])
            return orch.submit_next_level(handle, _args(0x1000), worker=0)

        if ccu:
            with pytest.raises(RuntimeError, match="not prepared"):
                submit()
            assert native.calls == []
        else:
            assert submit() == "submitted"
            assert len(native.calls) == 1
    finally:
        worker._live_domains.clear()
        worker.close()


def test_direct_chip_submission_refuses_missing_ccu_backend_before_native_launch(monkeypatch):
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=0)
    try:
        handle = worker.register(_callable())
        native_calls = []
        monkeypatch.setattr(worker, "_chip_worker", SimpleNamespace())
        monkeypatch.setattr(worker, "_submit_l2_locked", lambda *args, **kwargs: native_calls.append(args))
        with pytest.raises(RuntimeError, match="cannot execute"):
            worker._submit_locked(handle, TaskArgs(), None)
        assert native_calls == []
    finally:
        worker._chip_worker = None
        worker.close()


def test_native_preparation_refuses_ccu_before_argument_reservation(monkeypatch):
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=0)
    try:
        handle = worker.register(_callable())
        calls = []

        def reserve(*args, **kwargs):
            calls.append("reserve")
            raise AssertionError("unsupported CCU reached argument reservation")

        monkeypatch.setattr(worker, "_chip_worker", SimpleNamespace())
        monkeypatch.setattr(worker, "_operation_lease", lambda _api: contextlib.nullcontext())
        monkeypatch.setattr(worker, "_prepare_l2_args", reserve)
        with pytest.raises(RuntimeError, match="cannot execute"):
            worker.prepare_submission(handle, TaskArgs(), caller_stream=123)
        assert calls == []
        assert worker._chip_run_callable_ids == {}
    finally:
        worker._chip_worker = None
        worker.close()


@pytest.mark.parametrize(
    "platform,runtime,members,message",
    [
        ("a2a3", "tensormap_and_ringbuffer", [0, 1], "requires A5"),
        ("a5sim", "tensormap_and_ringbuffer", [0, 1], "requires A5"),
        ("a5", "host_build_graph", [0, 1], "A5 TMR"),
        ("a5", "tensormap_and_ringbuffer", [0], "exactly two"),
        ("a5", "tensormap_and_ringbuffer", [0, 1, 2], "exactly two"),
    ],
)
def test_ccu_domain_scope_rejected_before_allocation(platform, runtime, members, message):
    worker = Worker(level=3, platform=platform, runtime=runtime, device_ids=[0, 1, 2])
    try:
        with pytest.raises(ValueError, match=message):
            worker._domain_config(members, 1024, [], True, ("ccu",))
        assert worker._live_domains == {}
    finally:
        worker.close()


def test_ccu_two_rank_tmr_configuration_has_its_own_workspace_bit():
    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[0, 1])
    try:
        config = worker._domain_config([0, 1], 1024, [], True, ("ccu",))
        assert config.workspace_mask() == 8
    finally:
        worker.close()


def test_ccu_domain_preparation_is_ordered_cached_and_refreshes_new_definitions(monkeypatch):
    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[4, 5])
    calls = []
    monkeypatch.setattr(worker, "_control_admission", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(worker, "_broadcast_py_control", lambda *a, **kw: calls.append(kw))
    registry = {
        b"z" * 32: SimpleNamespace(target=SimpleNamespace(ccu_function_count=1)),
        b"a" * 32: SimpleNamespace(target=SimpleNamespace(ccu_function_count=1)),
        b"b" * 32: SimpleNamespace(target=SimpleNamespace(ccu_function_count=0)),
    }
    monkeypatch.setattr(worker, "_identity_registry", registry)
    try:
        domain = _domain()
        worker._prepare_ccu_domain(domain)
        assert [call["digest"] for call in calls] == [b"a" * 32, b"z" * 32]
        worker._prepare_ccu_domain(domain)
        assert len(calls) == 2
        registry[b"c" * 32] = SimpleNamespace(target=SimpleNamespace(ccu_function_count=1))
        worker._prepare_ccu_domain(domain)
        assert calls[-1]["digest"] == b"c" * 32
        assert len(calls) == 3
        assert calls[-1]["payload"][-8:] == b"\x04\0\0\0\x05\0\0\0"
    finally:
        worker._identity_registry.clear()
        worker.close()


def test_ccu_prepare_failure_does_not_admit_a_submission(monkeypatch):
    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[0, 1])
    monkeypatch.setattr(worker, "_control_admission", lambda _: contextlib.nullcontext())
    worker._identity_registry = {b"a" * 32: SimpleNamespace(target=SimpleNamespace(ccu_function_count=1))}

    def fail(*args, **kwargs):
        raise RuntimeError("peer prepare failed")

    monkeypatch.setattr(worker, "_broadcast_py_control", fail)
    try:
        with pytest.raises(RuntimeError, match="peer prepare failed"):
            worker._prepare_ccu_domain(_domain())
        assert worker._ccu_prepared_domains == set()
    finally:
        worker._identity_registry.clear()
        worker.close()


def test_ccu_last_handle_unregister_preserves_identity_until_domain_release(monkeypatch):
    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[0, 1])
    target = _callable()
    first = worker.register(target)
    last = worker.register(target)
    state = worker._identity_registry[first.digest]
    domain = _domain()
    worker._ccu_prepared_domains.add((first.digest, domain.domain_id, domain.generation))
    broadcasts = []
    released = []
    monkeypatch.setattr(worker, "_pre_start_unregister_if_needed", lambda _: False)
    monkeypatch.setattr(worker, "_operation_lease", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(worker, "_broadcast_unregister", broadcasts.append)
    monkeypatch.setattr(worker, "_release_domain_now", released.append)
    try:
        assert state.ref_count == 2
        worker.unregister(first)
        assert state.ref_count == 1
        assert broadcasts == [first.digest]

        identities = dict(worker._identity_registry)
        handles = dict(worker._live_handles)
        registry = dict(worker._callable_registry)
        with pytest.raises(RuntimeError, match="release prepared CCU domains"):
            worker.unregister(last)
        assert worker._identity_registry == identities
        assert worker._live_handles == handles
        assert worker._callable_registry == registry
        assert worker._pending_unregister_cids == set()
        assert state.ref_count == 1
        assert broadcasts == [first.digest]

        # A recorded successful backend release permits the normal fence cleanup.
        worker._domain_free_results[domain.allocation_id] = None
        worker._free_domain_after_fence(domain)
        assert released == [domain]
        assert worker._ccu_prepared_domains == set()
        worker.unregister(last)
        assert broadcasts == [first.digest, first.digest]
        assert first.digest not in worker._identity_registry
        assert state.slot_id not in worker._callable_registry
        assert last._handle_id not in worker._live_handles
    finally:
        worker._ccu_prepared_domains.clear()
        worker.close()

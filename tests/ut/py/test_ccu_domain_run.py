# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU coverage for rank joins, READY admission, and retained stop failures."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import simpler.worker as worker_mod
from simpler.ccu import ChipCcuBinding
from simpler.comm_domain import DomainConfig
from simpler.orchestrator import _callback_run
from simpler.task_interface import CommDomainHandle
from simpler.worker import Worker


def test_registration_collective_needs_topology_but_no_allocation():
    observed = []

    def control(_kind, chip, _command, payload, _timeout):
        allocation, generation, operation, digest, result = worker_mod._CCU_DOMAIN_REQUEST.unpack_from(payload)
        assert (allocation, generation, operation, digest, result) == (0, 0, 5, b"c" * 32, 0)
        tail = worker_mod._CCU_DOMAIN_REQUEST.size
        assert worker_mod.struct.unpack_from("<II2I", payload, tail) == (2, chip, 0, 1)
        observed.append(chip)
        reply = bytearray(payload)
        worker_mod.struct.pack_into("<i", reply, tail - 4, 17)
        return reply

    worker: Any = SimpleNamespace(_worker=SimpleNamespace(control_payload=control), _py_control_timeout_s=1)
    topology = SimpleNamespace(workers=(0, 1), allocation_id=0)
    assert Worker._ccu_domain_collective(worker, topology, 5, b"c" * 32) == [17, 17]
    assert sorted(observed) == [0, 1]


def _domain():
    handle = CommDomainHandle(name="ccu", workers=(0, 1), allocation_id=9, _release_fn=lambda _: None)
    handle._config = DomainConfig((0, 1), 4096, (), True, ())
    return handle


def test_collective_joins_other_ranks_before_raising():
    seen = []

    def control(_kind, rank, _command, payload, _timeout):
        seen.append(rank)
        if rank == 0:
            raise RuntimeError("rank zero refused")
        return payload

    worker: Any = SimpleNamespace(_worker=SimpleNamespace(control_payload=control), _py_control_timeout_s=1)
    domain = _domain()
    with pytest.raises(RuntimeError, match="rank zero"):
        Worker._ccu_domain_collective(worker, domain, 1)
    assert sorted(seen) == [0, 1]
    assert domain._ccu_run_generation == 1


def test_ready_wait_requires_every_rank():
    replies = iter(([0, 1], [0, 0]))
    observed = []

    def collective(_handle, operation):
        observed.append(operation)
        return next(replies)

    worker: Any = SimpleNamespace(_ccu_domain_collective=collective, _py_control_timeout_s=1)
    Worker._ccu_domain_wait(worker, _domain(), 2)
    assert observed == [2, 2]


def test_use_domain_only_records_keepalive_without_launch():
    worker: Any = SimpleNamespace(level=3, _building_run_resources=worker_mod._RunResources())
    observed = []
    domain = _domain()

    def collective(*_):
        pytest.fail("borrowing a domain must not prepare or launch CCU")

    worker._control_admission = lambda _api: nullcontext()
    worker._ccu_domain_collective = collective
    worker._ccu_domain_wait = lambda _handle, operation: observed.append(operation)
    worker._release_domain_handle = lambda *_: None
    worker._abort_ccu_domain = lambda _: pytest.fail("unexpected abort")
    with _callback_run(1, worker):
        Worker._use_domain(worker, domain)
        Worker._use_domain(worker, domain)
    assert observed == []
    assert worker._building_run_resources.ccu_domains == {}
    assert worker._building_run_resources.requires_ordered_cleanup


def test_task_failure_aborts_all_used_ccu_domains():
    first, second = _domain(), _domain()
    first.name, second.name = "first", "second"
    aborted = []
    worker: Any = SimpleNamespace(_abort_ccu_domain=lambda handle: aborted.append(handle.name))
    resources = worker_mod._RunResources(ccu_domains={"first": first, "second": second})
    error = RuntimeError("ordinary task failed")
    with pytest.raises(RuntimeError, match="CCU run aborted") as result:
        Worker._stop_run_ccu_domains(worker, resources, error)
    assert result.value.__cause__ is error
    assert aborted == ["first", "second"]
    assert list(resources.ccu_domains) == ["first", "second"]


def test_abort_marks_unconfirmed_resources():
    observed = []

    def collective(_handle, operation):
        assert operation == 4
        raise RuntimeError("stream stop unconfirmed")

    worker: Any = SimpleNamespace(
        _ccu_domain_collective=collective,
        _record_unreclaimable=lambda message, error: observed.append((message, error)),
    )
    Worker._abort_ccu_domain(worker, _domain())
    assert "retained" in observed[0][0]
    assert str(observed[0][1]) == "stream stop unconfirmed"


def test_prepare_infers_callable_and_reuses_domain_without_launch():
    domain = _domain()
    resources = worker_mod._RunResources(live_domains={domain.name: domain})
    state = SimpleNamespace(target=SimpleNamespace(ccu_module_count=1), eligible_worker_ids=(0, 1))
    observed = []
    worker: Any = SimpleNamespace(
        level=3,
        _config={"platform": "a5"},
        _building_run_resources=resources,
        _resolve_handle=lambda *_args, **_kwargs: state,
        _control_admission=lambda _: nullcontext(),
        _ccu_domain_collective=lambda handle, operation, digest: observed.append((handle, operation, digest)),
    )
    callable_handle = SimpleNamespace(digest=b"c" * 32)
    Worker._prepare_ccu_submission(worker, callable_handle, [0])
    Worker._prepare_ccu_submission(worker, callable_handle, [1])
    assert observed == [(domain, 0, callable_handle.digest)]
    assert resources.ccu_domains == {domain.name: domain}
    with pytest.raises(ValueError, match="different CCU callables"):
        Worker._prepare_ccu_submission(worker, SimpleNamespace(digest=b"d" * 32), [0])


@pytest.mark.parametrize("domain_count", [0, 2])
def test_ccu_submission_rejects_missing_or_ambiguous_domain(domain_count):
    domains = {str(i): _domain() for i in range(domain_count)}
    worker: Any = SimpleNamespace(
        level=3,
        _config={"platform": "a5"},
        _building_run_resources=worker_mod._RunResources(live_domains=domains),
        _resolve_handle=lambda *_args, **_kwargs: SimpleNamespace(target=SimpleNamespace(ccu_module_count=1)),
    )
    with pytest.raises(ValueError, match="exactly one"):
        Worker._prepare_ccu_submission(worker, SimpleNamespace(digest=b"c" * 32), [0])


def test_l2_launches_only_on_begin_and_confirms_stop_before_reuse():
    operations = []
    worker = SimpleNamespace(control_domain_ccu=lambda comm, domain, op, gen: operations.append((op, gen)) or 0)
    binding = ChipCcuBinding(worker, 10, 9, 1)
    assert operations == []
    binding.begin()
    with pytest.raises(RuntimeError, match="overlapping"):
        binding.begin()
    binding.finish()
    binding.begin()
    binding.finish()
    assert operations == [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)]
    assert not binding.active


@pytest.mark.parametrize("failed_operation", [1, 2, 3])
def test_l2_failure_aborts_and_keeps_binding_active(failed_operation):
    operations = []

    def control(comm, domain, operation, generation):
        operations.append(operation)
        if operation == failed_operation:
            raise RuntimeError("original failure")
        if operation == 4:
            raise RuntimeError("abort unconfirmed")
        return 0

    binding = ChipCcuBinding(SimpleNamespace(control_domain_ccu=control), 10, 9, 1)
    with pytest.raises(RuntimeError, match="original failure"):
        binding.begin()
        binding.finish()
    assert operations[-1] == 4
    assert binding.active

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
from simpler.comm_domain import DomainConfig
from simpler.orchestrator import _callback_run
from simpler.task_interface import CommDomainHandle
from simpler.worker import Worker


def _domain():
    handle = CommDomainHandle(name="ccu", workers=(0, 1), allocation_id=9, _release_fn=lambda _: None)
    handle._config = DomainConfig((0, 1), 4096, (), True, (), ccu_binding=(bytes(32), 7, bytes(32), bytes(32)))
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


def test_use_domain_records_keepalive_before_launch_and_ready():
    worker: Any = SimpleNamespace(level=3, _building_run_resources=worker_mod._RunResources())
    observed = []
    domain = _domain()

    def collective(handle, operation, _binding):
        assert worker._building_run_resources.ccu_domains[handle.name] is handle
        observed.append(operation)

    worker._control_admission = lambda _api: nullcontext()
    worker._ccu_domain_collective = collective
    worker._ccu_domain_wait = lambda _handle, operation: observed.append(operation)
    worker._release_domain_handle = lambda *_: None
    worker._abort_ccu_domain = lambda _: pytest.fail("unexpected abort")
    with _callback_run(1, worker):
        Worker._use_domain(worker, domain)
        Worker._use_domain(worker, domain)
    assert observed == [1, 2]
    assert worker._building_run_resources.requires_ordered_cleanup


def test_stop_failure_aborts_all_domains_and_preserves_first_error():
    first, second = _domain(), _domain()
    first.name, second.name = "first", "second"
    aborted = []
    stopped = []

    def stop(handle, operation):
        stopped.append((handle.name, operation))
        raise RuntimeError(handle.name)

    worker: Any = SimpleNamespace(_ccu_domain_wait=stop, _abort_ccu_domain=lambda handle: aborted.append(handle.name))
    resources = worker_mod._RunResources(ccu_domains={"first": first, "second": second})
    with pytest.raises(RuntimeError, match="first"):
        Worker._stop_run_ccu_domains(worker, resources, None)
    assert stopped == [("first", 3), ("second", 3)]
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

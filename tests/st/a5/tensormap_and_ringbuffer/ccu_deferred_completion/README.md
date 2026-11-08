# CCU deferred completion on A5

This scene uses the public Worker path, an embedded ABI 2 builder module,
the PR1 Session provider, and the PR2 TMR completion engine on two A5 devices.
The `smoke` case runs four rounds; `thousand_rounds` runs 1000 rounds per rank.

Each round initializes domain-owned source, destination and scratch tensors,
submits a CCU request from AIV, and returns without waiting. A dependent AIV
consumer checks the received data, the unchanged source and scratch, and both
destination guards. The next round depends on that consumer. Host validation
checks every round's status, root, count and offset, then checks domain cleanup.

The two embedded entries implement broadcasts rooted at rank 0 and rank 1.
Rounds alternate 64/1024 elements, cycle eight 64-byte-aligned offsets and use
different source data on each rank. Both entries receive three positional
tensors and one scalar; users do not supply callable IDs or platform handles.

The test builds `kernels/host/ccu_positional.cpp` with the native `CXX` compiler
(default `c++`), `-std=c++17 -O2 -shared -fPIC -I src/common`. It embeds the
resulting bytes in `CcuFunction`; no module path is passed to a child worker.
The usual scene compiler builds the orchestration and AIV kernels. All three
PRs and their runtime/extension build products must be present before running.

After the repository A5 architecture precheck, use the hardware queue:

```bash
task-submit --device auto --device-num 2 --timeout 1800 --max-time 1800 \
  --run 'python -m pytest tests/st/a5/tensormap_and_ringbuffer/ccu_deferred_completion --platform a5 --device "$TASK_DEVICE" -v'
```

Select `-k smoke` for the initial run. No fake backend, stand-alone request
polling loop, or direct SDK launch substitutes for the normal Worker path.
The test exercises deferred dependencies; it does not measure how much of a
particular hardware request remains outstanding at the AIV return instruction.

## Validation status (2026-11-06)

Combined snapshot `68e00c466` passed both cases through the normal Worker/TMR
path on devices 0 and 1. `smoke` passed in 23.8 seconds
(`task_20261106_191642_263921111571`); `thousand_rounds` passed in a 9.1-second job,
with pytest reporting 8.60 seconds (`task_20261106_191739_268998428471`). Both exited
0 after the per-round checks, domain reclamation and normal Worker close.

The wrapper reads Tensor metadata field by field from GM before forming local
positional arguments. This fixes the earlier descriptor-copy failure at the
first send. The full TMR build, offline device compilation, 23 completion CPU
cases, five shutdown CPU cases and the shared-memory test group passed.

This accepts the tested single-domain, two-entry scene. It does not cover
retained-domain reuse/recreation, cross-domain/callable isolation, one-rank
preparation failures or fault injection during active CCU execution. It also
does not force a request to remain pending at AIV FIN. Standalone PR1 results
remain separate component evidence. See the
[integration contract](../../../../../docs/design/ccu-pr3-requirements.md)
for preparation, ownership and shutdown requirements.

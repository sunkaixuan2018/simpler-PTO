# Embedded CCU callable and domain integration

The requirements and implementation status are also available in [Chinese](../zh-cn/design/ccu-pr3-requirements.md).
This page describes the current integration candidate. Runtime and hardware validation are reported separately;
passing the earlier component tests does not validate the full path.

## Public contract

`CcuFunction.build(binary, entry)` embeds an ELF64 host builder in a normal
`ChipCallable`. Its list position is the `ccu_id`. The module exports
`uint32_t simpler_ccu_module_abi_version()` returning 2, and an entry with the
signature `int(CcuBuildContext*)`. Tensors and scalars use positional indices.
Legacy module ABI 1 must be rebuilt. The independent internal `Builder/Program`
test component retains ABI 1 and does not load production modules.

A domain must select `workspace_engines=("ccu",)` and have exactly two members
on A5 using `tensormap_and_ringbuffer`. Task arguments carry a live domain context
and domain-owned tensors. Simulators, other architectures/runtimes and direct L2
CCU submissions remain unsupported. Caller-provided program keys, platform
handles and callable IDs are not part of the public interface.

## Preparation and execution

Domain creation or reuse prepares each currently registered CCU callable in
stable digest order, before the orchestration callback submits its first task.
New callables are prepared at the next domain reuse. Each callable/domain pair
has its own Session, communicator root information, module ownership and memory
registration. This eager strategy has one initialization cost per pair. Both
ranks verify the domain generation and hidden native callable ID.

The domain owner retains its VMM window, Session and uploaded publication images.
The PR2 Workspace points to the PR1 CcuDeviceContext; the real request record is
owned by PR1. Workspace/context ABI is 2. ChipCallable wire format remains 1,
with its embedded module ABI field independently set to 2. Rebuild the Python
extension, host runtime, modules and device kernels together.

Native preparation acquires a run lease. After ordinary producers stop, PR2
polls or drains the CCU request before releasing its dependencies and owners.
A preparation proven never launched cancels only its own run reference.
Unconfirmed readers and partially created objects stay owned until SDK cleanup
is confirmed. Domain close stops CCU before freeing windows and parent
communication resources. A closed Session alone does not authorize device reset:
the parent CommHandle must also be fully released. Recovery skips device-wide
synchronization while CCU communication owners remain. Fatal finalization stops
Host collectors first, then retains the context and runtime if readers, service
shutdown or parent communication cleanup remain unconfirmed. An error code is
not proof that a producer stopped. The ordinary non-CCU path is unchanged.
Release a domain before unregistering its callable;
a published instance prevents hidden callable-slot reuse.

Preparation fanout joins all members before rollback. Failed creation and
failed preparation of a newly registered callable in a reused domain both
retire through the existing fence-owned domain ledger. Failed cleanup remains
visible and prevents subsequent work.

## Validation

The [two-rank end-to-end scene](../../tests/st/a5/tensormap_and_ringbuffer/ccu_deferred_completion/README.md)
loads two positional entries, returns from AIV after asynchronous submission,
and validates dependent consumers over four or 1000 changing-address rounds.
It checks data, destination guards, unchanged source/scratch and domain cleanup.
This test requires the combined three-PR build and real A5 hardware.

Focused component tests remain under `tests/ut/cpp/common/platform_comm/ccu/`
and `tests/ut/py/test_ccu_runtime.py`. The PR1 standalone onboard fixture lives
under `tests/ut/cpp/a5/platform/support/ccu_session_onboard.cpp`; it is useful
for platform diagnosis but does not replace Worker/TMR end-to-end validation.

### Current evidence (2026-11-06)

- The latest focused CPU suite passed 26 cases: 21 completion-protocol cases and
  five shutdown/ownership cases. The full A5 TMR build and offline device
  compilation also passed.
- The combined `05d9…` hardware run submitted at 18:47:36 on devices 2 and 3
  failed at the first send with error 101 (`ASYNC_COMPLETION_INVALID`) in about
  7.4 seconds. It did not repeat the earlier roughly 100-second device drains
  or teardown segmentation faults. This verifies that failure-cleanup path;
  the combined execution is still failing and is not accepted.
- PR1 standalone `ba178…` passed 1000 two-rank rounds. That component result does
  not validate the Worker/TMR path. Both successful four-round and 1000-round
  integrated runs, including dependency completion and domain reclamation,
  remain required.

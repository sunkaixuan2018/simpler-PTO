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

The combined `68e00c466` Worker/TMR scene passed on A5 devices 0 and 1:

| Case | Hardware task | Result |
| ---- | ------------- | ------ |
| Four rounds | `task_20261106_191642_263921111571` | PASS, 23.8 s, exit 0 |
| 1000 rounds per rank | `task_20261106_191739_268998428471` | PASS, 9.1 s job duration (8.60 s pytest), exit 0 |

Both cases checked per-round data, guards, unchanged source/scratch, domain
reclamation and normal Worker shutdown. The AIV wrapper reads each Tensor field
explicitly from GM into its local descriptor before constructing positional
arguments; this fixes the earlier first-send error 101 caused by the descriptor
copy path. There is no per-round Host preparation.

The full A5 TMR build and offline device compilation passed. The focused CPU
runs passed 23 completion cases, five shutdown cases and the shared-memory test
group. PR1 standalone `ba178…` also passed 1000 two-rank rounds, independently of
the combined evidence above. The earlier `05d9…` failure returned in about
7.4 seconds without the old device-drain delay or teardown crash; that remains
failure-cleanup evidence, not a successful communication run.

These results accept the tested single-domain, two-entry execution path. Hardware
acceptance remains open for retained-domain reuse/recreation, cross-domain and
cross-callable isolation, one-rank preparation failure and faults with an active
CCU request. The scene does not force or measure CCU still being pending at the
exact AIV FIN instant; a controlled delayed-completion test remains required for
that stronger timing claim.

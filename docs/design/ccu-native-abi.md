# Native CCU module and three-Event ABI

ABI V2 packages native HCOMM CCU definitions with a `ChipCallable`. The module
owns the algorithm and its control flow. Simpler validates exact module bytes
and provides domain resources and asynchronous completion through later runtime
layers. The public C/POD layouts are in
`src/common/platform_comm/ccu/ccu_native_abi.h`.

## Version and packaging

V2 is incompatible with V1. A module exports:

```c
const SimplerCcuModuleManifestV2 *simpler_ccu_module_manifest_v2(void);
```

The entry must not throw and returns an immutable contiguous manifest. Kernel
and string tables use byte offsets from its base. All reserved
fields and flags must be zero unless an explicit capability is required.
Kernel records require `SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS` and V2 registration
and task layouts. Group selection is deferred; all declared kernels are prepared.

The simplified V2 registration/workspace layout tag is `0x00020001`. Task
arguments remain layout 2. Old V2 instance-table fields are now reserved and
must be zero, so old modules fail validation instead of being misinterpreted.
Modules and device wrappers must be rebuilt together for this revision.

The package envelope remains version 1 because its byte layout is unchanged;
each module record now requires ABI version 2. An old package record, V1-only
export, old manifest version, invalid layout, unknown capability or mismatched
SDK/PTO-ISA build identity is rejected before resource preparation. Packaging
preserves opaque bytes; it does not inspect ELF exports. The native loader
performs manifest validation when opening the module.

The loader opens exact ELF bytes through a private unlinked file with
`RTLD_NOW | RTLD_LOCAL`. Cache identity includes those bytes, the complete
manifest, native entry names and both build identities.
The backing file descriptor stays alive until after `dlclose`, preventing its
`/proc/self/fd` path from being reused by another live module. Cache lookup
currently loads and validates a candidate first, including its returned manifest;
it shares validated module ownership rather than avoiding all loader work.

## Kernel List And Callable Index

`SimplerCcuKernelRecordV2` describes a kernel's native entry, parameter size and
alignment, and supported layouts. Kernel IDs must be unique within a module.
The entry name must be a valid C symbol exported by that module.

One listed kernel owns one local execution resource set. There is no separate
instance table or configurable mapping. For multiple modules, kernel records
are concatenated in module package order and record order. Their zero-based
position is `ccu_id`, scoped to that L2 callable. Module-local kernel IDs are
metadata; equal IDs in different modules are allowed and never select device
resources. Registration and workspace construction preserve list order.

## Registration and launch

The native definition signature is:

```c
int32_t entry(const SimplerCcuRegistrationBindingV2 *binding);
```

It must not throw. Query and registration receive the same one-element argument
array `{&binding}`. The Host-only binding identifies the callable-local `ccu_id`
and rank, with a channel and three distinct opaque Event handles:
`ready_event_handle`, `trigger_event_handle`, `done_event_handle`. These handles
are SDK values, not device addresses. Each Event uses bit 0.

`SimplerCcuLaunchArgumentsV2` contains four 64-bit words, passed directly to
`HcommCcuKernelLaunch`:

| Word | Value |
| ---- | ----- |
| 0 | `layout_version` = 2 |
| 1 | `parameter_addr` |
| 2 | `parameter_token` |
| 3 | `reserved` = 0 |

The native definition loads all four words; `HcommCcuGetTaskArgsNum` must report
4. The address and access token are per-run values for the kernel's declared
parameter region, and must not be captured in cached registration. Ready,
trigger and done are bound through registration. There are no run-control or
completion-slot launch words.

## Domain workspace

The domain runtime context publishes `SimplerCcuWorkspaceV2`: ABI/layout version,
domain identity, run generation, `ccu_count`, control stride and `controls_addr`
pointing to ordered `SimplerCcuControlV2` entries. It is a read-only view
for the duration of the run. Each control supplies its callable-local `ccu_id`,
module metadata, matching run generation, parameter address/size/alignment
and token, and three `{address, mask}` Event descriptors. Masks equal
`SIMPLER_CCU_EVENT_MASK` (1). Addresses are opaque platform mappings; register
encoding, cache maintenance and clear confirmation belong in platform helpers.

The runtime validates versions, sizes, domain/run identity, indexes, parameter
ranges/alignment and distinct Event mappings before admitting submissions.
The AIV resolves `comm_ctx -> workspace -> controls[ccu_id]` with bounds and
generation checks. It never scans kernel IDs or translates another index. Orchestration
does not pass individual resource addresses in business arguments.

## Submission and completion

1. Host safely resets Events and launches the missions. After required
   initialization, CCU records ready. AICPU observes, clears and confirms ready
   before admitting ordinary tasks.
2. AIV resolves its mission and registers its task's done completion condition
   before writing parameters. It publishes parameter/input visibility, records
   trigger with the required synchronization, and returns immediately.
3. CCU waits for and consumes trigger, reads parameters, executes the algorithm
   and completes data movement and required cross-rank synchronization before
   publishing done with output visibility.
4. The existing asynchronous task mechanism waits for both ordinary execution
   completion and done. The completion adapter clears done and confirms the
   clear before retiring the condition and releasing dependent tasks.

There is no consumed Event, acknowledgement wait, GM completion sequence,
mandatory stop/final-wake protocol, or CCU-published EXITED state. Trigger means
submission has been issued; it does not prove CCU has consumed the parameters.
Task IDs, owner generations and error reporting belong to the runtime, not to
a CCU-written completion slot. Event bits do not encode native error codes.

Users serialize operations on one mission through the entire asynchronous
task's completion, including before the next parameter write. Parameters and
referenced data stay alive until then. Waiting only for AIV return is
insufficient. Repeated submission to the same mission within one AIV is outside
this contract. Independent missions may execute concurrently. The initial
implementation provides no owner/busy detection or queue; an atomic owner claim
before parameter writes is a separate extension.

## Exit and failure

The user's finite kernel control flow returns normally. Host confirms termination
using the mission's dedicated stream query or bounded synchronization. Final done
does not prove stream termination. The runtime rejects overlapping runs sharing
mutable resources and reuses them only after confirmed termination.

Startup, completion and final stream waits have bounded failure handling through
runtime/SDK error channels. A failure may terminate the whole run. If stopping
cannot be confirmed, submissions remain disabled and module, Event, parameter,
window, token and stream resources remain alive. Neither a timeout nor a clear
attempt authorizes resource reuse. These runtime obligations and hardware
acceptance are implemented in the dependent runtime changes, not in this ABI.

## A5 runtime integration

`comm_register_ccu_callable` accepts callable bytes, build identities and rank
topology, with no allocation ID or window dependency. It loads and validates
modules, creates HCOMM resources and registers every listed kernel, without
launching. A shared collective identity selects a cached registration; exact
bytes, build identities and rank topology must match it. HCOMM programs capture channel/Event handles, so different
topologies require different registration resources.

The L3 parent assigns a shared `collective_id` to each callable/topology and
sends it to every participant. It uses the same identity allocator as windows
to avoid subcommunicator ID collisions. Rank-local registration cache IDs are
separate and may differ when ranks have participated in different subsets.
Binding state is optional: domain ID zero is a valid bound allocation.

`comm_bind_domain_ccu` attaches an existing registration ID to compatible windows;
it does no module loading or HCOMM registration. Window release detaches the
registration. The enclosing communicator retains and reuses its programs,
Events and streams until confirmed communicator teardown. The cache retains
one resource set per distinct callable/topology for that communicator lifetime.
Registering a different callable does not overwrite another callable's resources.

The Python submission path lazily registers the selected callable for the domain's
rank topology, then binds it. The native registration API also supports registration
before any windows exist. L2 activation
launches its resources and gates ordinary work on ready observation and confirmed
clear. The startup helper observes mapped Events on the device; asynchronous
done polling and clearing run on the AICPU. Normal completion requires the
dedicated mission streams to terminate. Abort without confirmed stop retains
owners and closes admission.

The prototype exposes callable-local list indexes through the revised V2
workspace. Multiple matching domains, overlapping runs on one
domain and different callables on one domain within a parent run are unsupported.
One registration cannot be attached to two live domain allocations simultaneously.

The hardware smoke uses finite programs with one request per mission per L2
execution. Host run repetition does not establish dynamic repeated CCU calls
within one L2 execution. Same-mission repeat within an AIV and the proposed
generic Tensor call API are outside this integration.

For allocated device windows, `workspace_engines=("ccu",)` publishes a
rank-ordered `CommCcuWindow` descriptor table through the CCU slot in
`CommContext::workspaces`. Each descriptor contains the owner's address, the
requested valid byte range and the token acquired during allocation. Imported
AIV peer addresses remain separate. The descriptor table is immutable while W
is alive and is released only after domain quiescence.

The typed AIV submit overload resolves `CcuCallArg` window references against
that table before completion admission or trigger. A window argument expands
to two uint64 words (address plus byte offset, token); a scalar expands to one.
Native modules declare the expanded parameter size and consume the same order.
This does not alter V2 launch arguments or reuse P's token as a W token.
The CommContext layout grows to 1120 bytes; all producers and consumers must
be rebuilt together. The CCU native module V2 layout remains unchanged.

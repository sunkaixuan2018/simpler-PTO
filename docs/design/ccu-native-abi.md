# Native CCU module and completion ABI

This contract packages native HCOMM CCU definitions with a `ChipCallable`.
The module owns the communication algorithm and control flow. Simpler loads and
checks the module, then later runtime layers may pass its native entry addresses
to HCOMM query and registration APIs.

This ABI does not contain `CcuBuildContext`, a builder callback, an algorithm
DSL, a shared service program, or a Host request queue. It also does not define
HCOMM domain preparation, registration, launch, or TMR scheduling behavior.

## Callable package

`ChipCallable` stores a versioned module table after its orchestration binary
and AICore children. Every table reference is a byte offset from the table
base. A module record contains the exact ELF bytes supplied by the caller; it
does not contain a path or a Host pointer.

The loader writes those same bytes to a private unlinked file and opens its
`/proc/self/fd` path with `RTLD_NOW | RTLD_LOCAL`. The cache identity includes:

- the exact ELF bytes;
- the complete returned manifest bytes;
- every native entry symbol;
- the HCOMM SDK and PTO-ISA build identities.

Changing any item produces a different cache identity.

## Module exports

A version-1 module exports this C symbol:

```c
const SimplerCcuModuleManifestV1 *
simpler_ccu_module_manifest_v1(void);
```

The function must not throw across the C boundary. It returns one immutable,
packed manifest. Kernel records and entry names are reached through offsets
inside that manifest. Each entry name resolves to a native CCU definition
function in the same ELF. The later registration layer passes that function
address directly to `HcommCcuKernelQueryResReq` and
`HcommCcuKernelRegister`.

The loader rejects an unknown version or capability, a changed struct size,
a nonzero reserved field, duplicate or invalid kernel IDs, an invalid group,
an unsupported registration or task-argument layout/count, a bad parameter
size/alignment, an out-of-range or invalid C entry name, a missing symbol, and
an SDK or PTO-ISA build identity mismatch. These checks complete before domain
preparation. Version 1 requires both completion-slot and stop-gate
capabilities.

The exact field layouts and sizes are defined by
`src/common/platform_comm/ccu/ccu_native_abi.h`. All shared records are C/POD,
use fixed-width integers, and contain no Host pointers.

## Registration binding

Every native definition entry has this fixed C signature:

```c
int32_t entry(const SimplerCcuRegistrationBindingV1 *binding);
```

It must not throw across the C boundary. For both
`HcommCcuKernelQueryResReq` and `HcommCcuKernelRegister`, the adapter supplies
the same one-element argument array:

```c
const void *registration_args[] = {&binding};
```

The Host-only binding contains the kernel/group/rank identity, one domain
channel handle, and separate trigger and stop event handles. Handles are opaque
64-bit SDK values, not Host pointers. The binding is used only while HCOMM
builds the native definition; it is never uploaded as launch data. Version 1
does not admit adapter-specific extra registration arguments.

The Host event handles are deliberately separate from the device event
addresses stored in `SimplerCcuMissionDescriptorV1`. A handle is consumed by
the native definition at registration. A device address is consumed by the
triggering AIV or by Host stop/final-wake logic.

## Launch word layout

After registration, `HcommCcuGetTaskArgsNum` must report exactly
`SIMPLER_CCU_TASK_ARGUMENT_COUNT` (10) words. The adapter passes the bytes of
`SimplerCcuLaunchArgumentsV1` directly to `HcommCcuKernelLaunch`. Native code
loads these 64-bit words in this fixed order:

| Index | Field |
| ----- | ----- |
| 0 | `layout_version` |
| 1 | `run_generation` |
| 2 | `mission_id` |
| 3 | `run_control_addr` |
| 4 | `run_control_token` |
| 5 | `parameter_addr` |
| 6 | `parameter_token` |
| 7 | `completion_slot_addr` |
| 8 | `completion_slot_token` |
| 9 | `reserved` (must be 0) |

The address/token pairs are per-run launch values. They must not be captured in
a domain-cached registration. The parameter size is fixed by the kernel record;
the registration adapter obtains each access token for the matching address and
size before launch.

## Run and mission records

Each run owns one `SimplerCcuRunControlV1` per mission. Its state moves in this
direction:

```text
INIT -> LAUNCHED -> READY -> EXITING -> EXITED
                       \-> FAILED
```

`HcommCcuKernelLaunch` returning is not READY evidence. A native kernel may
publish READY only after its task arguments, trigger gate, stop gate, and run
memory are usable. A blocked `EventWait` must be woken through the stop gate or
the agreed final wake value; changing `stop_requested` in GM alone cannot wake
it.

`SimplerCcuMissionDescriptorV1` identifies the run generation, mission,
kernel, group, parameter region, run control, completion slot, and trigger/stop
gates. It carries the run-control, parameter, and completion access tokens that
also appear in the launch words. Address fields are integer device addresses.
They are not Host pointers.

## Completion publication

Every mission has a separate `SimplerCcuCompletionSlotV1`. Operations on one
mission use monotonically increasing 64-bit `operation_sequence` values. The
producer writes status and the native error, makes result data visible, and
commits `completed_sequence` last.

The TMR token contract is:

| Token field | Value |
| ----------- | ----- |
| `addr` | address of the mission's completion slot |
| `backend_cookie` | expected 64-bit operation sequence |
| `engine` | CCU |
| `expected_value` | `0` (reserved) |

Polling compares the complete 64-bit sequence. A value below the expected
sequence is pending. An equal value uses the slot status to return success or
failure. A greater value is a protocol error, not success.

Operation completion releases that operation's dependency. Mission EXITED is
separate evidence and is required before run resources can be reclaimed.

## Version-1 concurrency limit

One mission may have at most one unfinished operation at a time. Its previous
TMR condition must complete and retire before another producer triggers that
mission.

Version 1 does not support multiple producers for one mission, trigger
queueing, automatic parameter snapshots, or a multi-slot completion protocol.
Different missions have different control blocks, gates, and completion slots
and may operate independently. Violating the single-unfinished-operation rule
has no defined queueing or recovery behavior.

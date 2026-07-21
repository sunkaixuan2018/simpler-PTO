# Containing A2/A3 SDMA stream teardown after an AICore fault

**Date**: 2026-07-21
**Verdict**: contained for non-SDMA workloads; full recovery after SDMA use is
deferred pending a CANN runtime-and-driver fix

## Question

Can simpler keep PTO-ISA async SDMA available while avoiding the roughly
five-minute cleanup stall caused by its 48 device-only STARS streams after an
AICore fault?

The regression became universal when the runtime started provisioning the
workspace at every Worker initialization. The same risk was already latent in
the communication path: its first window/domain allocation provisioned SDMA on
each communication handle, then cached that manager for the handle's lifetime.

## What was tried

The `aicore_op_timeout` hardware test was made deterministic by observing the
AICore failure before AICPU cleanup, then timed through `Worker.close()`.
The following alternatives were tested or traced:

- no SDMA workspace versus the pinned PTO-ISA manager at `83d01313`;
- destroying the manager's streams before device reset;
- aborting the device-only streams before destroying them;
- ordinary streams (`flags=0`) and vector-core streams (`0x1000`);
- force stream destruction, stream clear, force/non-force device reset,
  separate ACL contexts, and a helper-process ownership model by following the
  CANN 9.0 runtime and driver source paths.

## Result

| Setup | Fault surfaced | Cleanup/reset | End-to-end |
| ----- | -------------- | ------------: | ---------: |
| No SDMA workspace | `507046` | ~0.3 s | ~9 s |
| 48 device-only SDMA streams | `507015` | ~306 s | ~318 s |

The delay is one 300,000 ms remote TRS event timeout, not 48 cumulative
timeouts. Once the device is `DEV_RUNNING_DOWN`, CANN's `Stream::~Stream()`
still calls `FreeLogicCq()`. That reaches
`NpuDriver::StreamUnBindLogicCq()` → `halResourceConfig()` → the remote TRS
synchronous event. The following 47 stream releases fail immediately after the
first event times out.

Moving destruction before reset only moves the wait. Stream abort rejects the
CP-process streams. Force destruction still reaches the same C++ destructor.
Both reset variants release the same primary context and streams.

Ordinary and vector-core streams create successfully but the AICPU STARS query
fails with `507018`. This is required by the hardware contract: ACL translates
`ACL_STREAM_DEVICE_USE_ONLY` (`0x20`) to runtime
`RT_STREAM_CP_PROCESS_USE` (`0x800`), which allocates CP-local SQ/CQ/register
resources. PTO's AICore code writes SQEs and rings those registers directly;
host-local stream mappings cannot replace them.

## Why not (now)

There is no supported application-level API that both provides the CP-local
SDMA queues and bypasses their synchronous remote teardown. A complete CANN
fix needs both:

1. Skip remote logic-CQ unbind/free when the device is already down, matching
   the existing guard around stream-ID release (and audit `DavidStream`).
2. Wake or fail outstanding remote TRS events when the CP destination dies, so
   teardown cannot lose the state-check race and wait for the full timeout.

The simpler-side containment is demand-driven provisioning. A kernel declares
`required_dma_workspaces=[DmaWorkspaceKind.SDMA]` when its `CoreCallable` is
built. The runtime unions all child declarations for the parent callable,
acquires the missing workspace immediately before that callable's first run,
and republishes the device configuration so kernels receive the address through
`GlobalContext::dma_workspace`. Registration and `Worker.init()` leave the
workspace absent, and communication domains no longer create SDMA streams or
transport the address through `CommContext`. This declaration is accepted only
by the a2a3 onboard `tensormap_and_ringbuffer` path that implements both the
provider and `GlobalContext` injection; unsupported runtimes/platforms reject
it during registration rather than creating an unusable workspace.

Therefore ordinary workloads retain fast fault recovery, including Workers
that merely register an unused SDMA callable. A fault after genuine SDMA use
still requires the CANN fix above.

## Containment validation

- The AICore-timeout regression, including `Worker.close()`, completed in
  12.6 seconds and refused a subsequent SDMA-declaring callable before any
  workspace provisioning.
- `prefetch_async_demo` completed twice in one process while an extra reference
  kept `libhost_runtime.so` loaded. Each recreated Worker provisioned a fresh
  manager, proving reset invalidation does not return stale stream handles.
- The two-device `sdma_async_completion_demo` completed with bit-exact output
  on both ranks after moving the workspace from `CommContext` to callable
  declaration and `GlobalContext` injection.

## When to reconsider

Retest full fault recovery when a CANN package includes the runtime guard and
driver event-cancellation changes. The acceptance pair is: the real
`prefetch_async_demo` still uses SDMA successfully, and an AICore fault after
that provisioning cleans up in seconds rather than 300 seconds.

## References

- [simpler issue #1425](https://github.com/hw-native-sys/simpler/issues/1425)
- [CANN runtime v9.0.0](https://gitcode.com/cann/runtime/tree/v9.0.0):
  `stream.cc`, `stream_sqcq_manage.cc`, `npu_driver_res.cc`,
  `coprocessor_stream.cc`, `context_manage.cc`
- [CANN driver v9.0.0-rc.1](https://gitcode.com/cann/driver/tree/v9.0.0-rc.1):
  `trs_sqcq.c`, `trs_interface.c`, `trs_master_event.c`
- [Ascend memfabric's device-only SDMA stream setup](https://github.com/Ascend/memfabric_hybrid/blob/004d9317289fe99bd6bf13def0500b3fa3795ccc/src/hybm/csrc/transport/device/aiv_sdma_transport_manager.cpp)
- PTO-ISA `83d01313`: `sdma_workspace_manager.hpp`,
  `sdma_async_intrin.hpp`, `sdma_types.hpp`

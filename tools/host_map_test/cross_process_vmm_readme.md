# Independent-process VMM visibility validation

## Objective

Validate the proposed replacement for the fork-based experiment:

1. Start the L2 VMM owner and L3 importer as independent `exec` processes.
2. Let L2 allocate VMM physical memory and export a shareable handle.
3. Let L3 import the handle and create its own process-local VMM mapping.
4. Compare the existing ACL-copy publication path with CPU stores through
   `halHostRegister` on the imported mapping.
5. Use an AICPU observer for L3-to-L2 visibility and a completion counter for
   L2-to-L3 visibility.

The coordinator does not initialize ACL. It only exchanges metadata and
orders the two child processes over Unix socket pairs. The L2 and L3 PIDs in
the JSON result prove that the driver state is not inherited through `fork`.

## Implementation

The executable probe is:

```text
tools/host_map_test/cross_process_vmm_visibility.py
```

It uses the production VMM helpers:

- `_l3_child_onboard_region_create`: allocate, map, and export in L2.
- `_worker_host_mapped_region_import_onboard`: import, reserve a local VA,
  map, and set access in L3.

The exported value is an Ascend VMM driver shareable handle created with
`ACL_MEM_HANDLE_TYPE_NONE` and PID validation disabled. It is a 64-bit driver
handle, not a POSIX file descriptor. This is also how the repository's HCCL
IPC implementation exchanges the handle through announce files.

## Reproduction

Run each case as an exclusive device job. `task-submit` injects the selected
device through `--device`.

```bash
task-submit --device auto --device-num 1 --max-time 120 --run \
  './tools/host_map_test/cross_process_vmm_visibility.py \
  --case acl-copy-import-control --timeout 30'

task-submit --device auto --device-num 1 --max-time 120 --run \
  './tools/host_map_test/cross_process_vmm_visibility.py \
  --case host-register-import --timeout 30'
```

## myserver result

- Date: 2026-08-13
- Platform: `a2a3`
- Device reported by the last valid precheck cache:
  `Ascend910_9392 / Ascend910_93`
- Tested commit: `9ef64be01b20f9487f9041df5c183642ccac1b84`
- Runtime: `tensormap_and_ringbuffer`
- Device selected by the queue: `3`
- Evidence root:
  `/data/sunkaixuan/skx_log_output/cross_process_vmm_9ef64be0_20260813/`

The live architecture precheck could not query `npu-smi` because DCMI returned
`-8005`. The existing per-user architecture cache matched `a2a3`, and all
device work was still submitted through the exclusive device queue.

- ACL-copy import control: PASS. L2 PID `1339701`, L3 PID `1342807`,
  AICPU completion `1`, and L3 completion `1`.
- Host-register import, run 1: FAIL. L2 completion `0xFFFFFFFE` and
  L3 completion `0`.
- Host-register import, run 2: FAIL with the same values as run 1.
- Host-register import, run 3: FAIL with the same values as run 1.

Task IDs:

- ACL-copy control: `task_20260813_035928_133922223472`
- Host-register run 1: `task_20260813_035942_135334326681`
- Host-register run 2: `task_20260813_035958_13658967437`
- Host-register run 3: `task_20260813_040003_137135617009`

## Interpretation

Explicit cross-process VMM export/import works. The ACL-copy control proves
that L3 and L2 refer to the same physical allocation and that AICPU can read
L3's publication and write a completion back.

`halHostRegister` also returns success for the L3 imported VMM VA, and the L3
CPU reads back its own payload and tail stores. However, in all three runs the
AICPU observer sees the tail as unpublished and writes the timeout sentinel
`0xFFFFFFFE`. L3's host mapping also does not see that device-side completion
and remains `0`.

Therefore, independent process initialization fixes the inherited driver
state problem, but it does not make CPU stores through `halHostRegister` on an
imported VMM mapping coherent with AICPU. The production implementation must
keep the VMM plus ACL-copy path on this driver version. Enabling the zero-copy
host-register path would silently lose high-frequency signals.

The next useful experiment requires a driver-supported cross-process SVM API
or an explicit cache/coherency operation documented for imported VMM memory.
Without one of those, process-model changes alone cannot make this path safe.

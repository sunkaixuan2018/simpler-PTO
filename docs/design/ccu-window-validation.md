# CCU Allocated Window Validation

Validated on 2026-11-11 with A5, Linux aarch64, CANN 9.3.0 and the pinned
PTO-ISA checkout. Native module ABI remains V2. This evidence concerns TMR;
the HBG submit wrapper was compiled but has no hardware acceptance here.

## Data Path

`allocate_domain(workspace_engines=("ccu",))` allocates W, obtains its CCU token,
exchanges owner addresses/ranges/tokens and uploads a rank-ordered descriptor
table referenced by CommContext. Allocation does not register or launch kernels.
Callables retain the independent registration/cache and per-L2 launch lifecycle.

AIV submits window references and scalars. Runtime validates the requested byte
ranges, resolves addresses and tokens from the domain, and publishes them to P
before triggering. The native kernel loads the parameters and executes CCU
`Write` through the HCOMM channel, waits for DMA completion, synchronizes with
its peer, and records done. A dependent AIV copies its local W for Host checking.
No AIV writes the peer window in this test. P's token and W's token are separate.

## Prototype Results

The tested feature source tree is `77923019df366d137f0fe8dc8e4244e3301929c4`,
on base `da46d7a91`. The local and remote Git trees matched before publication.

- Two-rank W transfer, 16 Host runs: passed; devices 2,6; task
  `task_20261111_141025_15066207996`.
- Two-rank W transfer, 1000 Host runs: passed in 85.97 seconds; devices 5,6;
  task `task_20261111_141637_185337911056`.
- Domain-zero binding and ordinary/replacement callable regression, 16 runs:
  all three cases passed in the latter task; exit 0 and device locks released.
- Focused Python regression: 209 passed, 2 skipped, 13 deselected.
- CCU device submission, workspace, module, registration and run lifecycle
  C++ suites: all five passed.
- Linux changed-file pre-commit: passed, including clang-tidy and pyright.

The test uses a 4 MiB W per rank and checks the complete 32 KiB named data
region after each run. Each rank supplies different payload bytes; two kernels
write to separate peer ranges. Transfer lengths vary from 64 to 1024 bytes and
offsets vary in 64-byte steps. Checking the complete region detects incorrect
payloads and changes outside the intended destinations. Halfway through the
test, W is released and freshly allocated; other runs reuse retained W.

These are separate Host `worker.run` calls, not repeated submissions inside one
L2 execution. The existing binding supplies a ring channel; all-peer channels,
eight-rank collectives, Host-source W and same-mission request queues are not
validated by this test. Earlier local scalar output and cross-rank notification
tests did not establish this W data path. The previously observed unexplained
extended-run CCU-done timeout is not disproved by these successful runs.

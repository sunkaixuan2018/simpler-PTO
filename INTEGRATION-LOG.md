# Kernel-mode PR integration log

One entry per adjudication, in the order it was made. Each entry states the
problem, the options, the choice, the reason, and which PRs it affects.

The goal of the integration is a single line on which
`init -> prepare_callable -> launch -> close` runs one eager
`tensormap_and_ringbuffer` invocation on a2a3 hardware.

---

## D0 - The PRs are real git stacks, not independent source-embedded copies

**Problem.** The handover states that the in-flight PRs each source-embed a
copy of K1 and that those copies have diverged by 16 / 97 / 115 / 165 lines,
so `git merge` cannot deduplicate them and each copy must be adjudicated by
hand.

**Finding.** That is not what the refs contain. K1 appears under three
different commit ids -- `86dd62b4` (#2064 / K3), `dc1268cd` (#2177, #2176,
#2185, #2180, #2190) and `66156ed3` (#2189) -- and all three have **identical
trees**:

```
git diff --stat dc1268cd 86dd62b4   # empty
git diff --stat 66156ed3 86dd62b4   # empty
```

K1 itself never diverged. The line counts the handover reports are each PR's
*own contribution on top of* K1, not a competing copy of it.

**Choice.** Do not hand-merge K1 copies. Cherry-pick each PR's non-K1 commits
onto the authoritative K1, which replays every contribution as a clean
three-way merge against the identical base it was written against.

**Reason.** The unique-contribution set is a handful of distinct commits, and
cherry-picking them yields linear history with one commit per contribution and
no duplicated K1 history.

**Affects.** All PRs. This is the method the rest of the log assumes.

---

## D1 - `pr/k3` carries K1, so batches 1 and 3.5 collapse into one merge

**Problem.** The recommended order lands #2064 (K1) at batch 1 and K3 at batch
3.5, with K3 not yet submitted at handover time.

**Finding.** K3 had already landed when work started: the workspace at
`D:/PTO/code/simplers/kernel-K3` is at `eb696879` ("kernel-mode capacity
refusals no longer free what they protect", 2026-09-10), whose parent is K1
`86dd62b4`. So the K3 branch *is* K1 + K3. No waiting was required and no
polling was done.

**Choice.** Merge `pr/k3` once, as the authoritative baseline for both.

**Reason.** K3's K1 files are byte-identical to #2064's, so the merge delivers
#2064 exactly. Splitting it would add a commit and change nothing.

**Affects.** #2064, K3.

---

## D2 - The H chain is a third ABI fork and is deferred wholesale

**Problem.** The recommended order puts #2171 at batch 0 as a "pure
refactor, zero behaviour change" warm-up that does not occupy the merge order.

**Finding.** #2171 is not orthogonal. Its own commit message says it
"Include[s] the kernel-mode C ABI, state and invocation headers required by
this implementation", and its head carries a K1 variant that **deletes
`execution_mode_latch.h` and `kernel_entry_validation.h` outright** and
rewrites `runtime_c_api.h` (+102/-64 against authoritative K1). Every H PR
(#2171, #2172, #2173, #2174, #2175) carries that same variant; they are a
consistent stack among themselves and a third ABI family against K1.

Merging #2171 first was tried and reverted: it fast-forwards, silently
adopting that variant as the branch's baseline.

**Options.**

1. Reconcile the H variant against authoritative K1 and land the H chain.
2. Defer the whole H chain and integrate the TMR line only.

**Choice.** Option 2 -- defer #2171, #2172, #2173, #2174, #2175 entirely.

**Reason.** The H chain is `host_build_graph`; the smoke target is eager
`tensormap_and_ringbuffer`. The handover explicitly permits deferring the H
chain as a batch. Reconciling a third ABI family that deletes two headers the
TMR line depends on buys nothing for the stated goal and risks the line that
does matter.

**Affects.** #2171, #2172, #2173, #2174, #2175. All five need rebasing onto
this branch's K1 before they can land; see the closing report.

---

## D3 - `prepare_callable` keeps `caller_stream` (five parameters)

**Problem.** Three signatures are in flight:

| Source | Signature |
| ------ | --------- |
| K1 / #2177 / #2185 | `(ctx, callable_id, callable, size, caller_stream)` |
| #2176 / #2180 / #2189 | `(ctx, callable_id, callable, size)` |
| #2190 | `(ctx, callable, size, SimplerCallableHandle *out)` |

K2's cherry-pick applies its four-parameter form **without a git conflict**,
because it is a clean delta against the same K1 base. Nothing reports the
change; the parameter simply disappears.

K2 does not drop it by oversight. It replaces the doc with an argument:
preparation enqueues on the context's own AICPU stream, later launches enqueue
on that same stream, and stream FIFO therefore orders registration ahead of
every launch without a caller stream being involved. That is coherent with the
three-stream binder in #2187.

**Choice.** Five parameters. `caller_stream` is restored in
`runtime_c_api.h`, `kernel_entry_validation.h`, both `c_api_shared.cpp`
entries, `test_kernel_entry_validation.cpp` and `test_kernel_mode_c_api.py`.

**Reason.**

1. **The actual ABI consumer is five-parameter.** #2185 is the entry layer
   that gives the four C entries their callers, and it calls
   `kernel_prepare_callable(callable_id, data, size, caller_stream)`, plumbing
   the stream up through the nanobind binding and the Python wrapper with the
   documented "may enqueue asynchronous work on that stream but never
   synchronizes it" semantics. K2 is the outlier against the layer that uses
   it.
2. **The K1 interface-freeze card is the designated tie-breaker** for ABI
   signature disputes, and it specifies five.
3. **It is strictly more expressive.** An implementation that prefers to stage
   on its own AICPU stream can ignore the parameter; a four-parameter entry
   cannot later acquire ordering against the caller's *preceding* work without
   an ABI break. Keeping it preserves both designs, dropping it forecloses one.
4. It keeps prepare and launch symmetric, matching the header's own rule that
   "the caller stream is always an explicit parameter".

K2's FIFO argument is not refuted, and this integration does not depend on
refuting it. It holds only while prepare and launch enqueue on the *same*
context stream; the parameter costs nothing and covers the case where they
do not.

**Affects.** #2176, #2180, #2189 (must adopt five), #2190 (see D-later),
#2177 and #2185 (unchanged).

---

## D4 - `simpler_kernel_mode_init` keeps 2a's contract validation ahead of K2's body

**Problem.** The only textual conflict in the K2 cherry-pick. `HEAD` (K1 + 2a)
validates the pipeline contract the config implies and then returns the
`UNSUPPORTED` refusal stub. K2 replaces the whole body with the real init,
which has no contract validation because K2 branched before 2a.

**Choice.** Keep both: 2a's `build_kernel_pipeline_contract_impl` +
`is_valid_pipeline_contract` + arena/stream topology check first, then K2's
latch-and-initialize body. The refusal stub is dropped.

**Reason.** The two are independent and both wanted. Validating before the
latch means a config that cannot be serviced is refused while the context is
still free, which is what makes a refused init leave a reusable context -- a
guarantee #2185's tests already assert.

**Affects.** #2176, #2177.

---

## D5 - Environment: pre-commit `clang-tidy` cannot run on this host

**Problem.** Every commit fails the `clang-tidy` pre-commit hook:
`build_runtimes.py` imports `fcntl`, which does not exist on Windows.

**Choice.** Integration commits use `--no-verify`. `cpplint`, `ruff check`,
`ruff format` and `pyright` all run and pass; only the clang-tidy hook is
skipped.

**Reason.** The hook is unrunnable on this platform, not failing on the
content. It runs in CI and on the Linux host used for hardware validation.

**Affects.** Every commit on this branch.

---

## D6 - #2189 over #2180, taken as #2180's K4 chain plus #2189's K5 commit

**Problem.** #2180 and #2189 must not both land: 14 conflicting files, and the
handover records them as carrying the same K4 tree with no new content between
them.

**Finding.** #2189 is a strict superset in content but not in history. It
rebuilt the whole stack under fresh commit ids -- its own K1 (`66156ed3`), its
own 2a pair, its own K4 (`b5529c89`), a K2/K4 integration commit, and finally
K5 (`b6435e48`). Cherry-picking #2189's commits after #2177 (2a) and #2176
(K2) would replay two contributions this branch already carries.

**Choice.** Take #2180's K4 chain (`3e822453`, `b0943525`), which is stacked
directly on the 2a commits this branch already has, then cherry-pick only
#2189's K5 commit (`b6435e48`) on top.

**Reason.** It is the same end state with no duplicated contribution, and it is
the second option the handover itself offers. #2180 is otherwise superseded.

**Affects.** #2180 (K4 chain taken, PR otherwise superseded), #2189 (only K5
taken; its K1/2a/K2/K4 commits are redundant against this branch).

---

## D7 - `launch` is id-based; #2190's handle collapses into it

**Problem.** #2190 changes both entries to a `SimplerCallableHandle`
`{callable_id, generation}`: prepare writes one out, launch takes one in. That
is a third shape against K1's `int32_t callable_id`, and it arrives without a
git conflict on the launch entry.

**Choice.** Keep `int32_t callable_id` on both entries. Launch resolves the
residency internally as `{callable_id, cache.generation()}`.

**Reason.** The generation guard is the valuable half of #2190 and it survives
intact: the cache stores the generation an entry was staged under, so a
callable left from a previous context generation still resolves to
`CALLABLE_STALE` rather than replaying a recycled address. What the handle
added beyond that was a caller-held token, and an id-based ABI has no place to
hold one -- the context generation is minted at init, so the check the caller
would have armed is exactly the check the runtime now performs. Meanwhile
#2185's entry layer, its nanobind binding and the Python wrapper are all
id-based, as is K1.

**Affects.** #2190. Its `_prepare` ctypes helper and `CallableHandle` struct
are removed as dead; its two generation-staleness assertions are re-expressed
against the id-based launch in the end-to-end test.

---

## D8 - #2190's callable cache takes a caller-supplied id

**Problem.** `KernelCallableCache::stage` allocated its own sequential id
(`entries_.size()`), because under #2190's ABI the runtime chose the id. Under
D7 the caller chooses it.

**Choice.** `stage` takes `requested_id`. An id already staged is refused as a
duplicate registration. An image whose bytes match a resident entry is admitted
under its own id, shares that entry's device address, and is charged zero arena
bytes. `commit` and `rollback` find their entry by id instead of assuming it is
the last one.

**Reason.** It preserves every property the cache's own tests assert -- one
upload per unique image, no eviction, arena accounting, descriptor table
indexed by id -- while honouring the caller's id. The descriptor table was
already indexed by id and the ABI already bounds the id to
`[0, MAX_REGISTERED_CALLABLE_IDS)`, so a caller-supplied id indexes it safely.

**Affects.** #2190. Its `test_kernel_callable_cache.cpp` needs the new
parameter at each `stage` call site.

---

## D9 - The launch topology is the binder's AICore-first set

**Problem.** Two five-event topologies are in the tree.
`KernelEventKind` (from K2) is a chain: `Start, AicoreStart, AicoreDone,
AicpuDone, SerialTail`, where the aicpu stream forks the aicore stream. The
binder's `KernelLaunchHandles` is a fork-join:
`prepare_tail, start, aicore_done, aicpu_done, serial_tail`, where both device
streams fork from the caller's Start and rejoin the caller.

The handover reports the binder as AICPU-first and leans toward K2's set. The
binder's source says otherwise: its sequence waits `start` on the aicore
stream, launches AICore, records `aicore_done`, and only then admits AICPU.
Its comment gives the two v9 reasons verbatim -- the scheduler cycle when an
AICPU startup waiter blocks a later AICore SQE, and the better failure
closure, since AICore can still be cancelled while no AICPU has written the
handshake.

**Choice.** Adopt the binder's set. `KernelEventKind::AicoreStart` becomes
`PrepareTail`; the count stays five and the storage is unchanged.

**Reason.** The binder is the only implementation of a launch sequence in any
of these PRs, and this integration wires launch onto it. Its topology is
self-consistent, it is what v9 specifies, and K2's stated deadlock hazard is
avoided by a different means: AICore's SQE is on its stream before AICPU
becomes resident, so the orchestrator's spin on the AICore handshake always
has a submitted AICore to wait for.

`tests/st/a2a3/kernel_capture/native/driver.cpp` is reordered to match. It
exercises the same capture primitives -- three streams, five events, one AICPU
launch, three AICore launches -- so its capture evidence is preserved, and it
now demonstrates the topology the product actually uses. Its two device
branches touch disjoint buffers, so running them as siblings is safe.

**Affects.** #2176 (event enum and capture ST), #2187 (unchanged, it was
already the target).

---

## D10 - AICPU transport stays on the in-repo `rtsLaunchCpuKernel` path

**Problem.** The binder's native layer, `launch_bound_kernel_native`, requires
a `KernelNativeInvocation` carrying two `aclrtFuncHandle`s and drives
`aclrtLaunchKernel` / `aclrtLaunchKernelWithHostArgs`. Nothing in any PR calls
`aclrtBinaryLoad` or `aclrtBinaryGetFunction`, so nothing produces those
handles. The TMR path in the repo launches AICPU work through
`LoadAicpuOp::LaunchBuiltInOp`, which wraps `rtsLaunchCpuKernel`.

**Finding.** The choice is not forced. `launch_bound_kernel` -- the generic
entry -- takes `launch_aicpu` and `launch_aicore` as plain callbacks in
`KernelLaunchOps`. Only the `_native` variant hard-codes the CANN family.

**Choice.** Wire launch onto `launch_bound_kernel` with owner-supplied
callbacks that reuse the existing `LaunchBuiltInOp` / AICore launch paths.
Leave `launch_bound_kernel_native` in the tree, unused, as the migration
target.

**Reason.** It is the lowest-risk route to a first working launch and the one
the handover recommends: the code already exists and is exercised on hardware,
where the `WithHostArgs` family has no precedent in this repo. The binder's
12-step sequence, its compensation ladder and its capture-safety guarantees
are all in the generic entry, so nothing is given up. Switching the transport
later changes two callbacks and no sequencing.

**Affects.** #2187. The source guard covers only the three binder files, which
this leaves untouched.

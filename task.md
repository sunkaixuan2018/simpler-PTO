# Task Record

## Current Task

- Summary: Implement mailbox and MPI dispatch for MPI worker groups on PR #1623.
- Status: completed
- Local Branch: `skx/mpi-mailbox-broadcast`
- Delivery Branch: `origin/use-mpi-and-remove-socket`
- Validation Hosts: servers 37 and 35 (external handoff required)
- Baseline PR Head: `d3d17b21593e642c6c5f3956ab163589cdb2bbe1`
- Latest Tested Tree: final local branch tree; final commit SHA is reported at handoff
- Last Updated: 2026-08-01

## Work Completed

- Read the repository instructions, always-on rules, and relevant workflows.
- Refreshed `upstream/pr-1623` to GitHub head `d3d17b21` and created an
  independent branch without rebasing or modifying `add-mpi-run`.
- Used this worktree's project `.venv`.
- Added a versioned named shared-memory mailbox protocol with explicit READY,
  accepted, done, failed, shutdown, terminal, sequence, target, and payload
  semantics.
- Added the C++ mailbox transport and endpoint. A complete MPI group submission
  becomes one per-rank request; directed and subset submissions remain directed.
- Replaced the MPI group's Simpler command/health TCP activation with a rank-0
  mailbox and a single-threaded MPI dispatcher using separate dispatch and
  Global CommDomain communicators.
- Added ranked result aggregation, terminal timeout behavior, direct `mpirun`
  process-group monitoring, and mailbox/manifest cleanup.
- Preserved the ordinary non-MPI Remote L3 socket path.
- Added unit tests, a device-free real-endpoint smoke, and the A3 2x2 compute
  plus global TLOAD smoke.

## Validation

- Local Windows named-mailbox unit tests:
  `10 passed in 0.56s`.
- Staged-file hooks passed: headers, English-only, large files, EOF,
  whitespace, markdownlint, Ruff, formatting, and Pyright.
- Clang-tidy is unavailable on this Windows host because the repository setup
  imports POSIX-only modules.
- The tests that import the Linux C++ extension cannot be collected on this
  Windows host.
- Single-host real `mpirun`, two-host MPI, and A3/NPU smoke are **NOT
  VALIDATED** here. Per user direction, `myserver` is not a validation host.
- The official MPI and A3 validation must run on servers 37 and 35 using the
  separately delivered agent prompt. No result may be called passing until
  its log is returned.

## Protocol Conclusions

1. MPI group operations never fall back to Simpler TCP. A mailbox/MPI failure
   terminates the group.
2. Full worker-id sets in `submit_next_level_group` use one `PER_RANK`
   mailbox request. One worker ID means directed execution; a subset remains
   an ordered set of directed requests.
3. Remote controls remain uniquely numbered 1 through 17. Local control 18
   remains reserved for committed-device-memory handling.
4. Bare host pointers, child-memory pointers without a transferable sidecar,
   and other unsafe remote addresses are rejected before dispatch.

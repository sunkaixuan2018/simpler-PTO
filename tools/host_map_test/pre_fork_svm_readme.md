# Pre-fork SVM visibility probe

This probe tests whether a host mapping created by `halHostRegister` in a
parent process remains usable after `fork()`. It covers both CPU access in the
child and AICPU access submitted by the child.

## Tested topology

1. The parent initializes `ChipWorker` and allocates an ordinary device region.
2. The parent calls `halHostRegister` before `fork()` and publishes a payload
   through the returned host virtual address.
3. The child either reads the inherited host address, reuses the inherited
   `ChipWorker`, or creates a new `ChipWorker`.
4. The AICPU observer reads the payload and writes a completion value that the
   parent checks through the registered host address.

The AICPU observer uses raw volatile loads and stores. A small AIV child is
also submitted so the ordinary task lifecycle completes independently of the
address under test.

## Reproduction

Build the editable package after checking out the target commit. The package
records the source commit in its compiled bindings, so it must be rebuilt even
when a new commit changes only scripts or documentation.

```bash
export REPO=/path/to/simpler
cd "$REPO"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-build-isolation -e .
PYTHONNOUSERSITE=1 .venv/bin/python \
  simpler_setup/build_runtimes.py --platforms a2a3
```

Run each case as a separate device task. `task-submit` supplies the selected
device as the probe's `--device` argument.

```bash
export LOG_ROOT=/path/to/logs/pre_fork_svm
export PATH="$REPO/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1

for case_name in \
  observer-noop-control \
  acl-copy-control \
  same-process-owner \
  fork-inherited-owner \
  fork-aicpu-only-owner \
  fork-child-reinit-owner
do
  case_dir="$LOG_ROOT/$case_name"
  mkdir -p "$case_dir"
  task-submit --device auto --device-num 1 --max-time 120 \
    --env PATH --env LD_LIBRARY_PATH --env PYTHONPATH \
    --env ASCEND_HOME_PATH --env PYTHONNOUSERSITE \
    --run "$REPO/tools/host_map_test/pre_fork_svm_visibility.py \
      --case $case_name --timeout 30 --output $case_dir/result.json"
done
```

The three fork cases intentionally return a nonzero exit code when the current
driver does not support the tested inheritance path. Run them independently if
the calling shell uses `set -e`.

## Validated result

Validated on a2a3 hardware at commit `f8191dc6130c66387044bbf9efcb7d0088854cdd`.

| Case | Result | Task ID | Evidence |
| ---- | ------ | ------- | -------- |
| `observer-noop-control` | Pass | `task_20260813_024734_107209818379` | AICPU lifecycle completed without touching the candidate address. |
| `acl-copy-control` | Pass | `task_20260813_024744_108340513804` | AICPU observed the copied publication and the host observed completion `1`. |
| `same-process-owner` | Pass | `task_20260813_024752_10937315217` | Host mapping and bidirectional AICPU visibility worked. |
| `fork-inherited-owner` | Fail | `task_20260813_024809_11164074236` | Child CPU read of the inherited host VA terminated with `SIGSEGV`. |
| `fork-aicpu-only-owner` | Fail | `task_20260813_024827_115623629328` | Inherited worker failed with ACL `507899` and an executed-times SVM allocation error. |
| `fork-child-reinit-owner` | Fail | `task_20260813_024709_104033817395` | New child worker terminated with `SIGSEGV`; completion remained `0`. |

The parent received an identity mapping in every registered case: the returned
host address and device address had the same numeric value. This identity does
not make the mapping process-independent. The child still lacks a usable CPU
mapping and driver/runtime context after the fork.

## Conclusion

The desired cross-process SVM register path is not supported by fork
inheritance in the tested driver/runtime stack. `halHostRegister` works for
host-to-AICPU counter visibility when registration, submission, and completion
handling stay in the same process. It does not currently replace the existing
VMM plus ACL copy path when L3 owns the registration and L2 performs device
submission in another process.

A supported zero-copy design needs an explicit driver contract for
cross-process SVM mapping or an export/import API for the registered mapping.
Until such an API is available, keep the SVM owner and device submission in one
process, or keep the current VMM plus ACL copy transport across the process
boundary.

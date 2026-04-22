#!/usr/bin/env bash
# Benchmark wrapper: run examples on hardware and summarize profiling metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EXAMPLE="$PROJECT_ROOT/examples/scripts/run_example.py"

# ---------------------------------------------------------------------------
# Examples to benchmark and their case lists, per runtime.
# Key   = directory name under tests/st/<platform>/<runtime>/
# Value = comma-separated case names to run (empty string = run DEFAULT_CASE)
# ---------------------------------------------------------------------------

# --- tensormap_and_ringbuffer ---
declare -A TMR_EXAMPLE_CASES=(
    [alternating_matmul_add]=""
    [benchmark_bgemm]=""
    [paged_attention_unroll]="Case1,Case2"
    [batch_paged_attention]=""
)
TMR_EXAMPLE_ORDER=(
    alternating_matmul_add
    benchmark_bgemm
    paged_attention_unroll
    batch_paged_attention
)

# --- aicpu_build_graph ---
declare -A ABG_EXAMPLE_CASES=(
    [paged_attention_unroll]="Case1,Case2"
)
ABG_EXAMPLE_ORDER=(
    paged_attention_unroll
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DEVICE_ID=0
ROUNDS=1
PLATFORM=a2a3
RUNTIME=tensormap_and_ringbuffer
VERBOSE=0
PREFETCH_MODE=compare
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE_ID="$2"
            shift 2
            ;;
        -n|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -r|--runtime)
            RUNTIME="$2"
            shift 2
            ;;
        --prefetch-mode)
            PREFETCH_MODE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            cat <<'USAGE'
benchmark_rounds.sh — run all examples and report per-round timing from device logs

Usage:
  ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>] [--prefetch-mode <mode>] [-v]

Options:
  -p, --platform         Platform to run on (default: a2a3)
  -d, --device           Device ID (default: 0)
  -n, --rounds           Override number of rounds for each example (default: 1)
  -r, --runtime          Runtime to benchmark: tensormap_and_ringbuffer (default), aicpu_build_graph
      --prefetch-mode    baseline | sdma | compare (default: compare)
  -v, --verbose          Save detailed run_example.py output to a timestamped log file
  -h, --help             Show this help

All other options are passed through to run_example.py (e.g. --case).

Edit the EXAMPLE_CASES map at the top of this script to control which
examples and cases to benchmark.

Output:
  AICore Exec (host profiling derived earliest AICore start to latest AICore end).
  AICPU Sch (host profiling derived earliest dispatch to latest finish).
  Device E2E (device log derived earliest orch/sched start to latest orch/sched end).
USAGE
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$PREFETCH_MODE" in
    baseline|sdma|compare) ;;
    *)
        echo "ERROR: unknown prefetch mode '$PREFETCH_MODE'. Use baseline, sdma, or compare."
        exit 1
        ;;
esac
case "$PREFETCH_MODE" in
    compare) RUN_MODES=(baseline sdma) ;;
    *) RUN_MODES=("$PREFETCH_MODE") ;;
esac

# ---------------------------------------------------------------------------
# Verbose logging setup
# ---------------------------------------------------------------------------
VERBOSE_LOG=""
OUTPUTS_DIR="$PROJECT_ROOT/outputs"
if [[ $VERBOSE -eq 1 ]]; then
    mkdir -p "$OUTPUTS_DIR"
    VERBOSE_LOG="$OUTPUTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).log"
    echo "Verbose log: $VERBOSE_LOG"
fi
mkdir -p "$OUTPUTS_DIR"

vlog() {
    if [[ -n "$VERBOSE_LOG" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$VERBOSE_LOG"
    fi
}

# ---------------------------------------------------------------------------
# Derive arch from platform and set examples directory
# ---------------------------------------------------------------------------
EXAMPLES_DIR="$PROJECT_ROOT/tests/st/${PLATFORM}/${RUNTIME}"

# Clock frequency (MHz) for converting cycle counts to microseconds
case "$PLATFORM" in
    a2a3) FREQ=50 ;;
    a5)   FREQ=1000 ;;
    *)    echo "ERROR: unsupported platform '$PLATFORM'. Use a2a3 or a5."; exit 1 ;;
esac

# Select example cases and order based on runtime
case "$RUNTIME" in
    tensormap_and_ringbuffer)
        declare -n EXAMPLE_CASES=TMR_EXAMPLE_CASES
        EXAMPLE_ORDER=("${TMR_EXAMPLE_ORDER[@]}")
        ;;
    aicpu_build_graph)
        declare -n EXAMPLE_CASES=ABG_EXAMPLE_CASES
        EXAMPLE_ORDER=("${ABG_EXAMPLE_ORDER[@]}")
        ;;
    *)
        echo "ERROR: unknown runtime '$RUNTIME'. Use tensormap_and_ringbuffer or aicpu_build_graph."
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Resolve device log directory (mirrors run_example.py / device_log_resolver.py)
# ---------------------------------------------------------------------------
if [[ -n "${ASCEND_WORK_PATH:-}" ]]; then
    LOG_ROOT="$ASCEND_WORK_PATH/log/debug"
    if [[ ! -d "$LOG_ROOT" ]]; then
        LOG_ROOT="$HOME/ascend/log/debug"
    fi
else
    LOG_ROOT="$HOME/ascend/log/debug"
fi
DEVICE_LOG_DIR="$LOG_ROOT/device-${DEVICE_ID}"

# list_device_logs
list_device_logs() {
    if [[ ! -d "$DEVICE_LOG_DIR" ]]; then
        return 0
    fi
    (
        shopt -s nullglob
        for _log in "$DEVICE_LOG_DIR"/*.log; do
            printf '%s\n' "$_log"
        done
    )
}

find_new_device_log() {
    local pre_snapshot="$1"
    local timeout_s=15
    local elapsed=0
    while (( elapsed < timeout_s )); do
        local newest=""
        local current_logs
        current_logs=$(list_device_logs)
        while IFS= read -r _log; do
            [[ -z "$_log" ]] && continue
            if ! grep -Fxq "$_log" <<<"$pre_snapshot"; then
                if [[ -z "$newest" || "$_log" -nt "$newest" ]]; then
                    newest="$_log"
                fi
            fi
        done <<<"$current_logs"
        if [[ -n "$newest" ]]; then
            printf '%s\n' "$newest"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

parse_device_e2e_avg() {
    local log_file="$1"
    local e2e_us
    e2e_us=$(awk -v freq="$FREQ" '
    function new_round() {
        flush_round()
        round++
        min_start = 0; max_end = 0
        delete sched_seen
        delete orch_seen
    }
    function flush_round() {
        if (round >= 0 && max_end > 0 && min_start > 0) {
            results[round] = (max_end - min_start) / freq
            count++
        }
    }
    BEGIN {
        round = 0; count = 0
        min_start = 0; max_end = 0
    }
    /sched_start=/ {
        match($0, /Thread ([0-9]+):/, tm)
        tid = tm[1] + 0
        if (tid in sched_seen) new_round()
        sched_seen[tid] = 1
        match($0, /sched_start=([0-9]+)/, m)
        val = m[1] + 0
        if (min_start == 0 || val < min_start) min_start = val
    }
    /orch_start=/ {
        match($0, /Thread ([0-9]+):/, tm)
        tid = tm[1] + 0
        if (tid in orch_seen) new_round()
        orch_seen[tid] = 1
        match($0, /orch_start=([0-9]+)/, m)
        val = m[1] + 0
        if (min_start == 0 || val < min_start) min_start = val
    }
    /sched_end[^=]*=/ {
        match($0, /sched_end[^=]*=([0-9]+)/, m)
        val = m[1] + 0
        if (val > max_end) max_end = val
    }
    /orch_end=/ {
        match($0, /orch_end=([0-9]+)/, m)
        val = m[1] + 0
        if (val > max_end) max_end = val
    }
    /orch_stage_end=/ {
        match($0, /orch_stage_end=([0-9]+)/, m)
        val = m[1] + 0
        if (val > max_end) max_end = val
    }
    END {
        flush_round()
        if (count == 0) exit 1
        sum_v = 0
        for (i = 0; i < count; i++) sum_v += results[i]
        printf "%.2f\n", sum_v / count
    }' "$log_file" 2>/dev/null || true)
    if [[ -n "$e2e_us" ]]; then
        echo "$e2e_us"
        return 0
    fi
    return 1
}

parse_aicpu_exec() {
    local run_output="$1"
    local exec_us
    exec_us=$(printf "%s\n" "$run_output" | awk '
        match($0, /Total Test Time: ([0-9.]+) us/, m) { print m[1]; found = 1; exit }
        END { if (!found) exit 1 }' 2>/dev/null || true)
    if [[ -n "$exec_us" ]]; then echo "$exec_us"; return 0; fi
    return 1
}

parse_aicore_exec() {
    local run_output="$1"
    local exec_us
    exec_us=$(printf "%s\n" "$run_output" | awk '
        match($0, /AICore Span: ([0-9.]+) us/, m) { print m[1]; found = 1; exit }
        END { if (!found) exit 1 }' 2>/dev/null || true)
    if [[ -n "$exec_us" ]]; then echo "$exec_us"; return 0; fi
    return 1
}

PROFILE_AICORE_EXEC="-"
PROFILE_AICPU_EXEC="-"
PROFILE_DEVICE_E2E="-"
run_profile_once() {
    local mode="$1" kernels_dir="$2" golden="$3" case_name="${4:-}"
    local profile_cmd=(
        env "PTO_SDMA_PREFETCH_MODE=$mode"
        python3 "$RUN_EXAMPLE"
        -k "$kernels_dir" -g "$golden"
        -p "$PLATFORM" -d "$DEVICE_ID"
        -n "$ROUNDS" --skip-golden --enable-profiling
    )
    if [[ -n "$case_name" ]]; then
        profile_cmd+=(--case "$case_name")
    fi
    profile_cmd+=("${EXTRA_ARGS[@]}")

    local pre_run_logs profile_tmp profile_output profile_rc=0
    pre_run_logs=$(list_device_logs)
    profile_tmp=$(mktemp)
    "${profile_cmd[@]}" >"$profile_tmp" 2>&1 || profile_rc=$?
    profile_output=$(<"$profile_tmp")
    rm -f "$profile_tmp"
    PROFILE_AICORE_EXEC="-"; PROFILE_AICPU_EXEC="-"; PROFILE_DEVICE_E2E="-"
    if [[ $profile_rc -ne 0 ]]; then return 1; fi
    parse_aicore_exec "$profile_output" >/dev/null && PROFILE_AICORE_EXEC=$(parse_aicore_exec "$profile_output")
    parse_aicpu_exec "$profile_output" >/dev/null && PROFILE_AICPU_EXEC=$(parse_aicpu_exec "$profile_output")
    local device_log
    if device_log=$(find_new_device_log "$pre_run_logs"); then
        parse_device_e2e_avg "$device_log" >/dev/null && PROFILE_DEVICE_E2E=$(parse_device_e2e_avg "$device_log")
    fi
    [[ -n "$VERBOSE_LOG" && -n "$profile_output" ]] && echo "$profile_output" >> "$VERBOSE_LOG"
    return 0
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
PASS=0
FAIL=0

# Summary collection arrays
SUMMARY_NAMES=()
declare -A SUMMARY_LABELS_SEEN=()
declare -A SUMMARY_AICORE_EXEC=()
declare -A SUMMARY_AICPU_EXEC=()
declare -A SUMMARY_DEVICE_E2E=()

echo ""
echo "Runtime: $RUNTIME"
echo "Tests dir: $EXAMPLES_DIR"
echo "Prefetch modes: ${RUN_MODES[*]}"

for example in "${EXAMPLE_ORDER[@]}"; do
    case_list="${EXAMPLE_CASES[$example]:-}"

    EXAMPLE_DIR="$EXAMPLES_DIR/$example"
    KERNELS_DIR="$EXAMPLE_DIR/kernels"
    GOLDEN="$EXAMPLE_DIR/golden.py"

    echo ""
    echo "================================================================"
    echo "  $example"
    echo "================================================================"

    if [[ ! -f "$GOLDEN" || ! -d "$KERNELS_DIR" ]]; then
        echo "  SKIP: missing kernels/ or golden.py"
        ((FAIL++)) || true
        continue
    fi

    run_one_case() {
        local _example="$1" _kernels="$2" _golden="$3" _case="${4:-}"
        local _label="$_example"
        [[ -n "$_case" ]] && _label="$_example ($_case)"
        if [[ -n "$_case" ]]; then echo "  ---- $_case ----"; fi
        for mode in "${RUN_MODES[@]}"; do
            echo "  Mode: $mode"
            if run_profile_once "$mode" "$_kernels" "$_golden" "$_case"; then
                ((PASS++)) || true
            else
                ((FAIL++)) || true
            fi
            [[ -z "${SUMMARY_LABELS_SEEN[$_label]+x}" ]] && SUMMARY_NAMES+=("$_label") && SUMMARY_LABELS_SEEN["$_label"]=1
            SUMMARY_AICORE_EXEC["$mode|$_label"]="$PROFILE_AICORE_EXEC"
            SUMMARY_AICPU_EXEC["$mode|$_label"]="$PROFILE_AICPU_EXEC"
            SUMMARY_DEVICE_E2E["$mode|$_label"]="$PROFILE_DEVICE_E2E"
            echo "  AICore Exec (us): $PROFILE_AICORE_EXEC"
            echo "  AICPU Sch (us): $PROFILE_AICPU_EXEC"
            echo "  Device E2E (us): $PROFILE_DEVICE_E2E"
        done
    }
    if [[ -z "${case_list:-}" ]]; then
        run_one_case "$example" "$KERNELS_DIR" "$GOLDEN"
    else
        IFS=',' read -ra cases <<< "$case_list"
        for c in "${cases[@]}"; do
            run_one_case "$example" "$KERNELS_DIR" "$GOLDEN" "$c"
        done
    fi
done

# ---------------------------------------------------------------------------
# Performance Summary Table
# ---------------------------------------------------------------------------
if [[ ${#SUMMARY_NAMES[@]} -gt 0 ]]; then
    echo ""
    echo "================================================================"
    echo "  Performance Summary ($RUNTIME)"
    echo "================================================================"
    echo ""

    if [[ "$PREFETCH_MODE" == "compare" ]]; then
        _hdr=$(printf "  %-40s  %20s  %20s  %18s  %18s  %18s  %18s" "Example" "Baseline AICore Exec" "SDMA AICore Exec" "Base AICPU Sch" "SDMA AICPU Sch" "Base Device E2E" "SDMA Device E2E")
        _sep=$(printf "  %-40s  %20s  %20s  %18s  %18s  %18s  %18s" "----------------------------------------" "--------------------" "--------------------" "------------------" "------------------" "------------------" "------------------")
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %20s  %20s  %18s  %18s  %18s  %18s" "$_label" "${SUMMARY_AICORE_EXEC["baseline|$_label"]:-"-"}" "${SUMMARY_AICORE_EXEC["sdma|$_label"]:-"-"}" "${SUMMARY_AICPU_EXEC["baseline|$_label"]:-"-"}" "${SUMMARY_AICPU_EXEC["sdma|$_label"]:-"-"}" "${SUMMARY_DEVICE_E2E["baseline|$_label"]:-"-"}" "${SUMMARY_DEVICE_E2E["sdma|$_label"]:-"-"}")
            echo "$_row"
        done
    else
        _mode_name="${RUN_MODES[0]}"
        _mode_title=$(printf "%s" "$_mode_name" | tr '[:lower:]' '[:upper:]')
        _hdr=$(printf "  %-40s  %20s  %18s  %18s" "Example" "${_mode_title} AICore Exec" "AICPU Sch" "Device E2E")
        _sep=$(printf "  %-40s  %20s  %18s  %18s" "----------------------------------------" "--------------------" "------------------" "------------------")
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %20s  %18s  %18s" "$_label" "${SUMMARY_AICORE_EXEC["${RUN_MODES[0]}|$_label"]:-"-"}" "${SUMMARY_AICPU_EXEC["${RUN_MODES[0]}|$_label"]:-"-"}" "${SUMMARY_DEVICE_E2E["${RUN_MODES[0]}|$_label"]:-"-"}")
            echo "$_row"
        done
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((PASS + FAIL))
echo ""
echo "================================================================"
echo "  Benchmark complete ($RUNTIME): $PASS passed, $FAIL failed ($TOTAL total)"
echo "================================================================"

if [[ -n "$VERBOSE_LOG" ]]; then
    echo "  Verbose log saved to: $VERBOSE_LOG"
fi

[[ $FAIL -eq 0 ]]

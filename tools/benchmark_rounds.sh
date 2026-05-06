#!/usr/bin/env bash
# Benchmark wrapper: run examples on hardware and summarize profiling spans plus device-log averaged E2E.

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
PROFILE_SAMPLE_ROUNDS=1

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
benchmark_rounds.sh — run all examples and report profiling spans plus device-log averaged E2E

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
  AICore Exec (profiling derived earliest AICore start to latest AICore end; single run with --enable-profiling).
  AICPU Dispatch->Finish (profiling derived earliest dispatch to latest finish; single run with --enable-profiling).
  Device E2E (profiling) (profiling derived earliest orch/sched/task start to latest orch/sched/task end; single run with --enable-profiling).
  Device E2E Avg (device log) (average of per-round E2E from a separate non-profiling run using --rounds).
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

calc_improvement_pct() {
    local baseline="$1"
    local sdma="$2"
    python3 - "$baseline" "$sdma" <<'PY'
import sys

try:
    baseline = float(sys.argv[1])
    sdma = float(sys.argv[2])
except (ValueError, TypeError):
    print("-")
    raise SystemExit(0)

if baseline == 0:
    print("-")
else:
    print(f"{(baseline - sdma) / baseline * 100:.2f}%")
PY
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

snapshot_device_logs() {
    if [[ ! -d "$DEVICE_LOG_DIR" ]]; then
        return 0
    fi
    (
        shopt -s nullglob
        for _log in "$DEVICE_LOG_DIR"/*.log; do
            local _size _mtime
            _size=$(stat -c '%s' "$_log" 2>/dev/null || echo 0)
            _mtime=$(stat -c '%Y' "$_log" 2>/dev/null || echo 0)
            printf '%s\t%s\t%s\n' "$_log" "$_size" "$_mtime"
        done
    )
}

UPDATED_DEVICE_LOG_PATH=""
UPDATED_DEVICE_LOG_OFFSET=0
find_updated_device_log() {
    local pre_snapshot="$1"
    local timeout_s=15
    local elapsed=0
    UPDATED_DEVICE_LOG_PATH=""
    UPDATED_DEVICE_LOG_OFFSET=0
    while (( elapsed < timeout_s )); do
        local newest=""
        local newest_offset=0
        local current_logs
        current_logs=$(list_device_logs)
        while IFS= read -r _log; do
            [[ -z "$_log" ]] && continue
            local current_size current_mtime snapshot_entry old_size old_mtime
            current_size=$(stat -c '%s' "$_log" 2>/dev/null || echo 0)
            current_mtime=$(stat -c '%Y' "$_log" 2>/dev/null || echo 0)
            snapshot_entry=$(awk -F '\t' -v target="$_log" '$1 == target { print; exit }' <<<"$pre_snapshot")
            if [[ -z "$snapshot_entry" ]]; then
                if [[ -z "$newest" || "$_log" -nt "$newest" ]]; then
                    newest="$_log"
                    newest_offset=0
                fi
                continue
            fi
            IFS=$'\t' read -r _snapshot_path old_size old_mtime <<<"$snapshot_entry"
            if (( current_size > old_size || current_mtime > old_mtime )); then
                if [[ -z "$newest" || "$_log" -nt "$newest" ]]; then
                    newest="$_log"
                    newest_offset="$old_size"
                fi
            fi
        done <<<"$current_logs"
        if [[ -n "$newest" ]]; then
            UPDATED_DEVICE_LOG_PATH="$newest"
            UPDATED_DEVICE_LOG_OFFSET="$newest_offset"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

extract_device_log_segment() {
    local log_file="$1"
    local byte_offset="${2:-0}"
    if (( byte_offset <= 0 )); then
        printf '%s\n' "$log_file"
        return 0
    fi
    local segment_tmp
    segment_tmp=$(mktemp)
    tail -c +"$((byte_offset + 1))" "$log_file" >"$segment_tmp"
    if [[ ! -s "$segment_tmp" ]]; then
        rm -f "$segment_tmp"
        return 1
    fi
    printf '%s\n' "$segment_tmp"
}

list_perf_jsons() {
    if [[ ! -d "$OUTPUTS_DIR" ]]; then
        return 0
    fi
    (
        shopt -s nullglob
        for _json in "$OUTPUTS_DIR"/perf_swimlane_*.json; do
            printf '%s\n' "$_json"
        done
    )
}

find_new_perf_json() {
    local pre_snapshot="$1"
    local timeout_s=5
    local elapsed=0
    while (( elapsed < timeout_s )); do
        local newest=""
        local current_jsons
        current_jsons=$(list_perf_jsons)
        while IFS= read -r _json; do
            [[ -z "$_json" ]] && continue
            if ! grep -Fxq "$_json" <<<"$pre_snapshot"; then
                if [[ -z "$newest" || "$_json" -nt "$newest" ]]; then
                    newest="$_json"
                fi
            fi
        done <<<"$current_jsons"
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

parse_perf_json_metrics() {
    local perf_json="$1"
    python3 - "$perf_json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

tasks = data.get("tasks") or []
sched_phases = data.get("aicpu_scheduler_phases") or []
orch_phases = data.get("aicpu_orchestrator_phases") or []
orch_summary = data.get("aicpu_orchestrator") or {}


def collect_task_window(start_key, end_key):
    starts = []
    ends = []
    for task in tasks:
        try:
            start = float(task[start_key])
            end = float(task[end_key])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        starts.append(start)
        ends.append(end)
    return starts, ends


def collect_phase_window(groups):
    starts = []
    ends = []
    for group in groups:
        for record in group or []:
            try:
                start = float(record["start_time_us"])
                end = float(record["end_time_us"])
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start:
                continue
            starts.append(start)
            ends.append(end)
    return starts, ends


def collect_summary_window(summary):
    starts = []
    ends = []
    if not summary:
        return starts, ends
    try:
        start = float(summary["start_time_us"])
        end = float(summary["end_time_us"])
    except (KeyError, TypeError, ValueError):
        return starts, ends
    if end <= start:
        return starts, ends
    starts.append(start)
    ends.append(end)
    return starts, ends


def span(starts, ends):
    if not starts or not ends:
        return None
    return max(ends) - min(starts)


task_starts, task_ends = collect_task_window("start_time_us", "end_time_us")
dispatch_starts, dispatch_ends = collect_task_window("dispatch_time_us", "finish_time_us")
sched_starts, sched_ends = collect_phase_window(sched_phases)
orch_phase_starts, orch_phase_ends = collect_phase_window(orch_phases)
orch_starts, orch_ends = collect_summary_window(orch_summary)

aicore_span = span(task_starts, task_ends)
dispatch_finish = span(dispatch_starts, dispatch_ends)

full_e2e_starts = []
full_e2e_ends = []
for starts in (dispatch_starts, task_starts, sched_starts, orch_phase_starts, orch_starts):
    if starts:
        full_e2e_starts.append(min(starts))
for ends in (dispatch_ends, task_ends, sched_ends, orch_phase_ends, orch_ends):
    if ends:
        full_e2e_ends.append(max(ends))
full_e2e = (max(full_e2e_ends) - min(full_e2e_starts)) if full_e2e_starts and full_e2e_ends else None


def emit(key, value):
    if value is None:
        print(f"{key}=-")
    else:
        print(f"{key}={value:.2f}")


emit("AICORE_SPAN", aicore_span)
emit("DISPATCH_FINISH", dispatch_finish)
emit("FULL_E2E", full_e2e)
PY
}

parse_dispatch_finish_fallback() {
    local run_output="$1"
    local exec_us
    exec_us=$(printf "%s\n" "$run_output" | awk '
        match($0, /Total Test Time: ([0-9.]+) us/, m) { print m[1]; found = 1; exit }
        END { if (!found) exit 1 }' 2>/dev/null || true)
    if [[ -n "$exec_us" ]]; then echo "$exec_us"; return 0; fi
    return 1
}

parse_aicore_exec_fallback() {
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
PROFILE_DEVICE_E2E_LOG="-"
PROFILE_DEVICE_E2E_PROF="-"
run_profiling_metrics_once() {
    local mode="$1" kernels_dir="$2" golden="$3" case_name="${4:-}"
    local profile_cmd=(
        env "PTO_SDMA_PREFETCH_MODE=$mode"
        python3 "$RUN_EXAMPLE"
        -k "$kernels_dir" -g "$golden"
        -p "$PLATFORM" -d "$DEVICE_ID"
        -n "$PROFILE_SAMPLE_ROUNDS" --skip-golden --enable-profiling
    )
    if [[ -n "$case_name" ]]; then
        profile_cmd+=(--case "$case_name")
    fi
    profile_cmd+=("${EXTRA_ARGS[@]}")

    local pre_run_perf_jsons profile_tmp profile_output profile_rc=0
    pre_run_perf_jsons=$(list_perf_jsons)
    profile_tmp=$(mktemp)
    "${profile_cmd[@]}" >"$profile_tmp" 2>&1 || profile_rc=$?
    profile_output=$(<"$profile_tmp")
    rm -f "$profile_tmp"
    PROFILE_AICORE_EXEC="-"; PROFILE_AICPU_EXEC="-"; PROFILE_DEVICE_E2E_PROF="-"
    if [[ $profile_rc -ne 0 ]]; then return 1; fi
    local perf_json perf_metrics
    if perf_json=$(find_new_perf_json "$pre_run_perf_jsons"); then
        vlog "Resolved perf JSON: $perf_json"
        perf_metrics=$(parse_perf_json_metrics "$perf_json" 2>/dev/null || true)
        while IFS='=' read -r metric_name metric_value; do
            [[ -z "${metric_name:-}" ]] && continue
            case "$metric_name" in
                AICORE_SPAN) PROFILE_AICORE_EXEC="$metric_value" ;;
                DISPATCH_FINISH) PROFILE_AICPU_EXEC="$metric_value" ;;
                FULL_E2E) PROFILE_DEVICE_E2E_PROF="$metric_value" ;;
            esac
        done <<<"$perf_metrics"
    fi
    if [[ "$PROFILE_AICORE_EXEC" == "-" ]]; then
        parse_aicore_exec_fallback "$profile_output" >/dev/null \
            && PROFILE_AICORE_EXEC=$(parse_aicore_exec_fallback "$profile_output")
    fi
    if [[ "$PROFILE_AICPU_EXEC" == "-" ]]; then
        parse_dispatch_finish_fallback "$profile_output" >/dev/null \
            && PROFILE_AICPU_EXEC=$(parse_dispatch_finish_fallback "$profile_output")
    fi
    [[ -n "$VERBOSE_LOG" && -n "$profile_output" ]] && {
        echo "===== profiling run (mode=$mode case=${case_name:-DEFAULT} rounds=$PROFILE_SAMPLE_ROUNDS) =====" >> "$VERBOSE_LOG"
        echo "$profile_output" >> "$VERBOSE_LOG"
    }
    return 0
}

run_device_log_rounds_once() {
    local mode="$1" kernels_dir="$2" golden="$3" case_name="${4:-}"
    local device_log_cmd=(
        env "PTO_SDMA_PREFETCH_MODE=$mode"
        python3 "$RUN_EXAMPLE"
        -k "$kernels_dir" -g "$golden"
        -p "$PLATFORM" -d "$DEVICE_ID"
        -n "$ROUNDS" --skip-golden
    )
    if [[ -n "$case_name" ]]; then
        device_log_cmd+=(--case "$case_name")
    fi
    device_log_cmd+=("${EXTRA_ARGS[@]}")

    local pre_run_logs device_tmp device_output device_rc=0
    pre_run_logs=$(snapshot_device_logs)
    device_tmp=$(mktemp)
    "${device_log_cmd[@]}" >"$device_tmp" 2>&1 || device_rc=$?
    device_output=$(<"$device_tmp")
    rm -f "$device_tmp"
    PROFILE_DEVICE_E2E_LOG="-"
    if [[ $device_rc -ne 0 ]]; then return 1; fi
    local device_log segment_log=""
    if find_updated_device_log "$pre_run_logs"; then
        device_log="$UPDATED_DEVICE_LOG_PATH"
        vlog "Resolved device log update: $device_log (offset=$UPDATED_DEVICE_LOG_OFFSET)"
        if segment_log=$(extract_device_log_segment "$device_log" "$UPDATED_DEVICE_LOG_OFFSET"); then
            parse_device_e2e_avg "$segment_log" >/dev/null && PROFILE_DEVICE_E2E_LOG=$(parse_device_e2e_avg "$segment_log")
            [[ "$segment_log" != "$device_log" ]] && rm -f "$segment_log"
        fi
    else
        echo "    Warning: no device log update detected after device-log pass"
    fi
    [[ -n "$VERBOSE_LOG" && -n "$device_output" ]] && {
        echo "===== device-log run (mode=$mode case=${case_name:-DEFAULT} rounds=$ROUNDS) =====" >> "$VERBOSE_LOG"
        echo "$device_output" >> "$VERBOSE_LOG"
    }
    return 0
}

run_benchmark_once() {
    local mode="$1" kernels_dir="$2" golden="$3" case_name="${4:-}"
    PROFILE_AICORE_EXEC="-"
    PROFILE_AICPU_EXEC="-"
    PROFILE_DEVICE_E2E_PROF="-"
    PROFILE_DEVICE_E2E_LOG="-"
    echo "    Profiling pass: rounds=$PROFILE_SAMPLE_ROUNDS, --enable-profiling"
    run_profiling_metrics_once "$mode" "$kernels_dir" "$golden" "$case_name" || return 1
    echo "    Device-log pass: rounds=$ROUNDS, no --enable-profiling"
    run_device_log_rounds_once "$mode" "$kernels_dir" "$golden" "$case_name" || return 1
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
declare -A SUMMARY_DEVICE_E2E_LOG=()
declare -A SUMMARY_DEVICE_E2E_PROF=()

echo ""
echo "Runtime: $RUNTIME"
echo "Tests dir: $EXAMPLES_DIR"
echo "Prefetch modes: ${RUN_MODES[*]}"
echo "Profiling metrics: 1 round with --enable-profiling"
echo "Device-log E2E Avg: $ROUNDS round(s) without --enable-profiling"

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
            if run_benchmark_once "$mode" "$_kernels" "$_golden" "$_case"; then
                ((PASS++)) || true
            else
                ((FAIL++)) || true
            fi
            [[ -z "${SUMMARY_LABELS_SEEN[$_label]+x}" ]] && SUMMARY_NAMES+=("$_label") && SUMMARY_LABELS_SEEN["$_label"]=1
            SUMMARY_AICORE_EXEC["$mode|$_label"]="$PROFILE_AICORE_EXEC"
            SUMMARY_AICPU_EXEC["$mode|$_label"]="$PROFILE_AICPU_EXEC"
            SUMMARY_DEVICE_E2E_LOG["$mode|$_label"]="$PROFILE_DEVICE_E2E_LOG"
            SUMMARY_DEVICE_E2E_PROF["$mode|$_label"]="$PROFILE_DEVICE_E2E_PROF"
            echo "  AICore Exec (profiling, single-run, us): $PROFILE_AICORE_EXEC"
            echo "  AICPU Dispatch->Finish (profiling, single-run, us): $PROFILE_AICPU_EXEC"
            echo "  Device E2E (profiling, single-run, us): $PROFILE_DEVICE_E2E_PROF"
            echo "  Device E2E Avg (device log, ${ROUNDS} round(s), us): $PROFILE_DEVICE_E2E_LOG"
        done
        if [[ "$PREFETCH_MODE" == "compare" ]]; then
            local _base_devlog="${SUMMARY_DEVICE_E2E_LOG["baseline|$_label"]:-"-"}"
            local _sdma_devlog="${SUMMARY_DEVICE_E2E_LOG["sdma|$_label"]:-"-"}"
            local _opt_pct="-"
            if [[ "$_base_devlog" != "-" && "$_sdma_devlog" != "-" ]]; then
                _opt_pct=$(calc_improvement_pct "$_base_devlog" "$_sdma_devlog")
            fi
            echo "  Device E2E Avg (device log) SDMA vs Baseline: $_opt_pct"
        fi
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
# Profiling Summary Table
# ---------------------------------------------------------------------------
if [[ ${#SUMMARY_NAMES[@]} -gt 0 ]]; then
    echo ""
    echo "================================================================"
    echo "  Profiling Summary ($RUNTIME)"
    echo "================================================================"
    echo ""

    if [[ "$PREFETCH_MODE" == "compare" ]]; then
        _hdr=$(printf "  %-40s  %20s  %20s  %18s  %18s  %21s  %21s" "Example" "Baseline AICore Exec" "SDMA AICore Exec" "Base AICPU D->F" "SDMA AICPU D->F" "Base E2E(profiling)" "SDMA E2E(profiling)")
        _sep=$(printf "  %-40s  %20s  %20s  %18s  %18s  %21s  %21s" "----------------------------------------" "--------------------" "--------------------" "------------------" "------------------" "---------------------" "---------------------")
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %20s  %20s  %18s  %18s  %21s  %21s" "$_label" "${SUMMARY_AICORE_EXEC["baseline|$_label"]:-"-"}" "${SUMMARY_AICORE_EXEC["sdma|$_label"]:-"-"}" "${SUMMARY_AICPU_EXEC["baseline|$_label"]:-"-"}" "${SUMMARY_AICPU_EXEC["sdma|$_label"]:-"-"}" "${SUMMARY_DEVICE_E2E_PROF["baseline|$_label"]:-"-"}" "${SUMMARY_DEVICE_E2E_PROF["sdma|$_label"]:-"-"}")
            echo "$_row"
        done
    else
        _mode_name="${RUN_MODES[0]}"
        _mode_title=$(printf "%s" "$_mode_name" | tr '[:lower:]' '[:upper:]')
        _hdr=$(printf "  %-40s  %20s  %18s  %21s" "Example" "${_mode_title} AICore Exec" "AICPU Dispatch->Finish" "E2E (profiling)")
        _sep=$(printf "  %-40s  %20s  %18s  %21s" "----------------------------------------" "--------------------" "------------------" "---------------------")
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %20s  %18s  %21s" "$_label" "${SUMMARY_AICORE_EXEC["${RUN_MODES[0]}|$_label"]:-"-"}" "${SUMMARY_AICPU_EXEC["${RUN_MODES[0]}|$_label"]:-"-"}" "${SUMMARY_DEVICE_E2E_PROF["${RUN_MODES[0]}|$_label"]:-"-"}")
            echo "$_row"
        done
    fi
fi

# ---------------------------------------------------------------------------
# Device-log E2E Avg Summary Table
# ---------------------------------------------------------------------------
if [[ ${#SUMMARY_NAMES[@]} -gt 0 ]]; then
    echo ""
    echo "================================================================"
    echo "  Device-log E2E Avg Summary ($RUNTIME)"
    echo "================================================================"
    echo ""

    if [[ "$PREFETCH_MODE" == "compare" ]]; then
        _hdr=$(printf "  %-40s  %20s  %20s  %16s" "Example" "Base E2E(devlog avg)" "SDMA E2E(devlog avg)" "SDMA vs Base")
        _sep=$(printf "  %-40s  %20s  %20s  %16s" "----------------------------------------" "--------------------" "--------------------" "----------------")
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _base_devlog="${SUMMARY_DEVICE_E2E_LOG["baseline|$_label"]:-"-"}"
            _sdma_devlog="${SUMMARY_DEVICE_E2E_LOG["sdma|$_label"]:-"-"}"
            _opt_pct="-"
            if [[ "$_base_devlog" != "-" && "$_sdma_devlog" != "-" ]]; then
                _opt_pct=$(calc_improvement_pct "$_base_devlog" "$_sdma_devlog")
            fi
            _row=$(printf "  %-40s  %20s  %20s  %16s" "$_label" "$_base_devlog" "$_sdma_devlog" "$_opt_pct")
            echo "$_row"
        done
    else
        _mode_name="${RUN_MODES[0]}"
        _mode_title=$(printf "%s" "$_mode_name" | tr '[:lower:]' '[:upper:]')
        _hdr=$(printf "  %-40s  %20s" "Example" "${_mode_title} E2E Avg(devlog)")
        _sep=$(printf "  %-40s  %20s" "----------------------------------------" "--------------------")
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %20s" "$_label" "${SUMMARY_DEVICE_E2E_LOG["${RUN_MODES[0]}|$_label"]:-"-"}")
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

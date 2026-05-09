#!/usr/bin/env bash
# Benchmark wrapper: run examples on hardware and compare average per-task AICore execution time.

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
benchmark_avg_aicore_exec.sh — run all examples and report avg(end_time_us - start_time_us) across all tasks

Usage:
  ./tools/benchmark_avg_aicore_exec.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>] [--prefetch-mode <mode>] [-v]

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
  Avg AICore Task Exec (us): avg(tasks[].end_time_us - tasks[].start_time_us)
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
    VERBOSE_LOG="$OUTPUTS_DIR/benchmark_avg_aicore_exec_$(date +%Y%m%d_%H%M%S).log"
    echo "Verbose log: $VERBOSE_LOG"
fi
mkdir -p "$OUTPUTS_DIR"

vlog() {
    if [[ -n "$VERBOSE_LOG" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$VERBOSE_LOG"
    fi
}

# ---------------------------------------------------------------------------
# Select example cases and order based on runtime
# ---------------------------------------------------------------------------
EXAMPLES_DIR="$PROJECT_ROOT/tests/st/${PLATFORM}/${RUNTIME}"
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
# Resolve profiling / device log output
# ---------------------------------------------------------------------------
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

list_device_logs() {
    local log_root
    if [[ -n "${ASCEND_WORK_PATH:-}" ]]; then
        log_root="$ASCEND_WORK_PATH/log/debug"
        if [[ ! -d "$log_root" ]]; then
            log_root="$HOME/ascend/log/debug"
        fi
    else
        log_root="$HOME/ascend/log/debug"
    fi
    local device_log_dir="$log_root/device-${DEVICE_ID}"
    if [[ ! -d "$device_log_dir" ]]; then
        return 0
    fi
    (
        shopt -s nullglob
        for _log in "$device_log_dir"/*.log; do
            printf '%s\n' "$_log"
        done
    )
}

snapshot_device_logs() {
    local log_root
    if [[ -n "${ASCEND_WORK_PATH:-}" ]]; then
        log_root="$ASCEND_WORK_PATH/log/debug"
        if [[ ! -d "$log_root" ]]; then
            log_root="$HOME/ascend/log/debug"
        fi
    else
        log_root="$HOME/ascend/log/debug"
    fi
    local device_log_dir="$log_root/device-${DEVICE_ID}"
    if [[ ! -d "$device_log_dir" ]]; then
        return 0
    fi
    (
        shopt -s nullglob
        for _log in "$device_log_dir"/*.log; do
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

parse_perf_json_avg_aicore_exec() {
    local perf_json="$1"
    python3 - "$perf_json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

tasks = data.get("tasks") or []
durations = []
for task in tasks:
    try:
        start = float(task["start_time_us"])
        end = float(task["end_time_us"])
    except (KeyError, TypeError, ValueError):
        continue
    if end <= start:
        continue
    durations.append(end - start)

if not durations:
    sys.exit(1)

print(f"{sum(durations) / len(durations):.2f}")
PY
}

parse_avg_aicore_exec_fallback() {
    local run_output="$1"
    local exec_us
    exec_us=$(printf "%s\n" "$run_output" | awk '
        match($0, /Per-task \(all\):[[:space:]]+Avg Exec = ([0-9.]+) us/, m) {
            print m[1]
            found = 1
            exit
        }
        END { if (!found) exit 1 }' 2>/dev/null || true)
    if [[ -n "$exec_us" ]]; then
        echo "$exec_us"
        return 0
    fi
    return 1
}

parse_host_prefetch_setup_outcome() {
    local run_output="$1"
    local outcome
    outcome=$(printf "%s\n" "$run_output" | awk '
        match($0, /host_prefetch_setup outcome=([^ ]+) channels=/, m) {
            print m[1]
            found = 1
            exit
        }
        END { if (!found) exit 1 }' 2>/dev/null || true)
    if [[ -n "$outcome" ]]; then
        echo "$outcome"
        return 0
    fi
    return 1
}

parse_prefetch_ctrl_total_us() {
    local log_file="$1"
    local total_us
    total_us=$(awk '
        match($0, /Prefetch control path summary: .*total=([0-9.]+)us/, m) {
            val = m[1]
        }
        END {
            if (val != "") {
                print val
            } else {
                exit 1
            }
        }' "$log_file" 2>/dev/null || true)
    if [[ -n "$total_us" ]]; then echo "$total_us"; return 0; fi
    return 1
}

parse_prefetch_issue_total_us() {
    local log_file="$1"
    local total_us
    total_us=$(awk '
        match($0, /SDMA prefetch issue summary: .*total=([0-9.]+)us/, m) {
            val = m[1]
        }
        END {
            if (val != "") {
                print val
            } else {
                exit 1
            }
        }' "$log_file" 2>/dev/null || true)
    if [[ -n "$total_us" ]]; then echo "$total_us"; return 0; fi
    return 1
}

parse_prefetch_debug_counts() {
    local log_file="$1"
    python3 - "$log_file" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="ignore")

def find(pattern):
    m = re.search(pattern, text, re.S)
    return m.group(1) if m else None

results = {
    "CTRL_CONSIDERED": find(r"Prefetch control path summary: .*?considered=(\d+)"),
    "CTRL_ELIGIBLE": find(r"Prefetch control path summary: .*?eligible_tasks=(\d+)"),
    "CTRL_TOTAL_US": find(r"Prefetch control path summary: .*?total=([0-9.]+)us"),
    "CTRL_AVG_US": find(r"Prefetch control path summary: .*?avg=([0-9.]+)us"),
    "CTRL_ELIGIBLE_TOTAL_US": find(r"Prefetch control path summary: .*?eligible_total=([0-9.]+)us"),
    "CTRL_ELIGIBLE_AVG_US": find(r"Prefetch control path summary: .*?eligible_avg=([0-9.]+)us"),
    "CTRL_SKIP_NOT_SDMA": find(r"Prefetch task summary: .*?skip_not_sdma=(\d+)"),
    "CTRL_SKIP_NOT_AVAILABLE": find(r"Prefetch task summary: .*?skip_not_available=(\d+)"),
    "CTRL_SKIP_NULL_PAYLOAD": find(r"Prefetch task summary: .*?skip_null_payload=(\d+)"),
    "CTRL_SKIP_BELOW_MIN_BYTES": find(r"Prefetch task summary: .*?skip_below_min_bytes=(\d+)"),
    "CTRL_SKIP_NO_VALID_TENSOR": find(r"Prefetch task summary: .*?skip_no_valid_tensor=(\d+)"),
    "CTRL_SKIP_SCHEDULER_SUPPRESSED": find(r"Prefetch task summary: .*?skip_scheduler_suppressed=(\d+)"),
    "CTRL_BYTES": find(r"Prefetch task summary: .*?bytes=(\d+)"),
    "CTRL_TENSORS": find(r"Prefetch task summary: .*?tensors=(\d+)"),
    "CTRL_MIN_BYTES": find(r"Prefetch task summary: .*?min_bytes=(\d+)"),
    "ISSUE_ENABLED": find(r"SDMA prefetch issue summary: .*?enabled=(\d+)"),
    "ISSUE_ATTEMPTS": find(r"SDMA prefetch issue summary: .*?attempts=(\d+)"),
    "ISSUE_ISSUES": find(r"SDMA prefetch issue summary: .*?issues=(\d+)"),
    "ISSUE_BYTES": find(r"SDMA prefetch issue summary: .*?bytes=(\d+)"),
    "ISSUE_ISSUE_BYTES": find(r"SDMA prefetch issue summary: .*?issue_bytes=(\d+)"),
    "ISSUE_SUPPRESSED": find(r"SDMA prefetch issue summary: .*?suppressed=(\d+)"),
    "ISSUE_QUEUE_FULL": find(r"SDMA prefetch issue summary: .*?queue_full=(\d+)"),
    "ISSUE_DUP_INSTR": find(r"SDMA prefetch issue summary: .*?dup_instr=(\d+)"),
    "ISSUE_DUP_INSTR_KERNEL": find(r"SDMA prefetch issue summary: .*?dup_instr_kernel=(\d+)"),
    "ISSUE_TOTAL_US": find(r"SDMA prefetch issue summary: .*?total=([0-9.]+)us"),
    "ISSUE_AVG_US": find(r"SDMA prefetch issue summary: .*?avg=([0-9.]+)us"),
    "ISSUE_ISSUE_TOTAL_US": find(r"SDMA prefetch issue summary: .*?issue_total=([0-9.]+)us"),
    "ISSUE_ISSUE_AVG_US": find(r"SDMA prefetch issue summary: .*?issue_avg=([0-9.]+)us"),
}

for key in sorted(results):
    if results[key] is not None:
        print(f"{key}={results[key]}")
PY
}

PROFILE_AVG_AICORE_EXEC="-"
PROFILE_PREFETCH_SETUP_OUTCOME="-"
PROFILE_PREFETCH_CTRL_US="-"
PROFILE_PREFETCH_ISSUE_US="-"
PROFILE_PREFETCH_CTRL_CONSIDERED="-"
PROFILE_PREFETCH_CTRL_ELIGIBLE="-"
PROFILE_PREFETCH_CTRL_SKIP_NOT_SDMA="-"
PROFILE_PREFETCH_CTRL_SKIP_NOT_AVAILABLE="-"
PROFILE_PREFETCH_CTRL_SKIP_NULL_PAYLOAD="-"
PROFILE_PREFETCH_CTRL_SKIP_BELOW_MIN_BYTES="-"
PROFILE_PREFETCH_CTRL_SKIP_NO_VALID_TENSOR="-"
PROFILE_PREFETCH_CTRL_SKIP_SCHEDULER_SUPPRESSED="-"
PROFILE_PREFETCH_ISSUE_ATTEMPTS="-"
PROFILE_PREFETCH_ISSUE_ISSUES="-"
PROFILE_PREFETCH_ISSUE_SUPPRESSED="-"
PROFILE_PREFETCH_ISSUE_QUEUE_FULL="-"
PROFILE_PREFETCH_ISSUE_DUP_INSTR="-"
PROFILE_PREFETCH_ISSUE_DUP_INSTR_KERNEL="-"
run_profile_once() {
    local mode="$1" kernels_dir="$2" golden="$3" case_name="${4:-}"
    local profile_cmd=(
        env "PTO_SDMA_PREFETCH_MODE=$mode" "PTO_SDMA_PREFETCH_DEBUG=1"
        python3 "$RUN_EXAMPLE"
        -k "$kernels_dir" -g "$golden"
        -p "$PLATFORM" -d "$DEVICE_ID"
        -n "$ROUNDS" --skip-golden --enable-profiling
    )
    if [[ -n "$case_name" ]]; then
        profile_cmd+=(--case "$case_name")
    fi
    profile_cmd+=("${EXTRA_ARGS[@]}")

    local pre_run_logs pre_run_perf_jsons profile_tmp profile_output profile_rc=0
    pre_run_logs=$(snapshot_device_logs)
    pre_run_perf_jsons=$(list_perf_jsons)
    profile_tmp=$(mktemp)
    "${profile_cmd[@]}" >"$profile_tmp" 2>&1 || profile_rc=$?
    profile_output=$(<"$profile_tmp")
    rm -f "$profile_tmp"

    PROFILE_AVG_AICORE_EXEC="-"
    PROFILE_PREFETCH_SETUP_OUTCOME="-"
    PROFILE_PREFETCH_CTRL_US="-"
    PROFILE_PREFETCH_ISSUE_US="-"
    PROFILE_PREFETCH_CTRL_CONSIDERED="-"
    PROFILE_PREFETCH_CTRL_ELIGIBLE="-"
    PROFILE_PREFETCH_CTRL_SKIP_NOT_SDMA="-"
    PROFILE_PREFETCH_CTRL_SKIP_NOT_AVAILABLE="-"
    PROFILE_PREFETCH_CTRL_SKIP_NULL_PAYLOAD="-"
    PROFILE_PREFETCH_CTRL_SKIP_BELOW_MIN_BYTES="-"
    PROFILE_PREFETCH_CTRL_SKIP_NO_VALID_TENSOR="-"
    PROFILE_PREFETCH_CTRL_SKIP_SCHEDULER_SUPPRESSED="-"
    PROFILE_PREFETCH_ISSUE_ATTEMPTS="-"
    PROFILE_PREFETCH_ISSUE_ISSUES="-"
    PROFILE_PREFETCH_ISSUE_SUPPRESSED="-"
    PROFILE_PREFETCH_ISSUE_QUEUE_FULL="-"
    PROFILE_PREFETCH_ISSUE_DUP_INSTR="-"
    PROFILE_PREFETCH_ISSUE_DUP_INSTR_KERNEL="-"
    if [[ $profile_rc -ne 0 ]]; then
        [[ -n "$VERBOSE_LOG" && -n "$profile_output" ]] && echo "$profile_output" >> "$VERBOSE_LOG"
        return 1
    fi

    local perf_json
    if perf_json=$(find_new_perf_json "$pre_run_perf_jsons"); then
        vlog "Resolved perf JSON: $perf_json"
        parse_perf_json_avg_aicore_exec "$perf_json" >/dev/null \
            && PROFILE_AVG_AICORE_EXEC=$(parse_perf_json_avg_aicore_exec "$perf_json")
    fi
    if [[ "$PROFILE_AVG_AICORE_EXEC" == "-" ]]; then
        parse_avg_aicore_exec_fallback "$profile_output" >/dev/null \
            && PROFILE_AVG_AICORE_EXEC=$(parse_avg_aicore_exec_fallback "$profile_output")
    fi
    if [[ "$PROFILE_PREFETCH_SETUP_OUTCOME" == "-" ]]; then
        parse_host_prefetch_setup_outcome "$profile_output" >/dev/null \
            && PROFILE_PREFETCH_SETUP_OUTCOME=$(parse_host_prefetch_setup_outcome "$profile_output")
    fi
    local device_log segment_log=""
    if find_updated_device_log "$pre_run_logs"; then
        device_log="$UPDATED_DEVICE_LOG_PATH"
        vlog "Resolved device log update: $device_log (offset=$UPDATED_DEVICE_LOG_OFFSET)"
        if segment_log=$(extract_device_log_segment "$device_log" "$UPDATED_DEVICE_LOG_OFFSET"); then
            parse_prefetch_ctrl_total_us "$segment_log" >/dev/null \
                && PROFILE_PREFETCH_CTRL_US=$(parse_prefetch_ctrl_total_us "$segment_log")
            parse_prefetch_issue_total_us "$segment_log" >/dev/null \
                && PROFILE_PREFETCH_ISSUE_US=$(parse_prefetch_issue_total_us "$segment_log")
        local prefetch_debug
            if prefetch_debug=$(parse_prefetch_debug_counts "$segment_log"); then
            while IFS='=' read -r key value; do
                [[ -z "${key:-}" ]] && continue
                case "$key" in
                    CTRL_CONSIDERED) PROFILE_PREFETCH_CTRL_CONSIDERED="$value" ;;
                    CTRL_ELIGIBLE) PROFILE_PREFETCH_CTRL_ELIGIBLE="$value" ;;
                    CTRL_SKIP_NOT_SDMA) PROFILE_PREFETCH_CTRL_SKIP_NOT_SDMA="$value" ;;
                    CTRL_SKIP_NOT_AVAILABLE) PROFILE_PREFETCH_CTRL_SKIP_NOT_AVAILABLE="$value" ;;
                    CTRL_SKIP_NULL_PAYLOAD) PROFILE_PREFETCH_CTRL_SKIP_NULL_PAYLOAD="$value" ;;
                    CTRL_SKIP_BELOW_MIN_BYTES) PROFILE_PREFETCH_CTRL_SKIP_BELOW_MIN_BYTES="$value" ;;
                    CTRL_SKIP_NO_VALID_TENSOR) PROFILE_PREFETCH_CTRL_SKIP_NO_VALID_TENSOR="$value" ;;
                    CTRL_SKIP_SCHEDULER_SUPPRESSED) PROFILE_PREFETCH_CTRL_SKIP_SCHEDULER_SUPPRESSED="$value" ;;
                    ISSUE_ATTEMPTS) PROFILE_PREFETCH_ISSUE_ATTEMPTS="$value" ;;
                    ISSUE_ISSUES) PROFILE_PREFETCH_ISSUE_ISSUES="$value" ;;
                    ISSUE_SUPPRESSED) PROFILE_PREFETCH_ISSUE_SUPPRESSED="$value" ;;
                    ISSUE_QUEUE_FULL) PROFILE_PREFETCH_ISSUE_QUEUE_FULL="$value" ;;
                    ISSUE_DUP_INSTR) PROFILE_PREFETCH_ISSUE_DUP_INSTR="$value" ;;
                    ISSUE_DUP_INSTR_KERNEL) PROFILE_PREFETCH_ISSUE_DUP_INSTR_KERNEL="$value" ;;
                esac
            done <<<"$prefetch_debug"
            fi
            [[ "$segment_log" != "$device_log" ]] && rm -f "$segment_log"
        fi
    fi
    [[ -n "$VERBOSE_LOG" && -n "$profile_output" ]] && echo "$profile_output" >> "$VERBOSE_LOG"
    return 0
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
PASS=0
FAIL=0

SUMMARY_NAMES=()
declare -A SUMMARY_LABELS_SEEN=()
declare -A SUMMARY_AVG_AICORE_EXEC=()
declare -A SUMMARY_PREFETCH_SETUP_OUTCOME=()
declare -A SUMMARY_PREFETCH_CTRL_US=()
declare -A SUMMARY_PREFETCH_ISSUE_US=()
declare -A SUMMARY_PREFETCH_CTRL_CONSIDERED=()
declare -A SUMMARY_PREFETCH_CTRL_ELIGIBLE=()
declare -A SUMMARY_PREFETCH_CTRL_SKIP_NOT_SDMA=()
declare -A SUMMARY_PREFETCH_CTRL_SKIP_NOT_AVAILABLE=()
declare -A SUMMARY_PREFETCH_CTRL_SKIP_NULL_PAYLOAD=()
declare -A SUMMARY_PREFETCH_CTRL_SKIP_BELOW_MIN_BYTES=()
declare -A SUMMARY_PREFETCH_CTRL_SKIP_NO_VALID_TENSOR=()
declare -A SUMMARY_PREFETCH_CTRL_SKIP_SCHEDULER_SUPPRESSED=()
declare -A SUMMARY_PREFETCH_ISSUE_ATTEMPTS=()
declare -A SUMMARY_PREFETCH_ISSUE_ISSUES=()
declare -A SUMMARY_PREFETCH_ISSUE_SUPPRESSED=()
declare -A SUMMARY_PREFETCH_ISSUE_QUEUE_FULL=()
declare -A SUMMARY_PREFETCH_ISSUE_DUP_INSTR=()
declare -A SUMMARY_PREFETCH_ISSUE_DUP_INSTR_KERNEL=()

echo ""
echo "Runtime: $RUNTIME"
echo "Tests dir: $EXAMPLES_DIR"
echo "Prefetch modes: ${RUN_MODES[*]}"
if (( ROUNDS > 1 )); then
    echo "Note: Avg AICore Task Exec is profiling-derived and reflects the profiled round only."
fi

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
            SUMMARY_AVG_AICORE_EXEC["$mode|$_label"]="$PROFILE_AVG_AICORE_EXEC"
            SUMMARY_PREFETCH_SETUP_OUTCOME["$mode|$_label"]="$PROFILE_PREFETCH_SETUP_OUTCOME"
            SUMMARY_PREFETCH_CTRL_US["$mode|$_label"]="$PROFILE_PREFETCH_CTRL_US"
            SUMMARY_PREFETCH_ISSUE_US["$mode|$_label"]="$PROFILE_PREFETCH_ISSUE_US"
            SUMMARY_PREFETCH_CTRL_CONSIDERED["$mode|$_label"]="$PROFILE_PREFETCH_CTRL_CONSIDERED"
            SUMMARY_PREFETCH_CTRL_ELIGIBLE["$mode|$_label"]="$PROFILE_PREFETCH_CTRL_ELIGIBLE"
            SUMMARY_PREFETCH_CTRL_SKIP_NOT_SDMA["$mode|$_label"]="$PROFILE_PREFETCH_CTRL_SKIP_NOT_SDMA"
            SUMMARY_PREFETCH_CTRL_SKIP_NOT_AVAILABLE["$mode|$_label"]="$PROFILE_PREFETCH_CTRL_SKIP_NOT_AVAILABLE"
            SUMMARY_PREFETCH_CTRL_SKIP_NULL_PAYLOAD["$mode|$_label"]="$PROFILE_PREFETCH_CTRL_SKIP_NULL_PAYLOAD"
            SUMMARY_PREFETCH_CTRL_SKIP_BELOW_MIN_BYTES["$mode|$_label"]="$PROFILE_PREFETCH_CTRL_SKIP_BELOW_MIN_BYTES"
            SUMMARY_PREFETCH_CTRL_SKIP_NO_VALID_TENSOR["$mode|$_label"]="$PROFILE_PREFETCH_CTRL_SKIP_NO_VALID_TENSOR"
            SUMMARY_PREFETCH_CTRL_SKIP_SCHEDULER_SUPPRESSED["$mode|$_label"]="$PROFILE_PREFETCH_CTRL_SKIP_SCHEDULER_SUPPRESSED"
            SUMMARY_PREFETCH_ISSUE_ATTEMPTS["$mode|$_label"]="$PROFILE_PREFETCH_ISSUE_ATTEMPTS"
            SUMMARY_PREFETCH_ISSUE_ISSUES["$mode|$_label"]="$PROFILE_PREFETCH_ISSUE_ISSUES"
            SUMMARY_PREFETCH_ISSUE_SUPPRESSED["$mode|$_label"]="$PROFILE_PREFETCH_ISSUE_SUPPRESSED"
            SUMMARY_PREFETCH_ISSUE_QUEUE_FULL["$mode|$_label"]="$PROFILE_PREFETCH_ISSUE_QUEUE_FULL"
            SUMMARY_PREFETCH_ISSUE_DUP_INSTR["$mode|$_label"]="$PROFILE_PREFETCH_ISSUE_DUP_INSTR"
            SUMMARY_PREFETCH_ISSUE_DUP_INSTR_KERNEL["$mode|$_label"]="$PROFILE_PREFETCH_ISSUE_DUP_INSTR_KERNEL"
            echo "  Avg AICore Task Exec (us): $PROFILE_AVG_AICORE_EXEC"
            echo "  Prefetch Setup Outcome: $PROFILE_PREFETCH_SETUP_OUTCOME"
            echo "  Prefetch Ctrl Path Total (us): $PROFILE_PREFETCH_CTRL_US"
            echo "  Prefetch SQE Issue Total (us): $PROFILE_PREFETCH_ISSUE_US"
            echo "  Prefetch Ctrl Counts: considered=$PROFILE_PREFETCH_CTRL_CONSIDERED eligible=$PROFILE_PREFETCH_CTRL_ELIGIBLE skip_not_sdma=$PROFILE_PREFETCH_CTRL_SKIP_NOT_SDMA skip_not_available=$PROFILE_PREFETCH_CTRL_SKIP_NOT_AVAILABLE skip_null_payload=$PROFILE_PREFETCH_CTRL_SKIP_NULL_PAYLOAD skip_below_min_bytes=$PROFILE_PREFETCH_CTRL_SKIP_BELOW_MIN_BYTES skip_no_valid_tensor=$PROFILE_PREFETCH_CTRL_SKIP_NO_VALID_TENSOR skip_scheduler_suppressed=$PROFILE_PREFETCH_CTRL_SKIP_SCHEDULER_SUPPRESSED"
            echo "  Prefetch Issue Counts: attempts=$PROFILE_PREFETCH_ISSUE_ATTEMPTS issues=$PROFILE_PREFETCH_ISSUE_ISSUES suppressed=$PROFILE_PREFETCH_ISSUE_SUPPRESSED queue_full=$PROFILE_PREFETCH_ISSUE_QUEUE_FULL dup_instr=$PROFILE_PREFETCH_ISSUE_DUP_INSTR dup_instr_kernel=$PROFILE_PREFETCH_ISSUE_DUP_INSTR_KERNEL"
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
    echo "  Avg AICore Task Exec Summary ($RUNTIME)"
    echo "================================================================"
    echo ""

    if [[ "$PREFETCH_MODE" == "compare" ]]; then
        _hdr=$(printf "  %-40s  %20s  %24s  %24s" "Example" "Setup Outcome" "Baseline Avg Exec (us)" "SDMA Avg Exec (us)")
        _sep=$(printf "  %-40s  %20s  %24s  %24s" "----------------------------------------" "--------------------" "------------------------" "------------------------")
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %20s  %24s  %24s" "$_label" "${SUMMARY_PREFETCH_SETUP_OUTCOME["sdma|$_label"]:-"-"}" "${SUMMARY_AVG_AICORE_EXEC["baseline|$_label"]:-"-"}" "${SUMMARY_AVG_AICORE_EXEC["sdma|$_label"]:-"-"}")
            echo "$_row"
        done
    else
        _mode_name="${RUN_MODES[0]}"
        _mode_title=$(printf "%s" "$_mode_name" | tr '[:lower:]' '[:upper:]')
        _hdr=$(printf "  %-40s  %20s  %24s" "Example" "Setup Outcome" "${_mode_title} Avg Exec (us)")
        _sep=$(printf "  %-40s  %20s  %24s" "----------------------------------------" "--------------------" "------------------------")
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %20s  %24s" "$_label" "${SUMMARY_PREFETCH_SETUP_OUTCOME["${RUN_MODES[0]}|$_label"]:-"-"}" "${SUMMARY_AVG_AICORE_EXEC["${RUN_MODES[0]}|$_label"]:-"-"}")
            echo "$_row"
        done
    fi
fi

# ---------------------------------------------------------------------------
# Prefetch Overhead Summary Table
# ---------------------------------------------------------------------------
if [[ ${#SUMMARY_NAMES[@]} -gt 0 ]]; then
    echo ""
    echo "================================================================"
    echo "  Prefetch Overhead Summary ($RUNTIME)"
    echo "================================================================"
    echo ""

    if [[ "$PREFETCH_MODE" == "compare" ]]; then
        _hdr=$(printf "  %-40s  %18s  %18s" "Example" "Baseline Ctrl(us)" "SDMA Ctrl(us)")
        _sep=$(printf "  %-40s  %18s  %18s" "----------------------------------------" "------------------" "------------------")
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %18s  %18s" "$_label" "${SUMMARY_PREFETCH_CTRL_US["baseline|$_label"]:-"-"}" "${SUMMARY_PREFETCH_CTRL_US["sdma|$_label"]:-"-"}")
            echo "$_row"
        done

        _hdr=$(printf "  %-40s  %18s  %18s" "Example" "Baseline Issue(us)" "SDMA Issue(us)")
        _sep=$(printf "  %-40s  %18s  %18s" "----------------------------------------" "------------------" "------------------")
        echo ""
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %18s  %18s" "$_label" "${SUMMARY_PREFETCH_ISSUE_US["baseline|$_label"]:-"-"}" "${SUMMARY_PREFETCH_ISSUE_US["sdma|$_label"]:-"-"}")
            echo "$_row"
        done

        _hdr=$(printf "  %-40s  %18s  %18s" "Example" "Baseline DupInstr" "SDMA DupInstr")
        _sep=$(printf "  %-40s  %18s  %18s" "----------------------------------------" "------------------" "------------------")
        echo ""
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %18s  %18s" "$_label" "${SUMMARY_PREFETCH_ISSUE_DUP_INSTR["baseline|$_label"]:-"-"}" "${SUMMARY_PREFETCH_ISSUE_DUP_INSTR["sdma|$_label"]:-"-"}")
            echo "$_row"
        done
    else
        _mode_name="${RUN_MODES[0]}"
        _mode_title=$(printf "%s" "$_mode_name" | tr '[:lower:]' '[:upper:]')
        _hdr=$(printf "  %-40s  %18s  %18s  %18s" "Example" "${_mode_title} Ctrl(us)" "${_mode_title} Issue(us)" "${_mode_title} DupInstr")
        _sep=$(printf "  %-40s  %18s  %18s  %18s" "----------------------------------------" "------------------" "------------------" "------------------")
        echo "$_hdr"; echo "$_sep"
        for _i in "${!SUMMARY_NAMES[@]}"; do
            _label="${SUMMARY_NAMES[$_i]}"
            _row=$(printf "  %-40s  %18s  %18s  %18s" "$_label" "${SUMMARY_PREFETCH_CTRL_US["${RUN_MODES[0]}|$_label"]:-"-"}" "${SUMMARY_PREFETCH_ISSUE_US["${RUN_MODES[0]}|$_label"]:-"-"}" "${SUMMARY_PREFETCH_ISSUE_DUP_INSTR["${RUN_MODES[0]}|$_label"]:-"-"}")
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

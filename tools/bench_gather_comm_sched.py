#!/usr/bin/env python3
"""
Batch benchmark for fake_kernel_comm_sched: gather operator performance.

Tests MTE (sync TGATHER), SDMA (async TGET_ASYNC), and Hybrid (scheduler-
adaptive) strategies across multiple total communication sizes on 4 cards.

Usage:
    python tools/bench_gather_comm_sched.py --platform a2a3 --first-device 4
    python tools/bench_gather_comm_sched.py --platform a2a3sim
    python tools/bench_gather_comm_sched.py --strategies mte sdma --sizes 4K 16K 64K
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

STRATEGY_NAMES = ["mte", "sdma", "hybrid"]

# Total communication volumes in bytes (all ranks combined, float32 gather)
SIZE_LABELS = ["1K", "4K", "8K", "16K", "32K", "64K", "256K"]
SIZE_BYTES = {
    "1K":   1 * 1024,
    "4K":   4 * 1024,
    "8K":   8 * 1024,
    "16K": 16 * 1024,
    "32K": 32 * 1024,
    "64K": 64 * 1024,
    "256K": 256 * 1024,
}

# Gather func_ids in the profiling data
FUNC_ID_GATHER_SYNC  = 1  # GatherSync  (MTE / TGATHER)
FUNC_ID_GATHER_ASYNC = 2  # GatherAsync (SDMA / TGET_ASYNC)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_OUTPUTS_DIR  = _PROJECT_ROOT / "outputs"

_KERNELS_DIR = _PROJECT_ROOT / "examples" / "tensormap_and_ringbuffer" / \
               "fake_kernel_comm_sched" / "kernels"
_GOLDEN_PATH  = _PROJECT_ROOT / "examples" / "tensormap_and_ringbuffer" / \
               "fake_kernel_comm_sched" / "golden.py"
_RUNNER       = _PROJECT_ROOT / "examples" / "scripts" / "multi_card_run_example.py"
_CONVERTER    = _PROJECT_ROOT / "tools" / "swimlane_converter.py"


def _gather_count_for(total_bytes: int, n_ranks: int) -> int:
    """Elements per rank for a given total comm volume (float32)."""
    return total_bytes // (n_ranks * 4)


# ---------------------------------------------------------------------------
# Run one benchmark case
# ---------------------------------------------------------------------------

def _find_newest_perf_file(root_device: int, after_mtime: float) -> Path | None:
    """Return the newest perf_swimlane_*_d{root_device}.json created after after_mtime."""
    if not _OUTPUTS_DIR.exists():
        return None
    pattern = f"perf_swimlane_*_d{root_device}.json"
    candidates = [
        p for p in _OUTPUTS_DIR.glob(pattern)
        if p.stat().st_mtime > after_mtime
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_converter(perf_file: Path) -> str:
    """Run swimlane_converter.py on perf_file; return combined stdout+stderr."""
    cmd = [
        sys.executable, str(_CONVERTER),
        str(perf_file),
        "-k", str(_KERNELS_DIR),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr


def _parse_gather_stats(converter_output: str) -> dict | None:
    """
    Parse gather func_id stats from swimlane_converter output.

    The statistics table has lines like:
        1        GatherSync   1      200.00/300.00   50.00/75.00  ...
        2        GatherAsync  1      120.00/180.00   30.00/45.00  ...

    Returns a dict with keys:
        func_id, func_name, count, avg_exec_us, avg_latency_us
    or None if no gather task was found.
    """
    # Match: func_id  name  count  total_exec/total_lat  avg_exec/avg_lat  ...
    pattern = re.compile(
        r'^\s*([12])\s+(\S+)\s+(\d+)\s+[\d.]+/[\d.]+\s+([\d.]+)/([\d.]+)',
        re.MULTILINE
    )
    best = None
    for m in pattern.finditer(converter_output):
        func_id   = int(m.group(1))
        func_name = m.group(2)
        count     = int(m.group(3))
        avg_exec  = float(m.group(4))
        avg_lat   = float(m.group(5))
        if func_id in (FUNC_ID_GATHER_SYNC, FUNC_ID_GATHER_ASYNC) and count > 0:
            best = {
                "func_id":        func_id,
                "func_name":      func_name,
                "count":          count,
                "avg_exec_us":    avg_exec,
                "avg_latency_us": avg_lat,
            }
    return best


def run_case(
    strategy: str,
    size_label: str,
    n_ranks: int,
    first_device: int,
    platform: str,
    verbose: bool,
) -> dict:
    """Run one (strategy, size) case and return result dict."""
    total_bytes   = SIZE_BYTES[size_label]
    gather_count  = _gather_count_for(total_bytes, n_ranks)
    root_device   = first_device  # rank 0 = first_device

    print(f"  [{strategy:6s}] {size_label:5s}  gather_count={gather_count:6d} ...", end="", flush=True)

    env = os.environ.copy()
    env["GATHER_COUNT"]    = str(gather_count)
    env["GATHER_STRATEGY"] = strategy
    env["N_DEVICES"]       = str(n_ranks)
    env["FIRST_DEVICE"]    = str(first_device)

    cmd = [
        sys.executable, str(_RUNNER),
        "-k", str(_KERNELS_DIR),
        "-g", str(_GOLDEN_PATH),
        "--n-devices", str(n_ranks),
        "--first-device", str(first_device),
        "-p", platform,
        "--enable-profiling",
        "--silent",
    ]

    before_mtime = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=not verbose, text=True)

    result = {
        "strategy":       strategy,
        "size":           size_label,
        "total_bytes":    total_bytes,
        "gather_count":   gather_count,
        "n_ranks":        n_ranks,
        "run_ok":         proc.returncode == 0,
        "func_id":        None,
        "func_name":      None,
        "avg_exec_us":    None,
        "avg_latency_us": None,
    }

    if not result["run_ok"]:
        print(f"  FAILED (exit {proc.returncode})")
        if verbose and proc.stderr:
            print(proc.stderr[-800:])
        return result

    # Find profiling file
    perf_file = None
    for _ in range(10):          # wait up to ~5s for async profiling write
        perf_file = _find_newest_perf_file(root_device, before_mtime)
        if perf_file:
            break
        time.sleep(0.5)

    if not perf_file:
        print("  WARN: no perf file found")
        return result

    converter_out = _run_converter(perf_file)
    stats = _parse_gather_stats(converter_out)

    if stats:
        result.update({
            "func_id":        stats["func_id"],
            "func_name":      stats["func_name"],
            "avg_exec_us":    stats["avg_exec_us"],
            "avg_latency_us": stats["avg_latency_us"],
        })
        kernel_tag = "GatherSync→MTE" if stats["func_id"] == FUNC_ID_GATHER_SYNC else "GatherAsync→SDMA"
        print(f"  exec={stats['avg_exec_us']:.1f}us  lat={stats['avg_latency_us']:.1f}us  [{kernel_tag}]")
    else:
        print("  WARN: gather stats not found in converter output")
        if verbose:
            print(converter_out[-600:])

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    header = (
        f"{'Strategy':8s}  {'Size':5s}  {'GATHER_COUNT':12s}  "
        f"{'Avg_Exec(us)':>13s}  {'Avg_Lat(us)':>12s}  {'Actual_Kernel'}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("Gather Performance Summary")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in results:
        exec_str = f"{r['avg_exec_us']:.2f}" if r["avg_exec_us"] is not None else "N/A"
        lat_str  = f"{r['avg_latency_us']:.2f}" if r["avg_latency_us"] is not None else "N/A"
        if r["func_name"]:
            kernel_tag = (
                f"{r['func_name']} (→MTE)"  if r["func_id"] == FUNC_ID_GATHER_SYNC
                else f"{r['func_name']} (→SDMA)"
            )
        else:
            kernel_tag = "FAILED" if not r["run_ok"] else "N/A"

        print(
            f"{r['strategy']:8s}  {r['size']:5s}  {r['gather_count']:12d}  "
            f"{exec_str:>13s}  {lat_str:>12s}  {kernel_tag}"
        )

    print("=" * len(header))


def save_csv(results: list[dict], out_path: Path) -> None:
    fields = [
        "strategy", "size", "total_bytes", "gather_count", "n_ranks",
        "run_ok", "func_id", "func_name", "avg_exec_us", "avg_latency_us",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch benchmark: fake_kernel_comm_sched gather performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/bench_gather_comm_sched.py --platform a2a3 --first-device 4
  python tools/bench_gather_comm_sched.py --platform a2a3sim
  python tools/bench_gather_comm_sched.py --strategies mte sdma --sizes 4K 64K
        """,
    )
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim"],
                        help="Platform (default: a2a3)")
    parser.add_argument("--first-device", type=int, default=0,
                        help="First device ID (default: 0)")
    parser.add_argument("--n-devices", type=int, default=4,
                        help="Number of devices/ranks (default: 4)")
    parser.add_argument("--strategies", nargs="+", default=STRATEGY_NAMES,
                        choices=STRATEGY_NAMES,
                        help="Strategies to test (default: all)")
    parser.add_argument("--sizes", nargs="+", default=SIZE_LABELS,
                        choices=list(SIZE_BYTES.keys()),
                        help="Total comm sizes to test (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show subprocess output")
    parser.add_argument("--output-dir", type=Path, default=_OUTPUTS_DIR,
                        help="Output directory for CSV (default: outputs/)")

    args = parser.parse_args()

    print(f"Benchmark: fake_kernel_comm_sched gather")
    print(f"  platform={args.platform}  first_device={args.first_device}  n_devices={args.n_devices}")
    print(f"  strategies={args.strategies}")
    print(f"  sizes={args.sizes}")
    print()

    results = []
    for strategy in args.strategies:
        print(f"--- Strategy: {strategy.upper()} ---")
        for size_label in args.sizes:
            r = run_case(
                strategy=strategy,
                size_label=size_label,
                n_ranks=args.n_devices,
                first_device=args.first_device,
                platform=args.platform,
                verbose=args.verbose,
            )
            results.append(r)
        print()

    print_summary(results)

    # Save CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"bench_gather_{ts}.csv"
    save_csv(results, csv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

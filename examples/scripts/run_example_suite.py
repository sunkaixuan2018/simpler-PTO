#!/usr/bin/env python3
"""Discover and run selected PTO examples with persistent logs and reports."""

import argparse
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Optional

if __package__:
    from .example_suite_report import ExampleResult, load_failed_examples, write_reports
else:
    from example_suite_report import ExampleResult, load_failed_examples, write_reports


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLATFORMS = ("a2a3sim", "a2a3")


@dataclass(frozen=True)
class Example:
    name: str
    path: Path

    @property
    def runtime(self) -> str:
        return self.name.split("/", 1)[0]


def discover_examples(examples_root: Path) -> list[Example]:
    """Find valid examples without loading their Python configuration."""
    examples = []
    for runtime in sorted(examples_root.iterdir()):
        if runtime.name == "scripts" or not runtime.is_dir() or runtime.is_symlink():
            continue
        for directory, subdirectories, files in os.walk(runtime):
            subdirectories[:] = sorted(
                name for name in subdirectories
                if not name.startswith(".") and name not in {"kernels", "__pycache__"}
            )
            path = Path(directory)
            if "golden.py" in files and (path / "kernels" / "kernel_config.py").is_file():
                name = path.relative_to(examples_root).as_posix()
                examples.append(Example(name, path))
    return sorted(examples, key=lambda item: item.name)


def select_examples(
    examples: list[Example],
    runtimes: list[str],
    include: list[str],
    exclude: list[str],
) -> list[Example]:
    available_runtimes = {example.runtime for example in examples}
    unknown_runtimes = set(runtimes) - available_runtimes
    if unknown_runtimes:
        raise ValueError(f"Unknown runtime(s): {', '.join(sorted(unknown_runtimes))}")

    return [
        example
        for example in examples
        if (not runtimes or example.runtime in runtimes)
        and (not include or any(fnmatchcase(example.name, pattern) for pattern in include))
        and not any(fnmatchcase(example.name, pattern) for pattern in exclude)
    ]


def build_command(example: Example, args: argparse.Namespace, project_root: Path) -> list[str]:
    command = [
        sys.executable,
        str(project_root / "examples" / "scripts" / "run_example.py"),
        "--kernels",
        str(example.path / "kernels"),
        "--golden",
        str(example.path / "golden.py"),
        "--platform",
        args.platform,
    ]
    if args.device is not None:
        command.extend(["--device", str(args.device), "--first-device", str(args.device)])
    if args.all:
        command.append("--all")
    elif args.case is not None:
        command.extend(["--case", args.case])
    return command


def _stop_process(process: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
    process.wait()


def run_example(
    example: Example,
    command: list[str],
    log_path: Path,
    project_root: Path,
    timeout: Optional[float],
) -> ExampleResult:
    started = time.monotonic()
    exit_code = None
    detail = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            try:
                exit_code = process.wait(timeout=timeout)
                status = "passed" if exit_code == 0 else "failed"
                if exit_code:
                    detail = f"Runner exited with status {exit_code}"
            except subprocess.TimeoutExpired:
                _stop_process(process)
                status = "timed_out"
                detail = f"Example exceeded the {timeout:g}s timeout"
            except KeyboardInterrupt:
                _stop_process(process)
                status = "interrupted"
                detail = "Run interrupted by the user"
    except OSError as error:
        status = "error"
        detail = str(error)
    return ExampleResult(
        example.name,
        command,
        status,
        time.monotonic() - started,
        log=log_path.name if log_path.is_file() else None,
        exit_code=exit_code,
        detail=detail,
    )


def _device_id(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("device ID must be non-negative")
    return number


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive number of seconds")
    return seconds


def make_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=PLATFORMS, default="a2a3sim")
    parser.add_argument("-r", "--runtime", action="append", default=[], help="Limit to this runtime (repeatable)")
    parser.add_argument(
        "--include", action="append", default=[],
        help="Include name glob, for example 'host_build_graph/*attention*' (repeatable)",
    )
    parser.add_argument(
        "--exclude", action="append", default=[],
        help="Exclude name glob after applying includes (repeatable)",
    )
    parser.add_argument("-d", "--device", type=_device_id, help="Override the device or first device ID")
    parser.add_argument("--timeout", type=_positive_seconds, help="Stop each example after this many seconds")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first unsuccessful example")
    parser.add_argument("--list", action="store_true", help="Print selected example names without running")
    parser.add_argument("--rerun-failed", type=Path, metavar="REPORT", help="Select failures from a JSON report")
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "outputs" / "example-suite",
        help="Parent directory for the run's logs and reports",
    )
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument("--all", action="store_true", help="Run all declared cases for each example")
    cases.add_argument("--case", help="Run the specified case for each selected example")
    return parser


def run_suite(
    examples: list[Example], args: argparse.Namespace, project_root: Path, output_dir: Path,
) -> list[ExampleResult]:
    """Run examples serially and record any cases left unstarted."""
    results = []
    stop_reason = None
    for index, example in enumerate(examples, start=1):
        command = build_command(example, args, project_root)
        if stop_reason:
            results.append(ExampleResult(example.name, command, "skipped", 0, detail=stop_reason))
            continue

        print(f"[{index}/{len(examples)}] {example.name}", flush=True)
        log_path = output_dir / f"{index:04d}.log"
        result = run_example(example, command, log_path, project_root, args.timeout)
        results.append(result)
        status = result.status.upper()
        suffix = f" ({result.log})" if result.log else ""
        print(f"  {status} in {result.duration_seconds:.2f}s{suffix}", flush=True)
        if result.status == "interrupted" or (args.fail_fast and result.status != "passed"):
            stop_reason = f"Stopped after {example.name} ({result.status})"
    return results


def main(argv: Optional[list[str]] = None, project_root: Path = PROJECT_ROOT) -> int:
    parser = make_parser(project_root)
    args = parser.parse_args(argv)
    try:
        examples = discover_examples(project_root / "examples")
        selected = select_examples(examples, args.runtime, args.include, args.exclude)
        if args.rerun_failed:
            failures = load_failed_examples(args.rerun_failed, args.platform)
            if not failures:
                print("The report contains no failed examples.")
                return 0
            missing = failures - {example.name for example in examples}
            if missing:
                raise ValueError(f"Examples from the report are missing: {', '.join(sorted(missing))}")
            selected = [example for example in selected if example.name in failures]
        if not selected:
            raise ValueError("No examples match the selection")
    except (ValueError, OSError) as error:
        parser.error(str(error))

    if args.list:
        for example in selected:
            print(example.name)
        return 0

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir))
        print(f"Run output: {output_dir.resolve()}", flush=True)
        results = run_suite(selected, args, project_root, output_dir)
        write_reports(output_dir, args.platform, started_at, results)
    except KeyboardInterrupt:
        return 130
    except OSError as error:
        print(f"Could not save example suite output: {error}", file=sys.stderr)
        return 1

    counts = Counter(result.status for result in results)
    print("Results: " + ", ".join(f"{count} {status}" for status, count in sorted(counts.items())))
    if counts["interrupted"]:
        return 130
    return 0 if counts["passed"] == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

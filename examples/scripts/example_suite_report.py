"""Result formats for the PTO example suite."""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = 1
FAILED_STATUSES = frozenset({"failed", "timed_out", "error"})
RESULT_STATUSES = FAILED_STATUSES | {"passed", "skipped", "interrupted"}
INVALID_XML = re.compile(r"[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


@dataclass(frozen=True)
class ExampleResult:
    example: str
    command: list[str]
    status: str
    duration_seconds: float
    log: Optional[str] = None
    exit_code: Optional[int] = None
    detail: Optional[str] = None


def _write_atomic(path: Path, data: bytes) -> None:
    temp_path = path.with_suffix(".tmp")
    try:
        temp_path.write_bytes(data)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_junit(path: Path, results: list[ExampleResult]) -> None:
    suite = ET.Element(
        "testsuite",
        name="pto-examples",
        tests=str(len(results)),
        failures=str(sum(result.status == "failed" for result in results)),
        errors=str(sum(result.status in {"error", "timed_out", "interrupted"} for result in results)),
        skipped=str(sum(result.status == "skipped" for result in results)),
        time=f"{sum(result.duration_seconds for result in results):.3f}",
    )
    for result in results:
        runtime, _, name = result.example.partition("/")
        case = ET.SubElement(
            suite,
            "testcase",
            classname=INVALID_XML.sub("", runtime),
            name=INVALID_XML.sub("", name),
            time=f"{result.duration_seconds:.3f}",
        )
        message = INVALID_XML.sub("", result.detail or result.status)
        if result.status == "failed":
            ET.SubElement(case, "failure", message=message)
        elif result.status in {"error", "timed_out", "interrupted"}:
            ET.SubElement(case, "error", type=result.status, message=message)
        elif result.status == "skipped":
            ET.SubElement(case, "skipped", message=message)
        if result.log:
            ET.SubElement(case, "system-out").text = f"Log: {result.log}"
    ET.indent(suite)
    _write_atomic(path, ET.tostring(suite, encoding="utf-8", xml_declaration=True))


def write_reports(
    output_dir: Path,
    platform: str,
    started_at: str,
    results: list[ExampleResult],
) -> None:
    """Save JSON for reruns and JUnit XML for test reporting systems."""
    report = {
        "schema_version": SCHEMA_VERSION,
        "platform": platform,
        "started_at": started_at,
        "results": [asdict(result) for result in results],
    }
    _write_atomic(output_dir / "report.json", (json.dumps(report, indent=2) + "\n").encode())
    _write_junit(output_dir / "junit.xml", results)


def load_failed_examples(report_path: Path, platform: str) -> set[str]:
    """Read failure names from a report produced for the requested platform."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported example suite report version in {report_path}")
    if report.get("platform") != platform:
        raise ValueError(f"Report platform {report.get('platform')!r} does not match {platform!r}")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Missing results in {report_path}")

    failed = set()
    seen = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(f"Invalid result in {report_path}")
        name, status = result.get("example"), result.get("status")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(status, str)
            or status not in RESULT_STATUSES
        ):
            raise ValueError(f"Invalid example name or status in {report_path}")
        if name in seen:
            raise ValueError(f"Duplicate example {name!r} in {report_path}")
        seen.add(name)
        if status in FAILED_STATUSES:
            failed.add(name)
    return failed

"""Exercise example selection, execution, and report consumption without a device."""

import json
import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from example_suite_report import ExampleResult, load_failed_examples, write_reports
from run_example_suite import Example, build_command, discover_examples, main, run_example, select_examples


def make_example(root, name, source=""):
    path = root / "examples" / name
    kernels = path / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "kernel_config.py").write_text("", encoding="utf-8")
    (path / "golden.py").write_text(source, encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path):
    scripts = tmp_path / "examples" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_example.py").write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import sys
            import time
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--golden", required=True)
            parser.add_argument("--kernels", required=True)
            parser.add_argument("--platform", required=True)
            parser.add_argument("--device", type=int)
            parser.add_argument("--first-device", type=int)
            parser.add_argument("--all", action="store_true")
            parser.add_argument("--case")
            args = parser.parse_args()
            name = Path(args.golden).parent.name
            print(json.dumps(vars(args)), flush=True)
            print(f"stderr for {name}", file=sys.stderr, flush=True)
            if name == "sleep":
                time.sleep(30)
            if name == "bad":
                raise SystemExit(7)
            """
        ),
        encoding="utf-8",
    )
    return tmp_path


def read_reports(root):
    run_dir, = (root / "outputs" / "example-suite").glob("run-*")
    return (
        run_dir,
        json.loads((run_dir / "report.json").read_text(encoding="utf-8")),
        ET.parse(run_dir / "junit.xml").getroot(),
    )


def test_discovery_requires_both_files_and_keeps_stable_names(project):
    last = make_example(project, "runtime-b/z")
    first = make_example(project, "runtime-a/group/a")
    incomplete = make_example(project, "runtime-a/incomplete")
    (incomplete / "kernels" / "kernel_config.py").unlink()
    missing_golden = make_example(project, "runtime-c/incomplete")
    (missing_golden / "golden.py").unlink()
    make_example(project, "runtime-a/.hidden/ignored")
    make_example(project, "runtime-a/__pycache__/ignored")
    make_example(project, "scripts/_deps/external")
    make_example(project, "runtime-a/kernels/nested")

    assert discover_examples(project / "examples") == [
        Example("runtime-a/group/a", first),
        Example("runtime-b/z", last),
    ]


def test_filters_compose_without_duplicate_results(project):
    for name in ("host/vector", "host/vector_small", "host/scalar", "device/vector"):
        make_example(project, name)

    examples = discover_examples(project / "examples")
    selected = select_examples(examples, ["host"], ["*/vector*", "host/vector"], ["*_small"])
    assert [example.name for example in selected] == ["host/vector"]
    assert [example.name for example in select_examples(examples, ["device", "host"], ["*/vector"], [])] == [
        "device/vector",
        "host/vector",
    ]
    with pytest.raises(ValueError, match="Unknown runtime.*missing"):
        select_examples(examples, ["missing"], [], [])


def test_command_preserves_spaces_and_forwards_device_and_case(project):
    path = make_example(project, "host/vector example")
    args = SimpleNamespace(platform="a2a3", device=3, case="large batch", all=False)
    command = build_command(Example("host/vector example", path), args, project)
    assert command == [
        sys.executable,
        str(project / "examples" / "scripts" / "run_example.py"),
        "--kernels",
        str(path / "kernels"),
        "--golden",
        str(path / "golden.py"),
        "--platform",
        "a2a3",
        "--device",
        "3",
        "--first-device",
        "3",
        "--case",
        "large batch",
    ]


def test_list_only_reads_paths(project, capsys):
    make_example(project, "host/vector", "raise RuntimeError('must not import')")
    make_example(project, "device/communication")

    assert main(["--list", "--runtime", "host"], project_root=project) == 0
    assert capsys.readouterr().out == "host/vector\n"
    assert not (project / "outputs").exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--timeout", "0"], "positive"),
        (["--timeout", "-5"], "positive"),
        (["--timeout", "NaN"], "positive"),
        (["--timeout", "inf"], "positive"),
        (["--device", "-1"], "non-negative"),
        (["--case", "big", "--all"], "not allowed with argument"),
        (["--exclude", "*"], "No examples match"),
        (["--runtime", "UNKNOWN"], "Unknown runtime"),
    ],
)
def test_invalid_options_fail_before_writing_output(project, capsys, arguments, message):
    make_example(project, "host/vector")

    with pytest.raises(SystemExit) as error:
        main(arguments, project_root=project)

    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (project / "outputs").exists()


def test_suite_preserves_logs_and_records_result_details(project, capsys):
    make_example(project, "host/good")
    make_example(project, "host/bad")

    assert main(["--all", "--device", "2"], project_root=project) == 1
    run_dir, report, junit = read_reports(project)
    failed, passed = report["results"]
    assert report["schema_version"] == 1
    assert report["platform"] == "a2a3sim"
    assert report["started_at"].endswith("+00:00")
    assert [(row["example"], row["status"], row["exit_code"]) for row in (failed, passed)] == [
        ("host/bad", "failed", 7),
        ("host/good", "passed", 0),
    ]
    assert failed["detail"] == "Runner exited with status 7"
    assert passed["detail"] is None
    assert passed["duration_seconds"] > 0
    assert passed["command"][-1] == "--all"
    passed_log = (run_dir / passed["log"]).read_text(encoding="utf-8").splitlines()
    actual_args = json.loads(passed_log[0])
    assert actual_args["device"] == actual_args["first_device"] == 2
    assert actual_args["all"]
    assert passed_log[1] == "stderr for good"
    assert junit.attrib["tests"] == "2"
    assert junit.attrib["failures"] == "1"
    assert junit.find("testcase/failure").attrib["message"] == failed["detail"]
    assert "1 failed, 1 passed" in capsys.readouterr().out


def test_timeout_and_fail_fast_record_the_unstarted_example(project):
    make_example(project, "host/sleep")
    make_example(project, "host/z-last")

    assert main(["--timeout", "0.5", "--fail-fast"], project_root=project) == 1
    run_dir, report, junit = read_reports(project)
    timeout, skipped = report["results"]
    assert timeout["status"] == "timed_out"
    assert timeout["exit_code"] is None
    assert timeout["duration_seconds"] < 10
    assert "0.5s timeout" in timeout["detail"]
    assert (run_dir / timeout["log"]).is_file()
    assert skipped["status"] == "skipped"
    assert skipped["log"] is None
    assert skipped["duration_seconds"] == 0
    assert junit.attrib["errors"] == "1"
    assert junit.attrib["skipped"] == "1"
    assert junit.find("testcase/error").attrib["type"] == "timed_out"


def test_rerun_selects_report_failures_and_preserves_original_report(project):
    make_example(project, "host/bad")
    make_example(project, "host/good")
    other_output = project / "rerun"

    assert main([], project_root=project) == 1
    first_dir, original, _ = read_reports(project)
    original_report = first_dir / "report.json"
    assert main(
        ["--rerun-failed", str(original_report), "--output-dir", str(other_output), "--case", "new-case"],
        project_root=project,
    ) == 1
    rerun_report, = other_output.glob("run-*/report.json")
    rerun = json.loads(rerun_report.read_text(encoding="utf-8"))["results"]
    assert [row["example"] for row in rerun] == ["host/bad"]
    assert rerun[0]["command"][-2:] == ["--case", "new-case"]
    assert json.loads(original_report.read_text(encoding="utf-8")) == original


def test_rerun_without_failures_creates_no_new_run(project, capsys):
    make_example(project, "host/good")
    assert main([], project_root=project) == 0
    first_dir, _, _ = read_reports(project)

    assert main(["--rerun-failed", str(first_dir / "report.json")], project_root=project) == 0
    assert read_reports(project)[0] == first_dir
    assert "no failed examples" in capsys.readouterr().out


def test_rerun_rejects_removed_examples_and_wrong_platform(project, capsys):
    make_example(project, "host/good")
    write_reports(
        project,
        "a2a3",
        "2026-01-01T00:00:00+00:00",
        [ExampleResult("host/removed", [], "failed", 0)],
    )
    report = project / "report.json"

    with pytest.raises(SystemExit) as wrong_platform:
        main(["--rerun-failed", str(report)], project_root=project)
    assert wrong_platform.value.code == 2
    assert "does not match" in capsys.readouterr().err

    with pytest.raises(SystemExit) as missing_example:
        main(["--platform", "a2a3", "--rerun-failed", str(report)], project_root=project)
    assert missing_example.value.code == 2
    assert "host/removed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"schema_version": 2}, "Unsupported"),
        ({"schema_version": 1, "platform": "a2a3"}, "Missing results"),
        ({"schema_version": 1, "platform": "a2a3", "results": [None]}, "Invalid result"),
        (
            {"schema_version": 1, "platform": "a2a3", "results": [{"example": "host/a", "status": []}]},
            "Invalid example name or status",
        ),
        (
            {
                "schema_version": 1,
                "platform": "a2a3",
                "results": [
                    {"example": "host/a", "status": "passed"},
                    {"example": "host/a", "status": "failed"},
                ],
            },
            "Duplicate example",
        ),
    ],
)
def test_rerun_rejects_invalid_reports(tmp_path, data, message):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_failed_examples(path, "a2a3")


def test_launch_error_is_saved_as_a_test_error(project):
    path = make_example(project, "host/good")
    command = [str(project / "nonexistent-python")]
    result = run_example(Example("host/good", path), command, project / "run.log", project, None)
    write_reports(project, "a2a3sim", "2026-01-01T00:00:00+00:00", [result])

    assert result.status == "error"
    assert result.detail
    assert result.exit_code is None
    assert ET.parse(project / "junit.xml").find("testcase/error").attrib["type"] == "error"


def test_junit_escapes_xml_metacharacters_and_discards_control_characters(tmp_path):
    result = ExampleResult("host/case&<>", [], "error", 0, detail="command failed\x1b<&")
    write_reports(tmp_path, "a2a3", "2026-01-01T00:00:00+00:00", [result])

    testcase = ET.parse(tmp_path / "junit.xml").find("testcase")
    assert testcase.attrib["name"] == "case&<>"
    assert testcase.find("error").attrib["message"] == "command failed<&"


def test_suite_executable_lists_examples_from_outside_project(project):
    script = SCRIPTS / "run_example_suite.py"
    result = subprocess.run(
        [sys.executable, str(script), "--list", "--include", "host_build_graph/vector_example"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["host_build_graph/vector_example"]


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("", "Default (default)\n"),
        (
            "ALL_CASES = {'small': {}, 'large': {}}\nDEFAULT_CASE = 'large'\n",
            "small\nlarge (default)\n",
        ),
    ],
)
def test_single_example_lists_cases_without_kernel_config(tmp_path, source, output):
    golden = tmp_path / "golden.py"
    golden.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_example.py"), "--list-cases", "-g", str(golden)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == output


def test_single_example_reports_invalid_default(tmp_path):
    golden = tmp_path / "golden.py"
    golden.write_text("ALL_CASES = {'small': {}}\nDEFAULT_CASE = 'missing'\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_example.py"), "--list-cases", "-g", str(golden)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "Default case 'missing' not found" in result.stderr

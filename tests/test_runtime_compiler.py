"""Tests for the CMake runtime build driver."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from runtime_compiler import RuntimeCompiler


def test_executable_detection_uses_path_without_running_a_process(monkeypatch, tmp_path):
    executable = tmp_path / ("test-tool.exe" if os.name == "nt" else "test-tool")
    executable.touch()
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=AssertionError("spawned a process")))

    assert RuntimeCompiler._find_executable(executable.name)
    assert RuntimeCompiler._find_executable(str(executable))
    assert not RuntimeCompiler._find_executable("missing-pto-test-tool")


@pytest.mark.parametrize("output_dir", [".", "Release"])
@pytest.mark.parametrize("parallel_level", [None, "", "3"])
def test_build_uses_cmake_generator_and_respects_parallel_level(
    monkeypatch, tmp_path, output_dir, parallel_level,
):
    if parallel_level is None:
        monkeypatch.delenv("CMAKE_BUILD_PARALLEL_LEVEL", raising=False)
    else:
        monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", parallel_level)
    monkeypatch.setattr("runtime_compiler.multiprocessing.cpu_count", lambda: 64)
    monkeypatch.setattr("runtime_compiler.tempfile.tempdir", str(tmp_path))
    compiler = RuntimeCompiler.__new__(RuntimeCompiler)
    calls = []

    def build_step(command, cwd, target, step):
        calls.append((command, cwd, target, step))
        if step == "CMake build":
            artifact_dir = Path(cwd) / output_dir
            artifact_dir.mkdir(exist_ok=True)
            (artifact_dir / "libhost_runtime.so").write_bytes(b"compiled host")

    monkeypatch.setattr(compiler, "_run_build_step", build_step)
    output = compiler._run_compilation(
        "source", ["-DCMAKE_CXX_COMPILER=g++"], "libhost_runtime.so", "HOST"
    )

    assert output == b"compiled host"
    configure, build = calls
    assert configure[0] == ["cmake", "source", "-DCMAKE_CXX_COMPILER=g++"]
    assert build[0][:6] == ["cmake", "--build", ".", "--config", "Release", "--verbose"]
    assert build[0][6:] == (["--parallel", "32"] if parallel_level is None else [])
    assert configure[1] == build[1]
    assert Path(build[1]).parent == tmp_path
    assert not Path(build[1]).exists()


def test_missing_output_names_both_locations(monkeypatch):
    compiler = RuntimeCompiler.__new__(RuntimeCompiler)
    monkeypatch.setattr(compiler, "_run_build_step", Mock())
    with pytest.raises(FileNotFoundError, match="aicore_kernel.o") as error:
        compiler._run_compilation("source", [], "aicore_kernel.o")
    assert str(Path("Release") / "aicore_kernel.o") in str(error.value)


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        ("build output\n", "", ["build output"]),
        ("output from generator\n", "compiler error\n", ["output from generator", "compiler error"]),
        ("", "", []),
    ],
)
def test_build_failure_includes_exit_status_and_diagnostics(monkeypatch, stdout, stderr, expected):
    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess(
        ["cmake"], 2, stdout, stderr
    )))
    compiler = RuntimeCompiler.__new__(RuntimeCompiler)
    with pytest.raises(RuntimeError, match=r"CMake build failed for HOST \(exit 2\)") as error:
        compiler._run_build_step(["cmake"], ".", "HOST", "CMake build")
    for diagnostic in expected:
        assert diagnostic in str(error.value)


def test_start_failure_names_executable_and_preserves_cause(monkeypatch):
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=FileNotFoundError("cmake missing")))
    compiler = RuntimeCompiler.__new__(RuntimeCompiler)
    with pytest.raises(RuntimeError, match="Could not start CMake configuration for AICPU using cmake") as error:
        compiler._run_build_step(["cmake"], ".", "AICPU", "CMake configuration")
    assert isinstance(error.value.__cause__, FileNotFoundError)

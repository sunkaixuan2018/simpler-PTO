import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List
from toolchain import Toolchain, CCECToolchain, Aarch64GxxToolchain, GxxToolchain
import env_manager
import multiprocessing

logger = logging.getLogger(__name__)


class BuildTarget:
    """CMake build target: composes a Toolchain with a source directory and output name.

    Used by RuntimeCompiler for CMake-based multi-file compilation.
    """

    def __init__(self, toolchain: Toolchain, root_dir: str, binary_name: str):
        self.toolchain = toolchain
        self._root_dir = os.path.abspath(root_dir)
        self._binary_name = binary_name

    def get_root_dir(self) -> str:
        return self._root_dir

    def get_binary_name(self) -> str:
        return self._binary_name

    def gen_cmake_args(self, include_dirs: List[str], source_dirs: List[str]) -> List[str]:
        """Generate CMake arguments list from toolchain args + custom directories."""
        inc = ";".join(os.path.abspath(d) for d in include_dirs)
        src = ";".join(os.path.abspath(d) for d in source_dirs)
        return self.toolchain.get_cmake_args() + [
            f"-DCUSTOM_INCLUDE_DIRS={inc}",
            f"-DCUSTOM_SOURCE_DIRS={src}",
        ]


class RuntimeCompiler:
    """
    Runtime compiler for compiling runtime binaries for multiple target platforms.

    Supports three target types:
    1. aicore - AICore accelerator kernels
    2. aicpu - AICPU device task scheduler
    3. host - Host runtime library

    Platform determines which toolchains and CMake directories are used:
    - "a2a3": ccec for aicore, aarch64 cross-compiler for aicpu, gcc for host
    - "a2a3sim": all use host gcc/g++ (builds host-compatible .so files)

    Use get_instance() to get a cached instance per platform.
    """
    _instances = {}

    @classmethod
    def get_instance(cls, platform: str = "a2a3") -> "RuntimeCompiler":
        """Get or create a RuntimeCompiler instance for the given platform."""
        if platform not in cls._instances:
            cls._instances[platform] = cls(platform)
        return cls._instances[platform]

    def __init__(self, platform: str = "a2a3"):
        self.platform = platform
        self.project_root = Path(__file__).parent.parent
        self.platform_dir = self.project_root / "src" / "platform" / platform

        if not self.platform_dir.is_dir():
            raise ValueError(
                f"Platform '{platform}' not found at {self.platform_dir}"
            )

        if platform == "a2a3":
            self._init_a2a3()
        elif platform == "a2a3sim":
            self._init_a2a3sim()
        else:
            raise ValueError(
                f"Unknown platform: {platform}. Supported: a2a3, a2a3sim"
            )

    def _init_a2a3(self):
        """Initialize toolchains for real a2a3 hardware."""
        env_manager.ensure("ASCEND_HOME_PATH")

        # AICore: Bisheng CCE compiler
        ccec = CCECToolchain()
        self.aicore_target = BuildTarget(
            ccec, str(self.platform_dir / "aicore"), "aicore_kernel.o"
        )

        # AICPU: aarch64 cross-compiler
        aarch64 = Aarch64GxxToolchain()
        self.aicpu_target = BuildTarget(
            aarch64, str(self.platform_dir / "aicpu"), "libaicpu_kernel.so"
        )

        # Host: standard gcc/g++
        self._ensure_host_compilers()
        host_gxx = GxxToolchain()
        self.host_target = BuildTarget(
            host_gxx, str(self.platform_dir / "host"), "libhost_runtime.so"
        )

    def _init_a2a3sim(self):
        """Initialize toolchains for simulation platform.
        All targets use host gcc/g++ with platform-specific CMake dirs.
        No Ascend SDK required.
        """
        self._ensure_host_compilers()
        gxx = GxxToolchain()

        self.aicore_target = BuildTarget(
            gxx, str(self.platform_dir / "aicore"), "libaicore_kernel.so"
        )
        self.aicpu_target = BuildTarget(
            gxx, str(self.platform_dir / "aicpu"), "libaicpu_kernel.so"
        )
        self.host_target = BuildTarget(
            gxx, str(self.platform_dir / "host"), "libhost_runtime.so"
        )

    def _ensure_host_compilers(self):
        if not self._find_executable("gcc"):
            raise FileNotFoundError("Host C compiler not found: gcc. Please install gcc.")
        if not self._find_executable("g++"):
            raise FileNotFoundError("Host C++ compiler not found: g++. Please install g++.")

    @staticmethod
    def _find_executable(name: str) -> bool:
        """Check for an executable path or look it up in PATH."""
        return shutil.which(name) is not None

    def compile(
        self,
        target_platform: str,
        include_dirs: List[str],
        source_dirs: List[str],
    ) -> bytes:
        """
        Compile binary for the specified target platform.

        Args:
            target_platform: Target platform ("aicore", "aicpu", or "host")
            include_dirs: List of include directory paths
            source_dirs: List of source directory paths

        Returns:
            Compiled binary data as bytes

        Raises:
            ValueError: If target platform is invalid
            RuntimeError: If CMake configuration or build fails
            FileNotFoundError: If output binary not found
        """
        if target_platform == "aicore":
            target = self.aicore_target
        elif target_platform == "aicpu":
            target = self.aicpu_target
        elif target_platform == "host":
            target = self.host_target
        else:
            raise ValueError(
                f"Invalid target platform: {target_platform}. "
                "Must be 'aicore', 'aicpu', or 'host'."
            )

        cmake_args = target.gen_cmake_args(include_dirs, source_dirs)
        cmake_source_dir = target.get_root_dir()
        binary_name = target.get_binary_name()

        return self._run_compilation(
            cmake_source_dir, cmake_args, binary_name, platform=target_platform.upper()
        )

    def _run_build_step(
        self,
        cmd: List[str],
        cwd: str,
        platform: str,
        step_name: str,
    ) -> None:
        """Run a CMake step with logging and error handling.

        Args:
            cmd: Command and arguments
            cwd: Working directory
            platform: Target name for logging
            step_name: Step name for logging

        Raises:
            RuntimeError: If the step fails or executable not found
        """
        logger.info(f"[{platform}] Running {step_name}...")
        logger.debug(f"  Working directory: {cwd}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, cwd=cwd, check=False, capture_output=True, text=True, errors="replace"
            )

            if result.stdout and logger.isEnabledFor(10):  # DEBUG = 10
                logger.debug(f"[{platform}] {step_name} stdout:")
                logger.debug(result.stdout)
            if result.stderr and logger.isEnabledFor(10):
                logger.debug(f"[{platform}] {step_name} stderr:")
                logger.debug(result.stderr)

            if result.returncode != 0:
                diagnostics = "\n".join(
                    output.strip() for output in (result.stdout, result.stderr) if output.strip()
                )
                message = f"{step_name} failed for {platform} (exit {result.returncode})"
                if diagnostics:
                    message = f"{message}:\n{diagnostics}"
                logger.error("[%s] %s", platform, message)
                raise RuntimeError(message)
        except OSError as error:
            raise RuntimeError(
                f"Could not start {step_name} for {platform} using {cmd[0]}: {error}"
            ) from error

    def _run_compilation(
        self,
        cmake_source_dir: str,
        cmake_args: List[str],
        binary_name: str,
        platform: str = "AICore"
    ) -> bytes:
        """
        Run CMake configuration and build in a temporary directory.

        Args:
            cmake_source_dir: Path to CMake source directory
            cmake_args: CMake command-line arguments
            binary_name: Name of output binary
            platform: Platform name for logging

        Returns:
            Compiled binary data as bytes

        Raises:
            RuntimeError: If CMake configuration or build fails
            FileNotFoundError: If output binary not found
        """
        with tempfile.TemporaryDirectory(prefix=f"{platform.lower()}_build_") as build_dir:
            cmake_cmd = ["cmake", cmake_source_dir] + cmake_args
            self._run_build_step(cmake_cmd, build_dir, platform, "CMake configuration")

            build_cmd = ["cmake", "--build", ".", "--config", "Release", "--verbose"]
            if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
                build_cmd.extend(["--parallel", str(min(multiprocessing.cpu_count(), 32))])
            self._run_build_step(build_cmd, build_dir, platform, "CMake build")

            paths = (Path(build_dir) / binary_name, Path(build_dir) / "Release" / binary_name)
            for path in paths:
                if path.is_file():
                    return path.read_bytes()

            expected = ", ".join(str(path) for path in paths)
            raise FileNotFoundError(f"Compiled binary {binary_name} not found. Checked: {expected}")


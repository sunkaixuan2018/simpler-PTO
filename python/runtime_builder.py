import importlib.util
import logging
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from runtime_compiler import RuntimeCompiler
from kernel_compiler import KernelCompiler

logger = logging.getLogger(__name__)
BUILD_TARGETS = ("aicore", "aicpu", "host")


def _read_build_config(config_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("build_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "BUILD_CONFIG", None)
    if not isinstance(config, Mapping):
        raise ValueError(f"{config_path}: BUILD_CONFIG must be a mapping")

    resolved = {}
    for target in BUILD_TARGETS:
        target_config = config.get(target)
        if not isinstance(target_config, Mapping):
            raise ValueError(f"{config_path}: BUILD_CONFIG[{target!r}] must be a mapping")

        groups = []
        for key in ("include_dirs", "source_dirs"):
            paths = target_config.get(key)
            if (
                not isinstance(paths, Sequence)
                or isinstance(paths, (str, bytes))
                or any(not isinstance(p, (str, PathLike)) for p in paths)
            ):
                raise ValueError(f"{config_path}: {target}.{key} must be a sequence of paths")
            groups.append([str((config_path.parent / path).resolve()) for path in paths])
        resolved[target] = tuple(groups)
    return resolved


class RuntimeBuilder:
    """Discovers and builds runtime implementations from src/runtime/.

    Accepts a platform selection to provide correctly configured
    RuntimeCompiler and KernelCompiler instances. Runtime and platform
    are orthogonal — the same runtime (e.g., host_build_graph) can
    be compiled for any platform (e.g., a2a3, a2a3sim).
    """

    def __init__(self, platform: str = "a2a3"):
        """
        Initialize RuntimeBuilder with platform selection.

        Args:
            platform: Target platform ("a2a3" or "a2a3sim")
        """
        if platform not in ("a2a3", "a2a3sim"):
            raise ValueError(f"Unknown platform: {platform}. Supported: a2a3, a2a3sim")
        self.platform = platform

        runtime_root = Path(__file__).parent.parent
        self.runtime_root = runtime_root
        self.runtime_dir = runtime_root / "src" / "runtime"

        # Discover available runtime implementations
        self._runtimes = {}
        if self.runtime_dir.is_dir():
            for entry in sorted(self.runtime_dir.iterdir()):
                config_path = entry / "build_config.py"
                if entry.is_dir() and config_path.is_file():
                    self._runtimes[entry.name] = config_path

        self._runtime_compiler = None
        self._kernel_compiler = None
        self._compiler_lock = Lock()

    def get_runtime_compiler(self) -> RuntimeCompiler:
        """Return the RuntimeCompiler configured for this platform."""
        with self._compiler_lock:
            if self._runtime_compiler is None:
                self._runtime_compiler = RuntimeCompiler.get_instance(platform=self.platform)
            return self._runtime_compiler

    def get_kernel_compiler(self) -> KernelCompiler:
        """Return the KernelCompiler configured for this platform."""
        with self._compiler_lock:
            if self._kernel_compiler is None:
                self._kernel_compiler = KernelCompiler(platform=self.platform)
            return self._kernel_compiler

    def list_runtimes(self) -> list:
        """Return names of discovered runtime implementations."""
        return list(self._runtimes.keys())

    def build(self, name: str) -> tuple:
        """
        Build a specific runtime implementation by name.

        Args:
            name: Name of the runtime implementation (e.g. 'host_build_graph')

        Returns:
            Tuple of (host_binary, aicpu_binary, aicore_binary) as bytes

        Raises:
            ValueError: If the named runtime is not found
        """
        if name not in self._runtimes:
            available = ", ".join(self._runtimes.keys()) or "(none)"
            raise ValueError(
                f"Runtime '{name}' not found. Available runtimes: {available}"
            )

        build_config = _read_build_config(self._runtimes[name])
        compiler = self.get_runtime_compiler()

        # Compile all three targets in parallel
        logger.info("Compiling AICore, AICPU, Host in parallel...")

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                target: executor.submit(compiler.compile, target, *build_config[target])
                for target in BUILD_TARGETS
            }
            binaries = {target: futures[target].result() for target in BUILD_TARGETS}

        logger.info("Build complete!")
        return binaries["host"], binaries["aicpu"], binaries["aicore"]

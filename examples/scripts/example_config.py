"""Load Python example configuration and read its test case metadata."""

import importlib.util
import sys
from pathlib import Path


def load_module_from_path(module_path: Path, module_name: str):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_golden_cases(module):
    """Return the case definitions and default name, including legacy defaults."""
    return getattr(module, "ALL_CASES", {"Default": {}}), getattr(module, "DEFAULT_CASE", "Default")

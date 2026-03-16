"""
SDMA workspace Python bindings for async communication.

Wraps libsdma_helper.so (C++ helper around SdmaWorkspaceManager).
Build: cd examples/scripts/sdma_helper && mkdir build && cd build && cmake .. && make
Then run with CANN env: source .../set_env.sh

Usage:
    from sdma_bindings import sdma_init, sdma_finalize
    workspace_ptr = sdma_init()   # returns int (device address)
    # ... pass workspace_ptr to kernel ...
    sdma_finalize()
"""

import ctypes
from ctypes import c_void_p, c_int
from pathlib import Path
from typing import Optional

_lib = None


def _find_helper_so() -> Optional[Path]:
    """Locate libsdma_helper.so next to this script or in sdma_helper/build."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "sdma_helper" / "build" / "libsdma_helper.so",
        script_dir / "build" / "libsdma_helper.so",
        script_dir / "libsdma_helper.so",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load():
    global _lib
    if _lib is not None:
        return

    path = _find_helper_so()
    if path is None:
        raise RuntimeError(
            "libsdma_helper.so not found. Build it with CANN env set:\n"
            "  source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash\n"
            "  cd examples/scripts/sdma_helper && mkdir build && cd build\n"
            "  cmake .. && make\n"
            "Then run your script with the same CANN env."
        )
    try:
        _lib = ctypes.CDLL(str(path))
    except OSError as e:
        raise RuntimeError(
            f"Failed to load {path}: {e}\n"
            "Ensure CANN env is set (source .../setenv.bash)."
        ) from e

    _lib.sdma_helper_init.argtypes = [ctypes.POINTER(c_void_p)]
    _lib.sdma_helper_init.restype = c_int

    _lib.sdma_helper_finalize.argtypes = []
    _lib.sdma_helper_finalize.restype = None


def sdma_init() -> int:
    """
    Initialize SDMA workspace on the current device.

    Returns:
        Device address of the SDMA workspace (as int).
    """
    _load()
    workspace = c_void_p()
    ret = _lib.sdma_helper_init(ctypes.byref(workspace))
    if ret != 0:
        raise RuntimeError("sdma_helper_init failed")
    return workspace.value or 0


def sdma_finalize() -> None:
    """Finalize SDMA workspace and release resources."""
    _load()
    _lib.sdma_helper_finalize()

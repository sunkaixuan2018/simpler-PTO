"""
Helpers for activating the vendored minimal SDMA provider.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def _parse_prefetch_mode() -> Optional[str]:
    mode = os.environ.get("PTO_SDMA_PREFETCH_MODE")
    if mode:
        normalized = mode.strip().lower()
        aliases = {
            "0": "baseline",
            "1": "twoslot",
            "2": "sdma",
            "3": "sdma_fake",
            "fake": "sdma_fake",
        }
        return aliases.get(normalized, normalized)

    enable = os.environ.get("PTO_ENABLE_SDMA_PREFETCH")
    if not enable:
        return None
    normalized = enable.strip().lower()
    if normalized in {"0", "false", "off", "no"}:
        return "twoslot"
    return "sdma"


def _is_valid_provider_root(root: Path) -> bool:
    return (root / "x86_64-linux" / "lib64" / "libopapi.so").exists() and (root / "opp").exists()


def discover_provider_root(project_root: Path) -> Optional[Path]:
    env_root = os.environ.get("PTO_SDMA_LEGACY_ROOT") or os.environ.get("PTO_SDMA_PROVIDER_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _is_valid_provider_root(candidate):
            return candidate
        logger.warning("SDMA provider root invalid: %s", candidate)
    vendored_root = project_root / "_deps" / "sdma_legacy_provider"
    if _is_valid_provider_root(vendored_root):
        return vendored_root
    return None


def resolve_sdma_runtime_env(
    *,
    project_root: Path,
    platform: str,
    n_devices: int,
    requires_comm: bool,
) -> Dict[str, str]:
    if platform != "a2a3":
        return {}

    mode = _parse_prefetch_mode()

    if requires_comm and n_devices > 1:
        if mode in (None, "sdma"):
            logger.info(
                "SDMA provider: forcing PTO_SDMA_PREFETCH_MODE=twoslot for multi-card comm run"
            )
            return {"PTO_SDMA_PREFETCH_MODE": "twoslot"}
        return {}

    if mode in {"baseline", "twoslot", "sdma_fake"}:
        return {}

    provider_root = discover_provider_root(project_root)
    if provider_root is None:
        return {}

    opp_dir = str((provider_root / "opp").resolve())
    logger.info("SDMA provider: using vendored provider at %s", provider_root)
    return {
        "PTO_SDMA_PROVIDER_ROOT": str(provider_root),
        "ASCEND_OPP_PATH": opp_dir,
    }

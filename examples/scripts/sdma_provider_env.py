"""
Helpers for configuring SDMA prefetch runtime environment.
"""

from __future__ import annotations

import logging
import os
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

def resolve_sdma_runtime_env(
    *,
    project_root,
    platform: str,
    n_devices: int,
    requires_comm: bool,
) -> Dict[str, str]:
    del project_root
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

    logger.info("SDMA provider: no extra provider bundle needed for current HAL-based setup")
    return {}

"""
comm_gather: validation is done in C++ (root checks gathered data).
generate_inputs / compute_golden are for code_runner interface only; no Python-side comparison.
"""

import numpy as np

ALL_CASES = {"Default": {}}
DEFAULT_CASE = "Default"
__outputs__ = []  # no outputs compared in Python


def generate_inputs(params):
    """Return minimal placeholder; C++ executable owns data and validation."""
    return []


def compute_golden(tensors, params):
    """No-op; validation is in C++ (RunGatherKernel on root)."""
    pass

"""
Utility functions and modules for needle path planning.

This package provides utilities for:
- Finite Element Analysis
- Path manipulation and validation
- Common mathematical operations
"""

from .fe_model import run_model, ExitCode
from .path_utils import (
    interpolate_path,
    validate_path,
    compute_curvature,
    smooth_path,
    convert_to_point32,
    convert_to_point,
    resample_path
)

__all__ = [
    'run_model',
    'ExitCode',
    'interpolate_path',
    'validate_path',
    'compute_curvature',
    'smooth_path',
    'convert_to_point32',
    'convert_to_point',
    'resample_path'
]
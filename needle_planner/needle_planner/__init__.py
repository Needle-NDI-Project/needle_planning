"""
Needle Planner ROS2 Package.

This package provides various approaches for needle path planning,
including geometric, machine learning, and hybrid methods.
"""

from importlib.metadata import version, PackageNotFoundError
from . import planners
from . import utils
from . import models

try:
    __version__ = version("needle_planner")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.1.0"

# Define public interface
__all__ = [
    'planners',
    'utils',
    'models',
    '__version__'
]
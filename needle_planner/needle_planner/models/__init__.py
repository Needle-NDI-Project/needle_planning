"""
Pre-trained models for needle path planning.

This module contains pre-trained machine learning models:
- Linear regression model for path prediction
- Neural network models for path generation
- Hybrid model components
"""

import os
import pickle
from pathlib import Path

def get_model_path(model_name: str) -> str:
    """Get the full path to a model file."""
    module_path = Path(__file__).parent
    return str(module_path / model_name)

def load_pickle_model(model_name: str):
    """Load a pickled model file."""
    model_path = get_model_path(model_name)
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Default model paths
LINEAR_REGRESSION_MODEL = 'linear_regression_model.pkl'
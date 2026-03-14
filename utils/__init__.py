"""
Utilities for temporal feature extraction and mental health classification.

This package contains modules for:
- Data loading and preprocessing
- Feature extraction from social media posts
- Model training and evaluation
- Cross-validation and analysis
"""

# Import key classes and functions for easy access
from .data_loader import DataLoader, verify_splits
from .feature_extractor import TemporalFeatureExtractor
from .trainer import Trainer, TimeSeriesDataset, EarlyStopping
from .evaluation import ModelEvaluator
from .models import (
    get_model,
    LSTMClassifier,
    FCNClassifier,
    InceptionTimeClassifier,
    MCNNClassifier
)

# Version info
__version__ = '1.0.0'
__author__ = 'AQZ'

# Define what gets imported with "from utils import *"
__all__ = [
    # Data loading
    'DataLoader',
    'verify_splits',
    
    # Feature extraction
    'TemporalFeatureExtractor',
    
    # Training
    'Trainer',
    'TimeSeriesDataset',
    'EarlyStopping',
    
    # Evaluation
    'ModelEvaluator',
    
    # Models
    'get_model',
    'LSTMClassifier',
    'FCNClassifier',
    'InceptionTimeClassifier',
    'MCNNClassifier',
]

# Optional: Print import confirmation (useful for debugging)
# print(f"✅ Utils package v{__version__} loaded")
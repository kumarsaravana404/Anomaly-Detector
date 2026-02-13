"""
Login Anomaly Detection System
================================

A production-ready cybersecurity project for detecting suspicious login attempts
using machine learning (Isolation Forest).

Modules:
--------
- data_generator: Generate realistic synthetic login data
- preprocessor: Data preprocessing and feature engineering
- anomaly_detector: Isolation Forest model training and prediction
- visualizer: Visualization utilities

Author: Senior Data Scientist & Cybersecurity Engineer
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_generator import LoginDataGenerator
from .preprocessor import LoginDataPreprocessor
from .anomaly_detector import LoginAnomalyDetector
from .visualizer import AnomalyVisualizer

__all__ = [
    "LoginDataGenerator",
    "LoginDataPreprocessor",
    "LoginAnomalyDetector",
    "AnomalyVisualizer",
]

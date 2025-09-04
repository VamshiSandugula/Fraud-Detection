"""
Configuration file for Financial Crime Analytics System
Centralized configuration for all system parameters
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "file_path": "creditcard_2023.csv",
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True,
    "sample_size": None,  # Set to integer for sampling, None for full dataset
}

# Feature configuration
FEATURE_CONFIG = {
    "id_column": "id",
    "target_column": "Class",
    "amount_column": "Amount",
    "feature_columns": [f"V{i}" for i in range(1, 29)],  # V1 to V28
    "categorical_columns": [],
    "numerical_columns": [f"V{i}" for i in range(1, 29)] + ["Amount"],
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "scaler": "robust",  # Options: "standard", "robust", "minmax"
    "handle_missing": True,
    "handle_outliers": True,
    "outlier_method": "iqr",  # Options: "iqr", "zscore", "isolation_forest"
    "outlier_threshold": 1.5,
}

# SMOTE configuration
SMOTE_CONFIG = {
    "enabled": True,
    "random_state": 42,
    "k_neighbors": 5,
    "sampling_strategy": "auto",  # Options: "auto", "minority", float
}

# Model configuration
MODEL_CONFIG = {
    "isolation_forest": {
        "enabled": True,
        "n_estimators": 100,
        "max_samples": "auto",
        "contamination": 0.1,
        "random_state": 42,
        "max_features": 1.0,
        "bootstrap": False,
    },
    "random_forest": {
        "enabled": True,
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "logistic_regression": {
        "enabled": True,
        "random_state": 42,
        "class_weight": "balanced",
        "max_iter": 1000,
        "C": 1.0,
        "solver": "lbfgs",
        "penalty": "l2",
    },
    "xgboost": {
        "enabled": True,
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "scale_pos_weight": 1,
        "n_jobs": -1,
    },
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "cv_folds": 5,
    "scoring_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "threshold_range": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "confidence_level": 0.95,
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "style": "seaborn-v0_8",
    "palette": "husl",
    "figure_size": (15, 12),
    "dpi": 300,
    "save_format": "png",
    "interactive": True,
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "page_title": "Financial Crime Analytics Dashboard",
    "page_icon": "🚨",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_upload_size": 200,  # MB
    "auto_refresh": False,
    "refresh_interval": 30,  # seconds
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "chunk_size": 10000,
    "parallel_processing": True,
    "n_jobs": -1,
    "memory_efficient": True,
    "cache_results": True,
}

# Fraud thresholds
FRAUD_THRESHOLDS = {
    "high_risk": 0.8,
    "medium_risk": 0.6,
    "low_risk": 0.4,
    "alert_threshold": 0.7,
    "block_threshold": 0.9,
}

def get_config():
    """Get configuration based on current environment"""
    config = {
        "dataset": DATASET_CONFIG,
        "features": FEATURE_CONFIG,
        "preprocessing": PREPROCESSING_CONFIG,
        "smote": SMOTE_CONFIG,
        "models": MODEL_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "dashboard": DASHBOARD_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "thresholds": FRAUD_THRESHOLDS,
    }
    
    return config

if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Number of sections: {len(config)}")
    print(f"Sections: {list(config.keys())}")

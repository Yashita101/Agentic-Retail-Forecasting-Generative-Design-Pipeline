"""
forecaster.py
Subtask 1 Module: Performance Analytics and Sales Volume Forecasting.
Encapsulates model evaluation and production inference logic.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
from typing import TypedDict


class ModelMetrics(TypedDict):
    """Structured container for regression metrics."""
    mae: float
    rmse: float
    r2: float


class SalesForecaster:
    """
    Encapsulates the ML champion model artifacts for catalog-level
    predictions and distinct style assortment selection.
    """
    _DIVERSITY_COLUMNS = ["product_type_name"]
    _PRED_UNITS_COL = "predicted_30d_sales_volume"

    def __init__(self, models_dir: str | Path = "models"):
        """Loads serialized XGBoost champion model from models directory."""
        self._models_dir = Path(models_dir)
        self._load_champion_model()

    def _load_champion_model(self) -> None:
        """Loads serialized champion artifacts. Relies on Jupyter export TURN."""
        print("Subtask 1: Loading ML artifacts...")
        model_path = self._models_dir / "xgboost_sales_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Champion model not found at {model_path}. Run Jupyter export TURN.")
        self._model = joblib.load(model_path)

    def evaluate_performance(self, processed_data_dir: str | Path) -> ModelMetrics:
        """
        Subtask 1 Requirement: Evaluate performance on test data.
        Loads test splits, predicts, restores unit scale (expm1), and calculates metrics.
        """
        data_dir = Path(processed_data_dir)
        print("Subtask 1: Evaluating performance on held-out test data...")

        # Load held-out test splits from Jupyter TURN export
        X_test = pd.read_parquet(data_dir / "X_test.parquet")
        y_test_log = pd.read_parquet(data_dir / "y_test.parquet")["target_sales_log"].values
        
        # In-distribution predictions (log scale)
        preds_log = self._model.predict(X_test)
        
        # RESTORE REAL UNIT SCALE (Inverting np.log1p) - CRITICAL for real units metrics
        y_true_real = np.expm1(y_test_log)
        y_pred_real = np.expm1(preds_log)
        
        # Calculate real-unit metrics
        metrics: ModelMetrics = {
            "mae": float(mean_absolute_error(y_true_real, y_pred_real)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_real, y_pred_real))),
            "r2": float(r2_score(y_true_real, y_pred_real)),
        }
        return metrics

    def predict_top_performers(self, processed_data_dir: str | Path, limit_distinct: int = 4) -> pd.DataFrame:
        """
        Runs production inference on full catalog and enforces assortment diversity.
        Returns top performes across distinct product categories.
        """
        data_dir = Path(processed_data_dir)
        print("Subtask 1: Running forecasting inference on full catalog...")

        # Load optimized inference artifacts
        X_final = pd.read_parquet(data_dir / "X_all_active.parquet")
        catalog_context = pd.read_parquet(data_dir / "catalog_context.parquet")
        
        # Production predictions (log scale) -> Units scale
        preds_log = self._model.predict(X_final)
        predicted_units = np.expm1(preds_log)
        
        # Merge predictions back to context metadata
        results = catalog_context.copy()
        results[self._PRED_UNITS_COL] = predicted_units
        
        # DIVERSITY LOGIC: Rank descending and keep highest rank per product type.
        # This prevents selecting three hoodies if hoodies fill ranks 1, 2, 3.
        top_performers = (
            results.sort_values(by=self._PRED_UNITS_COL, ascending=False)
            .drop_duplicates(subset=self._DIVERSITY_COLUMNS) # ENFORCE DISTINCT PRODUCTS
            .head(limit_distinct)
            .reset_index(drop=True)
        )
        return top_performers
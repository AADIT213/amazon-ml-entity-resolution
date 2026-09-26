"""
Phase 5: Baseline Pair Matching Model Module.

Implements interpretable, lightweight classifier for entity pair resolution
supporting probability estimation and F0.5 threshold optimization (TRD.md §6-7).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import LIGHTWEIGHT_FEATURE_NAMES


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Compute F-beta score from precision and recall.
    For beta=0.5, precision is weighted twice as heavily as recall.
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    return float((1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute TP, FP, FN, TN, precision, recall, and F0.5 score.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f_beta = compute_f_beta(precision, recall, beta=beta)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "predicted_matches": tp + fp,
        "actual_matches": tp + fn,
    }


class BaselineMatchClassifier:
    """
    Interpretable baseline logistic regression pair classifier.
    Handles feature normalization, probability estimation, and thresholding.
    """

    def __init__(
        self,
        c: float = 1.0,
        class_weight: Optional[str] = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.c = c
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.c,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.best_threshold: float = 0.5
        self.feature_names: List[str] = list(LIGHTWEIGHT_FEATURE_NAMES)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaselineMatchClassifier:
        """Fit scaler and logistic regression classifier on training features."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probability for each pair."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary match labels using specified or best threshold."""
        thresh = self.best_threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def evaluate_thresholds(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        thresholds: Sequence[float] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
        beta: float = 0.5,
    ) -> pd.DataFrame:
        """
        Evaluate precision, recall, F-beta across probability thresholds.
        """
        probs = self.predict_proba(X_val)
        records = []

        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            metrics = compute_classification_metrics(y_val, preds, beta=beta)
            metrics["threshold"] = thresh
            records.append(metrics)

        df = pd.DataFrame(records)
        cols = ["threshold", "precision", "recall", "f_beta", "predicted_matches", "tp", "fp", "fn", "tn"]
        df = df[cols]

        # Automatically record best threshold
        best_row = df.loc[df["f_beta"].idxmax()]
        self.best_threshold = float(best_row["threshold"])

        return df

    def get_feature_importances(self) -> Dict[str, float]:
        """Return learned logistic regression weights per feature."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        weights = self.model.coef_[0]
        return {name: float(w) for name, w in zip(self.feature_names, weights)}

    def save(self, filepath: str) -> None:
        """Serialize model and scaler to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> BaselineMatchClassifier:
        """Deserialize model from disk."""
        return joblib.load(filepath)

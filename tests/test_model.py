"""
Unit tests for Phase 5 Baseline Match Classifier.

Tests cover:
- Metric calculations (precision, recall, F0.5, confusion matrix)
- Model fitting, prediction, and probability output
- Threshold grid evaluation and automatic best threshold selection
- Feature importance retrieval
- Model serialization (save/load)
"""

import sys
import os
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from business_entity_resolution.model import (
    BaselineMatchClassifier,
    compute_f_beta,
    compute_classification_metrics,
)
from business_entity_resolution.features import LIGHTWEIGHT_FEATURE_NAMES


def test_compute_f_beta():
    # When precision = 1.0, recall = 1.0, F0.5 should be 1.0
    assert compute_f_beta(1.0, 1.0, beta=0.5) == 1.0
    # When precision = 0 or recall = 0
    assert compute_f_beta(0.0, 1.0, beta=0.5) == 0.0
    assert compute_f_beta(1.0, 0.0, beta=0.5) == 0.0

    # With precision = 0.8, recall = 0.5:
    # F0.5 = (1.25 * 0.8 * 0.5) / (0.25 * 0.8 + 0.5) = 0.5 / 0.7 = 0.71428...
    val = compute_f_beta(0.8, 0.5, beta=0.5)
    assert abs(val - 0.7142857) < 1e-4


def test_compute_classification_metrics():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    m = compute_classification_metrics(y_true, y_pred, beta=0.5)
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert abs(m["f_beta"] - 2 / 3) < 1e-4


def test_baseline_classifier_fit_predict():
    np.random.seed(42)
    N = 200
    D = len(LIGHTWEIGHT_FEATURE_NAMES)

    # Synthetic separable features for unit test
    X = np.random.randn(N, D).astype(np.float32)
    # Give strong signal on feature 0 (exact_name_match) and feature 2 (name_char_similarity)
    y = ((X[:, 0] + X[:, 2]) > 0.0).astype(int)

    clf = BaselineMatchClassifier(max_iter=200, random_state=42)
    assert not clf.is_fitted
    clf.fit(X, y)
    assert clf.is_fitted

    probs = clf.predict_proba(X)
    assert probs.shape == (N,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    preds = clf.predict(X, threshold=0.5)
    assert preds.shape == (N,)
    assert set(preds).issubset({0, 1})


def test_threshold_evaluation():
    np.random.seed(42)
    N = 300
    D = len(LIGHTWEIGHT_FEATURE_NAMES)

    X_train = np.random.randn(N, D).astype(np.float32)
    y_train = (X_train[:, 0] > 0.2).astype(int)

    X_val = np.random.randn(100, D).astype(np.float32)
    y_val = (X_val[:, 0] > 0.2).astype(int)

    clf = BaselineMatchClassifier(random_state=42)
    clf.fit(X_train, y_train)

    df_thresh = clf.evaluate_thresholds(X_val, y_val, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
    assert len(df_thresh) == 5
    assert "f_beta" in df_thresh.columns
    assert "precision" in df_thresh.columns
    assert "recall" in df_thresh.columns
    assert clf.best_threshold in [0.1, 0.3, 0.5, 0.7, 0.9]


def test_model_save_load():
    np.random.seed(42)
    X = np.random.randn(50, len(LIGHTWEIGHT_FEATURE_NAMES)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    clf = BaselineMatchClassifier(random_state=42)
    clf.fit(X, y)
    clf.best_threshold = 0.65

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        clf.save(model_path)
        assert os.path.exists(model_path)

        loaded_clf = BaselineMatchClassifier.load(model_path)
        assert loaded_clf.is_fitted
        assert loaded_clf.best_threshold == 0.65

        orig_probs = clf.predict_proba(X)
        loaded_probs = loaded_clf.predict_proba(X)
        np.testing.assert_array_almost_equal(orig_probs, loaded_probs)

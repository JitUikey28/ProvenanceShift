"""
Offline unit tests for Phase 5 persona representation validation.

Verifies:
    - Layer selection based on validation metrics only
    - Direction stability resampling (cosine similarity)
    - Random direction null distribution
    - Train-fitted PCA isolation
    - End-to-end validation report generation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.activations.validation import (
    compute_direction_stability,
    compute_train_fitted_pca,
    run_persona_validation,
    run_random_direction_distribution,
    select_best_layer,
)


# =====================================================================
# Validation-Based Layer Selection
# =====================================================================

class TestSelectBestLayer:

    def test_selects_best_validation_layer(self) -> None:
        df = pd.DataFrame({
            "layer": [0, 1, 2, 3],
            "val_roc_auc": [0.55, 0.72, 0.95, 0.88],
            "test_roc_auc": [0.50, 0.99, 0.80, 0.85],  # test has max at layer 1, but val has max at layer 2!
        })
        selected = select_best_layer(df, criterion="val_roc_auc")
        # MUST select layer 2 based on validation, NOT layer 1 from test!
        assert selected == 2

    def test_invalid_criterion_raises(self) -> None:
        df = pd.DataFrame({"layer": [0, 1], "val_roc_auc": [0.5, 0.6]})
        with pytest.raises(ValueError, match="not found"):
            select_best_layer(df, criterion="invalid_criterion")


# =====================================================================
# Stability Resampling
# =====================================================================

class TestDirectionStability:

    def test_stability_on_consistent_signal(self) -> None:
        """Synthetic data with clear 1st-dimension signal should have high stability."""
        n_train = 20
        dim = 8
        labels = np.array([1, 0] * (n_train // 2))

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_train, dim)) * 0.1
        X[labels == 1, 0] += 5.0
        X[labels == 0, 0] -= 5.0

        res = compute_direction_stability(X, labels, n_resamples=50, seed=42)
        assert res["mean_cosine_similarity"] > 0.95
        assert res["ci_lower"] > 0.90
        assert res["resamples_collected"] > 40


# =====================================================================
# Random Direction Null Distribution
# =====================================================================

class TestRandomDirectionDistribution:

    def test_null_distribution_properties(self) -> None:
        n_test = 20
        dim = 8
        labels = np.array([1, 0] * (n_test // 2))
        X = np.random.randn(n_test, dim)

        res = run_random_direction_distribution(X, labels, n_directions=50, seed=42)
        assert res["n_directions"] == 50
        # Mean random ROC-AUC on arbitrary noise is approximately 0.50
        assert 0.35 <= res["mean_random_roc_auc"] <= 0.65
        assert len(res["raw_aucs"]) == 50


# =====================================================================
# Train-Fitted PCA Isolation
# =====================================================================

class TestTrainFittedPCA:

    def test_pca_fit_on_train_only(self) -> None:
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_val = np.array([[2.0, 3.0]])
        X_test = np.array([[4.0, 5.0]])

        pca, X_tr_p, X_va_p, X_te_p = compute_train_fitted_pca(X_train, X_val, X_test, n_components=2)
        assert X_tr_p.shape == (3, 2)
        assert X_va_p.shape == (1, 2)
        assert X_te_p.shape == (1, 2)
        # Verify PCA mean matches training data mean
        np.testing.assert_array_almost_equal(pca.mean_, np.mean(X_train, axis=0))


# =====================================================================
# Synthetic End-to-End Validation Pipeline
# =====================================================================

class TestPersonaValidationPipeline:

    def test_end_to_end_validation(self, tmp_path: Path) -> None:
        exp_id = "test-persona-val"
        hidden_dim = 8
        n_train = 12
        n_val = 8
        n_test = 8
        total_n = n_train + n_val + n_test

        labels = np.array([1, 0] * (total_n // 2))
        splits = (["train"] * n_train) + (["validation"] * n_val) + (["test"] * n_test)

        prompts = [
            {"prompt_id": f"p_{i}", "persona_label": "assistant" if labels[i] == 1 else "alternative", "split": splits[i]}
            for i in range(total_n)
        ]
        manifest = {
            "experiment_id": exp_id,
            "prompts": prompts,
            "model": {"name": "synthetic-model"},
            "extraction_config": {"layers": [0, 1]},
        }

        rng = np.random.default_rng(42)
        activations = {}
        # Layer 0: noise
        activations[0] = rng.standard_normal((total_n, hidden_dim))
        # Layer 1: signal on dim 0
        l1 = rng.standard_normal((total_n, hidden_dim)) * 0.1
        l1[labels == 1, 0] += 4.0
        l1[labels == 0, 0] -= 4.0
        activations[1] = l1

        df, report, artifacts = run_persona_validation(
            activations_by_layer=activations,
            manifest=manifest,
            experiment_id=exp_id,
            n_random_directions=20,
            n_stability_resamples=20,
            bootstrap_samples=30,
            output_tables_dir=tmp_path / "tables",
            output_figures_dir=tmp_path / "figures",
            output_raw_dir=tmp_path / "raw",
        )

        assert report["selected_layer"] == 1
        assert artifacts["table"].exists()
        assert artifacts["fig_stability"].exists()
        assert artifacts["fig_null"].exists()
        assert artifacts["fig_pca"].exists()
        assert artifacts["metadata"].exists()

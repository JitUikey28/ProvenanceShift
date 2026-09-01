"""
Offline unit tests for candidate persona direction, statistical metrics,
baseline classifiers, controls, and synthetic end-to-end analysis (Phase 4).

All tests run offline on synthetic arrays without GPU or model downloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.activations.analysis import run_layer_sweep, save_analysis_results
from src.activations.direction import (
    bootstrap_confidence_interval,
    compute_cohens_d,
    compute_mean_difference_direction,
    compute_projection_metrics,
    evaluate_classifier,
    generate_random_direction,
    project_representations,
    shuffle_labels,
    train_linear_classifier,
)


# =====================================================================
# Mean Difference Direction
# =====================================================================

class TestMeanDifferenceDirection:

    def test_direction_computation_and_norm(self) -> None:
        """Verify direction vector is normalized and points from alt to asst."""
        # 2 assistant examples at [10, 0], 2 alternative examples at [0, 0]
        X = np.array([
            [10.0, 0.0],
            [10.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        y = np.array([1, 1, 0, 0])

        direction = compute_mean_difference_direction(X, y)
        assert direction.shape == (2,)
        # Unit norm
        assert np.isclose(np.linalg.norm(direction), 1.0)
        # Vector points strictly along positive x-axis [1, 0]
        np.testing.assert_array_almost_equal(direction, np.array([1.0, 0.0]))

    def test_sign_convention_projection(self) -> None:
        """Verify that assistant examples receive higher projection scores than alternative."""
        X = np.array([
            [5.0, 1.0],  # asst
            [-5.0, 1.0], # alt
        ])
        y = np.array([1, 0])

        v = compute_mean_difference_direction(X, y)
        scores = project_representations(X, v)

        assert scores[0] > scores[1]  # Assistant score > Alternative score

    def test_missing_class_raises(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_only_asst = np.array([1, 1])
        with pytest.raises(ValueError, match="No alternative"):
            compute_mean_difference_direction(X, y_only_asst)

    def test_zero_diff_raises(self) -> None:
        # Both classes have identical mean
        X = np.array([[1.0, 2.0], [1.0, 2.0]])
        y = np.array([1, 0])
        with pytest.raises(ValueError, match="zero norm"):
            compute_mean_difference_direction(X, y)


# =====================================================================
# Cohen's d Effect Size & Metrics
# =====================================================================

class TestEffectSizeAndMetrics:

    def test_cohens_d_calculation(self) -> None:
        # Group 1: mean=10, std=0.816; Group 2: mean=8, std=0.816 -> d approx 2.45
        x1 = np.array([9.0, 10.0, 11.0, 10.0])
        x2 = np.array([7.0, 8.0, 9.0, 8.0])
        d = compute_cohens_d(x1, x2)
        assert np.isclose(d, 2.449, atol=0.01)

    def test_projection_metrics_perfect_separation(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([5.0, 4.0, -4.0, -5.0])

        metrics = compute_projection_metrics(y_true, scores)
        assert metrics["accuracy"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["cohens_d"] > 0
        assert metrics["mean_projection_diff"] > 0

    def test_bootstrap_confidence_interval(self) -> None:
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        scores = np.array([2.0, 2.5, 2.2, 1.8, -1.0, -1.5, -2.0, -1.2])

        def dummy_metric(y, s):
            return float(np.mean(s[y == 1]) - np.mean(s[y == 0]))

        ci_lower, ci_upper = bootstrap_confidence_interval(
            dummy_metric, y_true, scores, n_bootstrap=200, seed=42
        )
        assert ci_lower <= ci_upper
        assert ci_lower > 0.0  # Since distributions are clearly separated


# =====================================================================
# Baseline Classifier & Controls
# =====================================================================

class TestClassifierAndControls:

    def test_linear_classifier_fit_and_eval(self) -> None:
        X_train = np.array([[2.0, 3.0], [3.0, 2.0], [-2.0, -3.0], [-3.0, -2.0]])
        y_train = np.array([1, 1, 0, 0])

        clf = train_linear_classifier(X_train, y_train, seed=42)
        eval_res = evaluate_classifier(clf, X_train, y_train)

        assert eval_res["clf_accuracy"] == 1.0
        assert eval_res["clf_roc_auc"] == 1.0

    def test_random_direction(self) -> None:
        r1 = generate_random_direction(dim=16, seed=42)
        r2 = generate_random_direction(dim=16, seed=42)
        r3 = generate_random_direction(dim=16, seed=99)

        assert r1.shape == (16,)
        assert np.isclose(np.linalg.norm(r1), 1.0)
        np.testing.assert_array_almost_equal(r1, r2)
        assert not np.allclose(r1, r3)

    def test_shuffle_labels(self) -> None:
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_shuff = shuffle_labels(y, seed=42)
        assert len(y_shuff) == len(y)
        assert np.sum(y_shuff) == np.sum(y)


# =====================================================================
# Synthetic End-to-End Analysis Test
# =====================================================================

class TestSyntheticEndToEndAnalysis:

    def test_full_analysis_pipeline(self, tmp_path: Path) -> None:
        """Verify full layer sweep, metric calculation, table and figure generation."""
        exp_id = "test-analysis-pipeline"
        hidden_dim = 16

        # Create 3 synthetic layers (0, 1, 2)
        # Layer 1 has strong planted persona signal along dim 0
        # Layer 0 (embedding) has weak signal
        # Layer 2 has moderate signal
        n_train = 12  # 6 asst, 6 alt
        n_val = 8     # 4 asst, 4 alt
        n_test = 8    # 4 asst, 4 alt
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
            "extraction_config": {"layers": [0, 1, 2]},
        }

        rng = np.random.default_rng(42)
        # Base noise
        noise = rng.standard_normal((total_n, hidden_dim))

        activations_by_layer = {}
        # Layer 0: pure noise (random)
        activations_by_layer[0] = noise.copy()

        # Layer 1: strong planted signal on dim 0
        l1 = rng.standard_normal((total_n, hidden_dim)) * 0.1
        l1[labels == 1, 0] += 5.0
        l1[labels == 0, 0] -= 5.0
        activations_by_layer[1] = l1

        # Layer 2: moderate planted signal on dim 0
        l2 = rng.standard_normal((total_n, hidden_dim)) * 0.1
        l2[labels == 1, 0] += 1.0
        l2[labels == 0, 0] -= 1.0
        activations_by_layer[2] = l2

        # Run layer sweep
        df, layer_details = run_layer_sweep(
            activations_by_layer=activations_by_layer,
            manifest=manifest,
            experiment_id=exp_id,
            bootstrap_samples=50,
            seed=42,
            run_controls=True,
        )

        assert len(df) == 3
        assert set(df["layer"]) == {0, 1, 2}

        # Layer 1 should have highest validation ROC-AUC
        best_layer = df.loc[df["val_roc_auc"].idxmax(), "layer"]
        assert best_layer == 1

        # Save artifacts
        saved = save_analysis_results(
            df=df,
            layer_details=layer_details,
            manifest=manifest,
            experiment_id=exp_id,
            output_tables_dir=tmp_path / "tables",
            output_figures_dir=tmp_path / "figures",
            output_raw_dir=tmp_path / "raw",
        )

        assert saved["table"].exists()
        assert saved["fig_accuracy"].exists()
        assert saved["fig_roc_auc"].exists()
        assert saved["fig_cohens_d"].exists()
        assert saved["fig_projection_distribution"].exists()
        assert saved["metadata"].exists()

        # Verify metadata
        with open(saved["metadata"]) as fh:
            meta = json.load(fh)
        assert meta["best_validation_layer"] == 1
        assert meta["best_layer_test_roc_auc"] == 1.0

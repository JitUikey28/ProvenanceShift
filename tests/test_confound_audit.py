# =============================================================================
# Unit Tests — Confound Audit Engine (Phase 5.5)
# =============================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.activations.confound_audit import (
    audit_dataset_properties,
    compute_expanded_training_direction,
    cross_direction_comparison,
    evaluate_control_shifts,
    generate_confound_audit_figures,
    run_expanded_bootstrap_stability,
)


class TestDatasetPropertyAudit:
    def test_audit_dataset_properties(self, tmp_path: Path) -> None:
        prompts = [
            {
                "prompt_id": "p1",
                "split": "train",
                "persona_label": "assistant",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Explain gravity."}
                ],
                "token_metadata": {"input_token_count": 10}
            },
            {
                "prompt_id": "p2",
                "split": "train",
                "persona_label": "alternative",
                "messages": [
                    {"role": "system", "content": "You are a Victorian naturalist describing celestial gravity."},
                    {"role": "user", "content": "Explain gravity."}
                ],
                "token_metadata": {"input_token_count": 14}
            }
        ]
        csv_path = tmp_path / "stats.csv"
        df_summary, raw = audit_dataset_properties(prompts, output_csv_path=csv_path)

        assert csv_path.exists()
        assert isinstance(df_summary, pd.DataFrame)
        assert "char_count" in df_summary["property"].values
        assert "token_count" in df_summary["property"].values
        assert len(raw["raw_records"]) == 2


class TestEvaluateControlShifts:
    def test_control_delta_computation(self) -> None:
        dim = 16
        direction = np.ones(dim) / np.sqrt(dim)
        control_prompts = [
            {
                "control_type": "length_control",
                "condition_name": "length_short",
                "metadata": {"task_id": "t1"},
                "messages": [{"role": "user", "content": "short"}],
            },
            {
                "control_type": "length_control",
                "condition_name": "length_medium",
                "metadata": {"task_id": "t1"},
                "messages": [{"role": "user", "content": "medium"}],
            },
            {
                "control_type": "positive_persona_control",
                "persona_label": "assistant",
                "metadata": {"task_id": "t2"},
                "messages": [{"role": "user", "content": "asst"}],
            },
            {
                "control_type": "positive_persona_control",
                "persona_label": "alternative",
                "metadata": {"task_id": "t2"},
                "messages": [{"role": "user", "content": "alt"}],
            },
        ]
        # Create activations with known projections
        activations = np.zeros((4, dim))
        activations[0] = 1.0  # proj = sqrt(16) = 4.0
        activations[1] = 2.0  # proj = 8.0 -> delta = +4.0
        activations[2] = 3.0  # proj = 12.0
        activations[3] = 1.0  # proj = 4.0 -> delta = +8.0

        df_summary, raw = evaluate_control_shifts(
            control_prompts=control_prompts,
            control_activations=activations,
            direction=direction,
            seed=42,
            n_bootstrap=100,
        )

        assert isinstance(df_summary, pd.DataFrame)
        assert "length_control" in df_summary["control_type"].values
        assert "positive_persona_control" in df_summary["control_type"].values


class TestExpandedStability:
    def test_expanded_training_direction_and_stability(self) -> None:
        n_samples = 40
        dim = 16
        rng = np.random.default_rng(42)

        # Create structured activations
        asst_acts = rng.normal(loc=1.0, scale=0.5, size=(n_samples // 2, dim))
        alt_acts = rng.normal(loc=-1.0, scale=0.5, size=(n_samples // 2, dim))
        X = np.vstack([asst_acts, alt_acts])
        y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

        d_orig = np.ones(dim) / np.sqrt(dim)
        d_exp, cos_sim = compute_expanded_training_direction(X, y, d_orig)

        assert d_exp.shape == (dim,)
        assert np.isclose(np.linalg.norm(d_exp), 1.0)
        assert cos_sim > 0.8  # Well aligned with true direction

        stability = run_expanded_bootstrap_stability(X, y, d_exp, n_resamples=50, seed=42)
        assert stability["mean_cosine_similarity"] > 0.8
        assert stability["n_resamples_collected"] == 50

    def test_cross_direction_comparison(self) -> None:
        dim = 32
        d1 = np.ones(dim) / np.sqrt(dim)
        d2 = np.ones(dim) / np.sqrt(dim)
        res = cross_direction_comparison(d1, d2, n_random=20, seed=42)
        assert np.isclose(res["cos_orig_expanded"], 1.0)
        assert abs(res["mean_cos_orig_rand"]) < 0.3


class TestFigureGeneration:
    def test_generate_all_figures(self, tmp_path: Path) -> None:
        df_raw = pd.DataFrame({
            "persona_label": ["assistant", "alternative"] * 5,
            "char_count": [100, 200] * 5,
            "token_count": [20, 40] * 5,
        })
        df_controls = pd.DataFrame([
            {
                "control_type": "length_control",
                "mean_delta": 0.5,
                "ci_95_lower": 0.2,
                "ci_95_upper": 0.8,
                "mean_abs_delta": 0.5,
            },
            {
                "control_type": "format_control",
                "mean_delta": -0.1,
                "ci_95_lower": -0.3,
                "ci_95_upper": 0.1,
                "mean_abs_delta": 0.2,
            },
        ])
        audit_results = {
            "dataset_audit": {"df_raw": df_raw},
            "controls_summary": df_controls,
            "controls_raw": {
                "paired_records": [
                    {"control_type": "length_control", "delta": 0.5},
                    {"control_type": "format_control", "delta": -0.1},
                ],
                "prompts": [
                    {"messages": [{"content": "hello"}], "projection_score": 1.0},
                    {"messages": [{"content": "world test"}], "projection_score": -0.5},
                ]
            },
            "cross_direction": {
                "cos_orig_expanded": 0.85,
                "mean_cos_orig_rand": 0.01,
                "mean_cos_exp_rand": -0.02,
            },
            "expanded_stability": {
                "mean_cosine_similarity": 0.88,
                "std_cosine_similarity": 0.05,
                "ci_95_lower": 0.78,
                "ci_95_upper": 0.95,
                "similarities": [0.88] * 10,
            },
            "n_expanded_prompts": 100,
        }

        fig_dir = tmp_path / "figures"
        figs = generate_confound_audit_figures(audit_results, fig_dir)
        assert len(figs) == 9
        for f in figs:
            assert f.exists()

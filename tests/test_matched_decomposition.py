# =============================================================================
# Unit Tests — Matched-Pair Decomposition Engine (Phase 5.75)
# =============================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.activations.matched_decomposition import (
    audit_match_quality,
    compute_effect_ratios,
    compute_pair_level_deltas,
    compute_token_length_regressions,
    generate_matched_decomposition_figures,
)


class TestMatchQualityAudit:
    def test_audit_match_quality(self) -> None:
        prompts = [
            {
                "pair_id": "pair_01_persona",
                "condition": "persona",
                "role_in_pair": "base",
                "messages": [{"role": "system", "content": "hello world assistant"}],
            },
            {
                "pair_id": "pair_01_persona",
                "condition": "persona",
                "role_in_pair": "manipulated",
                "messages": [{"role": "system", "content": "hello world alternative"}],
            },
        ]
        df_match = audit_match_quality(prompts)
        assert isinstance(df_match, pd.DataFrame)
        assert "persona" in df_match["condition"].values
        assert len(df_match) == 1
        assert df_match.loc[0, "n_pairs"] == 1


class TestComputePairLevelDeltas:
    def test_deltas_and_condition_statistics(self) -> None:
        dim = 8
        direction = np.ones(dim) / np.sqrt(dim)
        prompts = [
            {
                "pair_id": "p1_len",
                "condition": "length",
                "role_in_pair": "base",
                "messages": [{"role": "user", "content": "short"}],
            },
            {
                "pair_id": "p1_len",
                "condition": "length",
                "role_in_pair": "manipulated",
                "messages": [{"role": "user", "content": "longer prompt text"}],
            },
            {
                "pair_id": "p2_len",
                "condition": "length",
                "role_in_pair": "base",
                "messages": [{"role": "user", "content": "short2"}],
            },
            {
                "pair_id": "p2_len",
                "condition": "length",
                "role_in_pair": "manipulated",
                "messages": [{"role": "user", "content": "longer prompt text 2"}],
            },
        ]
        acts = np.zeros((4, dim))
        acts[0] = 1.0  # base p1
        acts[1] = 2.0  # manip p1 -> delta = +1.0 * sqrt(8)
        acts[2] = 1.0  # base p2
        acts[3] = 3.0  # manip p2 -> delta = +2.0 * sqrt(8)

        df_pairs, df_cond = compute_pair_level_deltas(prompts, acts, direction, seed=42, n_bootstrap=50)

        assert len(df_pairs) == 2
        assert len(df_cond) == 1
        assert df_cond.loc[0, "condition"] == "length"
        assert df_cond.loc[0, "mean_delta"] > 0
        assert df_cond.loc[0, "pct_positive"] == 100.0


class TestEffectRatiosAndRegressions:
    def test_ratios_and_regressions(self) -> None:
        df_cond = pd.DataFrame([
            {"condition": "persona", "mean_abs_delta": 2.0},
            {"condition": "length", "mean_abs_delta": 1.0},
            {"condition": "format", "mean_abs_delta": 0.5},
            {"condition": "lexical", "mean_abs_delta": 0.8},
            {"condition": "context", "mean_abs_delta": 0.4},
        ])
        ratios = compute_effect_ratios(df_cond)
        assert np.isclose(ratios["persona_to_length_ratio"], 2.0)
        assert np.isclose(ratios["persona_to_format_ratio"], 4.0)

        df_pairs = pd.DataFrame({
            "condition": ["persona", "persona", "length", "length"],
            "delta": [1.0, 2.0, -0.5, -1.0],
            "delta_tokens": [0, 1, 10, 20],
            "delta_chars": [0, 5, 50, 100],
        })
        df_reg = compute_token_length_regressions(df_pairs)
        assert isinstance(df_reg, pd.DataFrame)
        assert "delta_tokens_all" in df_reg["predictor"].values


class TestFigureGeneration:
    def test_generate_8_figures(self, tmp_path: Path) -> None:
        df_pairs = pd.DataFrame({
            "condition": ["persona", "length", "format", "lexical", "context"],
            "score_base": [1.0, 1.0, 1.0, 1.0, 1.0],
            "score_manipulated": [2.0, 0.5, 0.8, 0.9, 0.7],
            "delta": [1.0, -0.5, -0.2, -0.1, -0.3],
            "delta_tokens": [0, 15, 5, 2, 8],
            "delta_chars": [0, 60, 20, 10, 40],
        })
        df_cond = pd.DataFrame({
            "condition": ["persona", "length", "format", "lexical", "context"],
            "mean_delta": [1.0, -0.5, -0.2, -0.1, -0.3],
            "ci_95_lower": [0.8, -0.7, -0.4, -0.2, -0.5],
            "ci_95_upper": [1.2, -0.3, -0.1, 0.0, -0.1],
        })
        df_match = pd.DataFrame({
            "condition": ["persona", "length", "format", "lexical", "context"],
            "mean_abs_delta_chars": [5.0, 50.0, 20.0, 10.0, 30.0],
            "mean_abs_delta_tokens": [1.0, 12.0, 4.0, 2.0, 7.0],
        })
        ratios = {
            "persona_to_length_ratio": 2.0,
            "persona_to_format_ratio": 5.0,
            "persona_to_lexical_ratio": 10.0,
            "persona_to_context_ratio": 3.3,
        }

        fig_dir = tmp_path / "figures"
        figs = generate_matched_decomposition_figures(df_pairs, df_cond, df_match, ratios, fig_dir)
        assert len(figs) == 8
        for f in figs:
            assert f.exists()

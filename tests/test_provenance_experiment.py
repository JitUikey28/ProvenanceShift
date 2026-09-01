"""
Offline unit tests for Phase 6 controlled provenance experiments.

Verifies:
    - Matched task grouping and paired delta computations
    - Paired Wilcoxon signed-rank and Student's t-tests
    - Paired unit bootstrap confidence intervals
    - Multiple comparisons corrections (Bonferroni, Holm, Benjamini-Hochberg)
    - Representation-to-behavior association (Pearson, Spearman)
    - End-to-end synthetic provenance analysis pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.behavioral import BehavioralMetrics
from src.experiments.provenance import (
    apply_multiple_comparisons_correction,
    compute_paired_deltas,
    compute_paired_statistics,
    compute_representation_behavior_association,
    group_matched_tasks,
    run_provenance_analysis,
)


# =====================================================================
# Grouping & Paired Deltas
# =====================================================================

class TestMatchedGroupingAndDeltas:

    def test_group_and_deltas(self) -> None:
        prompt_items = [
            {"task_id": "t1", "condition": "baseline"},
            {"task_id": "t1", "condition": "provenance_manipulation"},
            {"task_id": "t1", "condition": "surface_control"},
            {"task_id": "t2", "condition": "baseline"},
            {"task_id": "t2", "condition": "provenance_manipulation"},
            {"task_id": "t2", "condition": "surface_control"},
        ]
        scores = [1.0, 3.0, 1.2, 2.0, 5.0, 2.1]
        behaviors = [
            BehavioralMetrics(formality_score=0.5),
            BehavioralMetrics(formality_score=0.9),
            BehavioralMetrics(formality_score=0.55),
            BehavioralMetrics(formality_score=0.4),
            BehavioralMetrics(formality_score=0.8),
            BehavioralMetrics(formality_score=0.45),
        ]

        grouped = group_matched_tasks(prompt_items, scores, behaviors)
        assert len(grouped) == 2
        assert "t1" in grouped and "t2" in grouped

        df_deltas = compute_paired_deltas(grouped)
        assert len(df_deltas) == 2
        assert "delta_score_provenance_manipulation" in df_deltas.columns

        # t1 delta = 3.0 - 1.0 = 2.0
        assert df_deltas.loc[df_deltas["task_id"] == "t1", "delta_score_provenance_manipulation"].values[0] == 2.0
        # t2 delta = 5.0 - 2.0 = 3.0
        assert df_deltas.loc[df_deltas["task_id"] == "t2", "delta_score_provenance_manipulation"].values[0] == 3.0
        # Formality delta for t1 = 0.9 - 0.5 = 0.4
        assert np.isclose(df_deltas.loc[df_deltas["task_id"] == "t1", "delta_formality_provenance_manipulation"].values[0], 0.4)


# =====================================================================
# Paired Statistics & Bootstrap
# =====================================================================

class TestPairedStatistics:

    def test_paired_statistics_positive_shift(self) -> None:
        deltas = np.array([1.5, 2.0, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3])
        res = compute_paired_statistics(deltas, test_name="test_shift", seed=42)

        assert res["n_pairs"] == 8
        assert np.isclose(res["mean_delta"], 1.9375)
        assert res["cohens_dz"] > 5.0  # Very large effect size
        assert res["t_pvalue"] < 0.001
        assert res["wilcoxon_pvalue"] < 0.01
        assert res["ci_lower"] > 1.5
        assert res["ci_upper"] < 2.3


# =====================================================================
# Multiple Comparisons Corrections
# =====================================================================

class TestMultipleComparisons:

    def test_bonferroni(self) -> None:
        p_vals = [0.01, 0.04, 0.20]
        adj_p, reject = apply_multiple_comparisons_correction(p_vals, method="bonferroni")
        assert np.isclose(adj_p[0], 0.03)  # 0.01 * 3
        assert np.isclose(adj_p[1], 0.12)  # 0.04 * 3
        assert np.isclose(adj_p[2], 0.60)  # 0.20 * 3
        assert reject[0] == True
        assert reject[1] == False

    def test_fdr_bh(self) -> None:
        p_vals = [0.001, 0.02, 0.50]
        adj_p, reject = apply_multiple_comparisons_correction(p_vals, method="fdr_bh")
        assert adj_p[0] <= 0.01
        assert adj_p[0] <= adj_p[1] <= adj_p[2]


# =====================================================================
# Representation vs Behavior Association
# =====================================================================

class TestRepresentationBehaviorAssociation:

    def test_positive_correlation(self) -> None:
        dr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        db = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        assoc = compute_representation_behavior_association(dr, db)
        assert np.isclose(assoc["pearson_r"], 1.0)
        assert np.isclose(assoc["spearman_rho"], 1.0)
        assert assoc["pearson_p"] < 0.01


# =====================================================================
# Synthetic End-to-End Provenance Pipeline
# =====================================================================

class TestSyntheticProvenancePipeline:

    def test_end_to_end_analysis(self, tmp_path: Path) -> None:
        exp_id = "test-prov-pilot"

        # Generate synthetic paired deltas table
        delta_df = pd.DataFrame({
            "task_id": [f"t_{i}" for i in range(10)],
            "baseline_score": np.random.randn(10),
            "delta_score_provenance_manipulation": np.array([1.5, 1.8, 2.0, 1.2, 1.7, 1.9, 2.1, 1.4, 1.6, 1.8]),
            "delta_score_surface_control": np.random.randn(10) * 0.1,
            "delta_score_neutral_control": np.random.randn(10) * 0.05,
            "delta_formality_provenance_manipulation": np.array([0.3, 0.4, 0.5, 0.2, 0.35, 0.45, 0.5, 0.3, 0.35, 0.4]),
        })

        summary_df, meta, artifacts = run_provenance_analysis(
            delta_df=delta_df,
            experiment_id=exp_id,
            output_tables_dir=tmp_path / "tables",
            output_figures_dir=tmp_path / "figures",
            output_raw_dir=tmp_path / "raw",
            seed=42,
        )

        assert len(summary_df) == 3
        assert artifacts["summary_table"].exists()
        assert artifacts["paired_deltas_table"].exists()
        assert artifacts["fig_deltas"].exists()
        assert artifacts["fig_association"].exists()
        assert artifacts["metadata"].exists()

        # Check that provenance manipulation has large effect size
        prov_row = summary_df.loc[summary_df["test_name"] == "provenance_manipulation_vs_baseline"]
        assert prov_row["mean_delta"].values[0] > 1.0
        assert prov_row["wilcoxon_pvalue"].values[0] < 0.05

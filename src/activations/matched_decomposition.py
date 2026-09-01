# =============================================================================
# Matched-Pair Decomposition Engine — ProvenanceShift Phase 5.75
# =============================================================================
"""Engine for matched-pair decomposition experiments isolating persona-associated
representation shifts from superficial prompt changes (length, formatting,
lexical rewording, and neutral context).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.activations.direction import compute_cohens_d, project_representations
from src.utils.logging import get_logger

logger = get_logger("matched_decomposition")


# ---------------------------------------------------------------------------
# 1. Match Quality Audit
# ---------------------------------------------------------------------------

def audit_match_quality(prompts: List[Dict[str, Any]]) -> pd.DataFrame:
    """Audit token and character balance across paired conditions.

    Parameters
    ----------
    prompts:
        List of prompt dictionaries.

    Returns
    -------
    pd.DataFrame
        Match quality summary statistics per condition.
    """
    pairs: Dict[str, Dict[str, Any]] = {}
    for p in prompts:
        pid = p["pair_id"]
        role = p.get("role_in_pair", "base")
        pairs.setdefault(pid, {})[role] = p

    rows = []
    for pid, pdict in pairs.items():
        base = pdict.get("base")
        manip = pdict.get("manipulated")
        if not base or not manip:
            continue

        cond = base.get("condition", "unknown")
        task_id = base.get("task_id", "unknown")
        domain = base.get("domain", "unknown")

        base_text = " ".join(m.get("content", "") for m in base.get("messages", []))
        manip_text = " ".join(m.get("content", "") for m in manip.get("messages", []))

        base_words = len(base_text.split())
        manip_words = len(manip_text.split())
        base_chars = len(base_text)
        manip_chars = len(manip_text)

        # Estimate tokens as approximate word count if token_metadata not present
        base_tokens = base.get("token_metadata", {}).get("input_token_count", base_words)
        manip_tokens = manip.get("token_metadata", {}).get("input_token_count", manip_words)

        rows.append({
            "pair_id": pid,
            "condition": cond,
            "task_id": task_id,
            "domain": domain,
            "base_chars": base_chars,
            "manip_chars": manip_chars,
            "delta_chars": manip_chars - base_chars,
            "abs_delta_chars": abs(manip_chars - base_chars),
            "base_tokens": base_tokens,
            "manip_tokens": manip_tokens,
            "delta_tokens": manip_tokens - base_tokens,
            "abs_delta_tokens": abs(manip_tokens - base_tokens),
            "base_n_msgs": len(base.get("messages", [])),
            "manip_n_msgs": len(manip.get("messages", [])),
        })

    df_pairs = pd.DataFrame(rows)

    summary_rows = []
    for cond in sorted(df_pairs["condition"].unique()):
        sub = df_pairs[df_pairs["condition"] == cond]
        d_chars = compute_cohens_d(sub["manip_chars"].to_numpy(), sub["base_chars"].to_numpy())
        d_tokens = compute_cohens_d(sub["manip_tokens"].to_numpy(), sub["base_tokens"].to_numpy())

        summary_rows.append({
            "condition": cond,
            "n_pairs": len(sub),
            "mean_base_chars": float(sub["base_chars"].mean()),
            "mean_manip_chars": float(sub["manip_chars"].mean()),
            "mean_delta_chars": float(sub["delta_chars"].mean()),
            "mean_abs_delta_chars": float(sub["abs_delta_chars"].mean()),
            "cohens_d_chars": float(d_chars),
            "mean_base_tokens": float(sub["base_tokens"].mean()),
            "mean_manip_tokens": float(sub["manip_tokens"].mean()),
            "mean_delta_tokens": float(sub["delta_tokens"].mean()),
            "mean_abs_delta_tokens": float(sub["abs_delta_tokens"].mean()),
            "cohens_d_tokens": float(d_tokens),
        })

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# 2. Paired Projection Delta & Condition Statistics
# ---------------------------------------------------------------------------

def compute_pair_level_deltas(
    prompts: List[Dict[str, Any]],
    activations: np.ndarray,
    direction: np.ndarray,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pair-level delta scores and condition summary statistics.

    Parameters
    ----------
    prompts:
        List of prompt dictionaries.
    activations:
        2D matrix of shape ``(n_prompts, hidden_dim)`` for target layer.
    direction:
        1D unit direction vector from Layer 2.
    seed:
        Random seed for bootstrap resampling.
    n_bootstrap:
        Number of bootstrap iterations.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (pair_level_df, condition_summary_df)
    """
    scores = project_representations(activations, direction)
    for i, p in enumerate(prompts):
        p["projection_score"] = float(scores[i])

    pairs: Dict[str, Dict[str, Any]] = {}
    for p in prompts:
        pid = p["pair_id"]
        role = p.get("role_in_pair", "base")
        pairs.setdefault(pid, {})[role] = p

    pair_rows = []
    for pid, pdict in pairs.items():
        base = pdict.get("base")
        manip = pdict.get("manipulated")
        if not base or not manip:
            continue

        cond = base.get("condition", "unknown")
        task_id = base.get("task_id", "unknown")
        domain = base.get("domain", "unknown")

        s_base = base["projection_score"]
        s_manip = manip["projection_score"]
        delta = s_manip - s_base
        abs_delta = abs(delta)

        base_text = " ".join(m.get("content", "") for m in base.get("messages", []))
        manip_text = " ".join(m.get("content", "") for m in manip.get("messages", []))
        base_words = len(base_text.split())
        manip_words = len(manip_text.split())
        base_chars = len(base_text)
        manip_chars = len(manip_text)

        pair_rows.append({
            "pair_id": pid,
            "condition": cond,
            "task_id": task_id,
            "domain": domain,
            "score_base": s_base,
            "score_manipulated": s_manip,
            "delta": delta,
            "abs_delta": abs_delta,
            "direction_is_positive": bool(delta > 0),
            "base_chars": base_chars,
            "manip_chars": manip_chars,
            "delta_chars": manip_chars - base_chars,
            "base_tokens": base_words,
            "manip_tokens": manip_words,
            "delta_tokens": manip_words - base_words,
        })

    df_pairs = pd.DataFrame(pair_rows)

    rng = np.random.default_rng(seed)
    cond_rows = []

    for cond in ["persona", "length", "format", "lexical", "context"]:
        sub = df_pairs[df_pairs["condition"] == cond]
        if sub.empty:
            continue

        deltas = sub["delta"].to_numpy()
        abs_deltas = sub["abs_delta"].to_numpy()
        n = len(deltas)

        mean_d = float(np.mean(deltas))
        median_d = float(np.median(deltas))
        std_d = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
        mean_abs_d = float(np.mean(abs_deltas))
        median_abs_d = float(np.median(abs_deltas))

        # Cohens dz = mean(delta) / std(delta)
        dz = float(mean_d / std_d) if std_d > 1e-12 else 0.0

        pct_pos = float(np.mean(deltas > 0) * 100.0)
        pct_neg = float(np.mean(deltas < 0) * 100.0)

        # Bootstrap 95% CI on mean delta
        boot_means = []
        for _ in range(n_bootstrap):
            b_idx = rng.choice(n, size=n, replace=True)
            boot_means.append(np.mean(deltas[b_idx]))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        # Significance tests against zero delta
        if n >= 2 and not np.all(deltas == deltas[0]):
            try:
                _, p_ttest = stats.ttest_1samp(deltas, 0.0)
            except Exception:
                p_ttest = 1.0
            try:
                _, p_wilcoxon = stats.wilcoxon(deltas)
            except Exception:
                p_wilcoxon = 1.0
        else:
            p_ttest = 1.0
            p_wilcoxon = 1.0

        cond_rows.append({
            "condition": cond,
            "N": n,
            "mean_delta": mean_d,
            "median_delta": median_d,
            "std_delta": std_d,
            "mean_abs_delta": mean_abs_d,
            "median_abs_delta": median_abs_d,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "cohens_dz": dz,
            "pct_positive": pct_pos,
            "pct_negative": pct_neg,
            "p_val_wilcoxon": float(p_wilcoxon),
            "p_val_ttest": float(p_ttest),
        })

    df_cond = pd.DataFrame(cond_rows)
    return df_pairs, df_cond


# ---------------------------------------------------------------------------
# 3. Persona-to-Surface Effect Ratios
# ---------------------------------------------------------------------------

def compute_effect_ratios(df_cond: pd.DataFrame) -> Dict[str, float]:
    """Compute persona effect magnitude relative to surface control magnitudes."""
    cond_map = {row["condition"]: row["mean_abs_delta"] for _, row in df_cond.iterrows()}
    persona_mag = cond_map.get("persona", 0.0)

    ratios = {}
    for ctrl in ["length", "format", "lexical", "context"]:
        ctrl_mag = cond_map.get(ctrl, 0.0)
        ratios[f"persona_to_{ctrl}_ratio"] = float(persona_mag / ctrl_mag) if ctrl_mag > 1e-12 else 0.0
    return ratios


# ---------------------------------------------------------------------------
# 4. Token & Character Count Regressions
# ---------------------------------------------------------------------------

def compute_token_length_regressions(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Run linear regressions of projection delta vs token and character deltas."""
    results = []

    # 1. Overall across all pairs
    for x_col, label in [("delta_tokens", "delta_tokens_all"), ("delta_chars", "delta_chars_all")]:
        x = df_pairs[x_col].to_numpy(dtype=float)
        y = df_pairs["delta"].to_numpy(dtype=float)
        r, p_pearson = stats.pearsonr(x, y)
        rho, p_spearman = stats.spearmanr(x, y)
        slope, intercept, _, _, stderr = stats.linregress(x, y)

        results.append({
            "subset": "all_conditions",
            "predictor": label,
            "pearson_r": float(r),
            "pearson_p": float(p_pearson),
            "spearman_rho": float(rho),
            "spearman_p": float(p_spearman),
            "slope_beta": float(slope),
            "intercept_alpha": float(intercept),
            "slope_stderr": float(stderr),
            "r_squared": float(r ** 2),
        })

    # 2. Per-condition regressions
    for cond in sorted(df_pairs["condition"].unique()):
        sub = df_pairs[df_pairs["condition"] == cond]
        for x_col, label in [("delta_tokens", f"delta_tokens_{cond}"), ("delta_chars", f"delta_chars_{cond}")]:
            x = sub[x_col].to_numpy(dtype=float)
            y = sub["delta"].to_numpy(dtype=float)
            if np.all(x == x[0]) or len(x) < 3:
                r, p_pearson, rho, p_spearman, slope, intercept, stderr = 0.0, 1.0, 0.0, 1.0, 0.0, float(np.mean(y)), 0.0
            else:
                r, p_pearson = stats.pearsonr(x, y)
                rho, p_spearman = stats.spearmanr(x, y)
                slope, intercept, _, _, stderr = stats.linregress(x, y)

            results.append({
                "subset": cond,
                "predictor": label,
                "pearson_r": float(r),
                "pearson_p": float(p_pearson),
                "spearman_rho": float(rho),
                "spearman_p": float(p_spearman),
                "slope_beta": float(slope),
                "intercept_alpha": float(intercept),
                "slope_stderr": float(stderr),
                "r_squared": float(r ** 2),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Visualizations (8 Publication-Grade Plots)
# ---------------------------------------------------------------------------

def generate_matched_decomposition_figures(
    df_pairs: pd.DataFrame,
    df_cond: pd.DataFrame,
    df_match: pd.DataFrame,
    ratios: Dict[str, float],
    output_dir: Path,
) -> List[Path]:
    """Generate the 8 required figures without hard-coded styles or color overrides."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = []

    # 1. Delta projection distributions by condition
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cond_order = ["persona", "length", "format", "lexical", "context"]
    plot_data = [df_pairs[df_pairs["condition"] == c]["delta"].tolist() for c in cond_order if c in df_pairs["condition"].values]
    labels = [c.capitalize() for c in cond_order if c in df_pairs["condition"].values]
    try:
        ax.boxplot(plot_data, tick_labels=labels, patch_artist=True)
    except TypeError:
        ax.boxplot(plot_data, labels=labels, patch_artist=True)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("Paired Delta (Manipulated - Base)")
    ax.set_title("Projection Shift Distributions Across Matched Conditions")
    p1 = output_dir / "1_delta_projection_distributions.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    fig_paths.append(p1)

    # 2. Paired base -> manipulated projection lines for persona
    fig, ax = plt.subplots(figsize=(6, 5))
    sub_p = df_pairs[df_pairs["condition"] == "persona"]
    for _, row in sub_p.iterrows():
        ax.plot([0, 1], [row["score_base"], row["score_manipulated"]], marker="o", alpha=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Base Assistant", "Manipulated Alternative"])
    ax.set_ylabel("Layer-2 Projection Score")
    ax.set_title("Paired Trajectories: Length-Matched Persona Manipulation")
    p2 = output_dir / "2_paired_trajectories_persona.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    fig_paths.append(p2)

    # 3. Paired base -> manipulated projection lines for surface controls
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    ctrls = ["length", "format", "lexical", "context"]
    for ax, ctrl in zip(axes, ctrls):
        sub_c = df_pairs[df_pairs["condition"] == ctrl]
        for _, row in sub_c.iterrows():
            ax.plot([0, 1], [row["score_base"], row["score_manipulated"]], marker="o", alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Base", "Manipulated"])
        ax.set_title(f"{ctrl.capitalize()} Control")
        if ax is axes[0]:
            ax.set_ylabel("Projection Score")
    p3 = output_dir / "3_paired_trajectories_surface_controls.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    fig_paths.append(p3)

    # 4. Mean effect comparison with 95% bootstrap confidence intervals
    fig, ax = plt.subplots(figsize=(8, 4.5))
    conds = df_cond["condition"].tolist()
    means = df_cond["mean_delta"].tolist()
    errs = [
        [m - l for m, l in zip(means, df_cond["ci_95_lower"])],
        [u - m for m, u in zip(means, df_cond["ci_95_upper"])],
    ]
    ax.bar(range(len(conds)), means, yerr=errs, capsize=5, edgecolor="black", alpha=0.7)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.capitalize() for c in conds])
    ax.set_ylabel("Mean Signed Delta (95% CI)")
    ax.set_title("Mean Projection Shift Across Conditions (N=30 pairs/condition)")
    p4 = output_dir / "4_mean_effects_ci95.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    fig_paths.append(p4)

    # 5. Delta projection vs Delta token count
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df_pairs["delta_tokens"], df_pairs["delta"], alpha=0.7, edgecolors="black")
    slope, intercept, r_val, _, _ = stats.linregress(df_pairs["delta_tokens"], df_pairs["delta"])
    x_vals = np.linspace(df_pairs["delta_tokens"].min(), df_pairs["delta_tokens"].max(), 50)
    ax.plot(x_vals, intercept + slope * x_vals, linestyle="--", label=f"Fit (r={r_val:.2f}, slope={slope:.3f})")
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.set_xlabel("Delta Word/Token Count (Manipulated - Base)")
    ax.set_ylabel("Delta Projection Score")
    ax.set_title("Projection Shift vs Token Count Perturbation")
    ax.legend(frameon=True)
    p5 = output_dir / "5_delta_proj_vs_delta_tokens.png"
    fig.tight_layout()
    fig.savefig(p5, dpi=300)
    plt.close(fig)
    fig_paths.append(p5)

    # 6. Delta projection vs Delta character count
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df_pairs["delta_chars"], df_pairs["delta"], alpha=0.7, edgecolors="black")
    slope_c, intercept_c, r_val_c, _, _ = stats.linregress(df_pairs["delta_chars"], df_pairs["delta"])
    x_c = np.linspace(df_pairs["delta_chars"].min(), df_pairs["delta_chars"].max(), 50)
    ax.plot(x_c, intercept_c + slope_c * x_c, linestyle="--", label=f"Fit (r={r_val_c:.2f}, slope={slope_c:.3f})")
    ax.axhline(0, linestyle=":", linewidth=1)
    ax.axvline(0, linestyle=":", linewidth=1)
    ax.set_xlabel("Delta Character Count (Manipulated - Base)")
    ax.set_ylabel("Delta Projection Score")
    ax.set_title("Projection Shift vs Character Count Perturbation")
    ax.legend(frameon=True)
    p6 = output_dir / "6_delta_proj_vs_delta_chars.png"
    fig.tight_layout()
    fig.savefig(p6, dpi=300)
    plt.close(fig)
    fig_paths.append(p6)

    # 7. Distribution of persona/surface effect ratios
    fig, ax = plt.subplots(figsize=(7, 4))
    r_labels = [k.replace("persona_to_", "").replace("_ratio", "").capitalize() for k in ratios.keys()]
    r_vals = list(ratios.values())
    ax.bar(r_labels, r_vals, edgecolor="black", alpha=0.7)
    ax.axhline(1.0, linestyle="--", linewidth=1.5, label="Parity (|Persona| = |Surface|)")
    ax.set_ylabel("Ratio: |Delta Persona| / |Delta Control|")
    ax.set_title("Persona vs Surface Control Effect-Size Ratios")
    ax.legend(frameon=True)
    p7 = output_dir / "7_effect_ratios_distribution.png"
    fig.tight_layout()
    fig.savefig(p7, dpi=300)
    plt.close(fig)
    fig_paths.append(p7)

    # 8. Match-quality distributions (Token & Character count deltas)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(len(df_match)), df_match["mean_abs_delta_chars"], edgecolor="black", alpha=0.7)
    axes[0].set_xticks(range(len(df_match)))
    axes[0].set_xticklabels([c.capitalize() for c in df_match["condition"]])
    axes[0].set_ylabel("Mean |Delta Characters|")
    axes[0].set_title("Character Imbalance by Condition")

    axes[1].bar(range(len(df_match)), df_match["mean_abs_delta_tokens"], edgecolor="black", alpha=0.7)
    axes[1].set_xticks(range(len(df_match)))
    axes[1].set_xticklabels([c.capitalize() for c in df_match["condition"]])
    axes[1].set_ylabel("Mean |Delta Tokens|")
    axes[1].set_title("Token Imbalance by Condition")

    p8 = output_dir / "8_match_quality_distributions.png"
    fig.tight_layout()
    fig.savefig(p8, dpi=300)
    plt.close(fig)
    fig_paths.append(p8)

    return fig_paths

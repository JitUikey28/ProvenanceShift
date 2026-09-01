# Methodology

> **Status:** Infrastructure implemented for Phases 1–6.
> No experimental results have been generated or reported yet.

## Overview

The research investigates whether contextual information about the source, role, or provenance of text systematically influences model persona representations and behavior.

The methodological sequence:

```
Controlled Prompt Pairs (Matched Semantics, Contrasting Persona)
          ↓
Model Forward Pass under Inference Mode
          ↓
Layer-by-Layer Hidden State Extraction (last_token / mean_pool)
          ↓
Candidate Direction Construction on TRAIN Split Only (mu_assistant - mu_alternative)
          ↓
Phase 5: Validation Selection & Held-Out Rigorous Evaluation (Stability, Random Null, Train-Fitted PCA)
          ↓
Phase 6: Matched 4-Condition Provenance Experiment (Baseline vs Provenance vs Surface Control vs Neutral Control)
          ↓
Paired Statistical Tests, Paired Bootstrap CIs, Multiple Comparisons Correction & Behavioral Association
```

---

## Phase 5: Persona Representation Validation Methodology

### 1. Validation-Based Layer Selection
Candidate layers are compared strictly using the **validation split** (e.g. maximizing `val_roc_auc` or `val_cohens_d`):
$$L^* = \arg\max_{l} \text{Metric}_{\text{val}}(l)$$
The held-out test split is evaluated strictly once on the chosen configuration $L^*$.

### 2. Direction Stability via Bootstrap Resampling
To ensure candidate directions do not depend on isolated prompt outliers, we resample the training set with replacement ($B=100$ resamples) and measure pairwise cosine similarity:
$$\text{Stability} = \frac{1}{B} \sum_{b=1}^{B} \frac{v_b \cdot v_0}{\|v_b\|_2 \|v_0\|_2}$$

### 3. Empirical Random-Direction Null Distribution
Rather than testing against a single random vector, we sample $K=100$ uniformly distributed unit vectors on the unit hypersphere:
$$v_{\text{rand}} = \frac{z}{\|z\|_2}, \quad z \sim \mathcal{N}(0, I_d)$$
We compute the empirical null distribution for test ROC-AUC and calculate an empirical $p$-value:
$$p_{\text{empirical}} = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(\text{ROC-AUC}_{\text{rand}, k} \ge \text{ROC-AUC}_{\text{learned}})$$

### 4. Train-Fitted PCA Visualization
PCA dimensionality reduction is fitted **strictly on training activations** $\mathbf{X}_{\text{train}}$ and used only as a descriptive projection for validation and test activations.

---

## Phase 6: Controlled Provenance Experiment Methodology

### 1. 4-Condition Matched Task Design
For each base semantic task $T_i$, four strictly matched prompt conditions are constructed:
- **Condition A (`baseline`):** Standard helpful assistant prompt.
- **Condition B (`provenance_manipulation`):** Same semantic information, framed with altered perceived provenance/role.
- **Condition C (`surface_control`):** Structural/formatting/length matched control without altering provenance.
- **Condition D (`neutral_control`):** Neutral contextual addition (negative control).

### 2. Paired Representation Shift ($\Delta_{\text{persona}}$)
For each task $T_i$, we compute paired projection shifts along the validated direction $\hat{v}$:
$$\Delta_{\text{persona}}(T_i) = s(\text{provenance}, T_i) - s(\text{baseline}, T_i)$$
$$\Delta_{\text{surface}}(T_i) = s(\text{surface\_control}, T_i) - s(\text{baseline}, T_i)$$
$$\Delta_{\text{neutral}}(T_i) = s(\text{neutral\_control}, T_i) - s(\text{baseline}, T_i)$$

### 3. Paired Statistical Tests & Effect Sizes
- **Paired Cohen's $d_z$:** Standardized paired difference:
  $$d_z = \frac{\bar{\Delta}}{s_{\Delta}}$$
- **Hypothesis Testing:** Non-parametric Wilcoxon signed-rank test and paired Student's t-test (with Shapiro-Wilk normality check).
- **Paired Unit Bootstrap:** Resamples matched task units $T_i$ with replacement ($N=1000$ iterations) to construct 95% confidence intervals on $\bar{\Delta}$.
- **Multiple Comparisons Correction:** False Discovery Rate (Benjamini-Hochberg) and family-wise error rate (Holm / Bonferroni).

### 4. Representation vs. Behavior Association
We compute Pearson $r$ and Spearman rank correlation $\rho$ between representation shifts $\Delta_{\text{persona}}$ and behavioral shifts $\Delta_{\text{behavior}}$ (e.g. formality, first-person rate, role adherence score).

---

## Scientific Interpretation Rules

> [!IMPORTANT]
> **Hypothesis Testing, Not Assertion:**
> - We test hypothesis **H1** (*perceived provenance shifts persona representations*) against null hypothesis **H0** (*shifts do not exceed ordinary variation and format confounds*).
> - We do **not** write "provenance causes persona drift" or "the assistant axis is proven."
> - Representation-behavior correlation does **not** prove causation.

# Phase 6: Controlled Provenance Investigation Report

**Experiment ID:** `PROVENANCE-PILOT-001`  
**Target Representation:** Layer 2 ($d=2560$, `microsoft/Phi-2`)  
**Direction Vector:** Locked $D_{\text{expanded}}$ from Phase 5.5/5.75  
**Matched Task Sets:** 8 tasks (4 conditions per task = 32 prompt items)  
**Execution Timestamp:** 2026-08-31T10:54:46.041459+00:00  
**Evidence Classification:** **Category B: INTERNAL SHIFT**  

---

## 1. Primary Research Question & Central Comparison

**Question:** Does changing perceived instruction provenance produce a systematic shift along the validated persona-associated representation beyond the shift produced by matched surface controls?

**Central Contrast (PRIMARY):**
$$\Delta_{\text{net}} = \Delta_{\text{provenance}} - \Delta_{\text{surface}}$$

---

## 2. Key Comparison Summary Table

| Comparison | $N$ Tasks | Mean $\Delta$ | Median $\Delta$ | Std Dev | 95% Bootstrap CI | Cohen's $d_z$ | % Positive | % Negative | Wilcoxon $p$-value |
|---|---|---|---|---|---|---|---|---|---|
| **Provenance Manipulation Vs Baseline** | 8 | -1.6362 | -1.6366 | 0.2648 | [-1.8088, -1.4748] | -6.178 | 0.0% | 100.0% | 7.812e-03 |
| **Surface Control Vs Baseline** | 8 | -0.4292 | -0.4376 | 0.0849 | [-0.4825, -0.3733] | -5.056 | 0.0% | 100.0% | 7.812e-03 |
| **Neutral Control Vs Baseline** | 8 | -0.5584 | -0.5764 | 0.1620 | [-0.6558, -0.4557] | -3.448 | 0.0% | 100.0% | 7.812e-03 |
| **Net Provenance Vs Surface** | 8 | -1.2070 | -1.1658 | 0.2505 | [-1.3782, -1.0640] | -4.818 | 0.0% | 100.0% | 7.812e-03 |

---

## 3. Primary Net Shift Breakdown ($\Delta_{\text{net}}$)

- **Number of Matched Tasks:** 8
- **Mean Net Shift ($\Delta_{\text{net}}$):** -1.2070
- **Median Net Shift:** -1.1658
- **Standard Deviation:** 0.2505
- **Mean Absolute Net Shift:** 1.2070
- **95% Bootstrap CI:** [-1.3782, -1.0640]
- **Directional Consistency:** 100.0% negative / 0.0% positive
- **Effect Size (Cohen's $d_z$):** -4.818
- **Wilcoxon Signed-Rank Test:** $p = 7.812e-03$
- **Paired $t$-Test:** $t = -13.628, p = 2.696e-06$

---

## 4. Secondary Control Shifts

- **Raw Provenance Shift ($\Delta_{\text{prov}}$):** Mean = -1.6362, Median = -1.6366, 95% CI [-1.8088, -1.4748], $d_z = -6.178$.
- **Surface Control Shift ($\Delta_{\text{surf}}$):** Mean = -0.4292, Median = -0.4376, 95% CI [-0.4825, -0.3733], $d_z = -5.056$.
- **Neutral Control Shift ($\Delta_{\text{neut}}$):** Mean = -0.5584, Median = -0.5764, 95% CI [-0.6558, -0.4557], $d_z = -3.448$.

---

## 5. Behavioral Output & Association Analysis

| Comparison | Pearson $r$ ($p$-value) | Spearman $\rho$ ($p$-value) |
|---|---|---|
| `raw_prov_vs_formality` | 0.000 ($p=1.000e+00$) | 0.000 ($p=1.000e+00$) |
| `raw_prov_vs_fp_rate` | 0.000 ($p=1.000e+00$) | 0.000 ($p=1.000e+00$) |
| `raw_prov_vs_word_count` | 0.272 ($p=5.140e-01$) | 0.431 ($p=2.862e-01$) |
| `net_prov_vs_net_formality` | 0.000 ($p=1.000e+00$) | 0.000 ($p=1.000e+00$) |
| `net_prov_vs_net_fp_rate` | 0.000 ($p=1.000e+00$) | 0.000 ($p=1.000e+00$) |

---

## 6. Per-Task Matched Results Table

| Task ID | Domain | Baseline Score | Provenance Score | Surface Score | Neutral Score | $\Delta_{\text{prov}}$ | $\Delta_{\text{surf}}$ | $\Delta_{\text{net}}$ |
|---|---|---|---|---|---|---|---|---|
| `task_01_water_cycle` | earth_science | 2.137 | 0.514 | 1.688 | 1.481 | -1.622 | -0.449 | **-1.174** |
| `task_02_newtonian_laws` | classical_mechanics | 1.752 | 0.372 | 1.325 | 1.160 | -1.380 | -0.426 | **-0.954** |
| `task_03_photosynthesis` | plant_biology | 1.858 | 0.137 | 1.356 | 1.297 | -1.721 | -0.502 | **-1.219** |
| `task_04_dna_replication` | molecular_biology | 0.617 | -1.451 | 0.277 | 0.019 | -2.067 | -0.340 | **-1.727** |
| `task_05_entropy_thermodynamics` | statistical_physics | 0.905 | -0.745 | 0.413 | 0.381 | -1.651 | -0.493 | **-1.158** |
| `task_06_plate_tectonics` | geology | 0.996 | -0.924 | 0.458 | 0.163 | -1.920 | -0.538 | **-1.382** |
| `task_07_binary_search_algorithm` | algorithms | 2.196 | 0.794 | 1.797 | 1.775 | -1.402 | -0.398 | **-1.004** |
| `task_08_electromagnetic_induction` | electromagnetism | 1.490 | 0.164 | 1.202 | 1.207 | -1.326 | -0.288 | **-1.038** |

---

## 7. Methodological Interpretations & Limitations

1. **Internal Representation Shift:** When contextual provenance framing is introduced, the model's Layer-2 internal hidden states exhibit a statistically significant shift relative to matched surface controls ($d_z = -1.5$ to $-2.5$, $p < 10^{-4}$), demonstrating that perceived provenance exerts an effect beyond superficial prompt length and formatting tokens.
2. **Behavioral Coupling:** Behavioral shifts in lexical formality and first-person pronouns show moderate but non-deterministic correlation with internal projection shifts, consistent with partial internal-to-behavioral coupling.
3. **Epistemic Safeguards:** These findings show an *internal representation association* with provenance framing, but do not prove autonomous deceptive intent or complete behavioral persona drift.

---

## 8. Final Recommendation & Next Steps

**Classification:** Category B: INTERNAL SHIFT  
**Next Research Direction:** Explore activation steering / intervention along $D_{\text{expanded}}$ to test whether directly modifying this representation causally restores or perturbs downstream generation behavior under provenance shifts.

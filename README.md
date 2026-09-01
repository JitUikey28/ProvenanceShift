# ProvenanceShift

**"Who Does the Model Think Is Speaking? Provenance and Contextual Role as Mechanisms for Covert Persona Drift"**

## What is this?

ProvenanceShift is a research framework for investigating whether contextual information about the *source*, *role*, or *provenance* of text can influence which persona an instruction-tuned language model is operating from.

The conceptual causal chain under investigation:

```
Contextual / provenance manipulation
          ↓
Persona representation / latent state
          ↓
Observable behavioural change
```

**Important:** This is an empirical hypothesis under test, not an established finding.

---

## Current Scope & Capabilities (Phases 1–6)

1. **Configuration & Data Schemas (Phase 1):** Validated YAML configuration system, reproducible environment snapshots, structured logging, and schemas for prompts, activations, and experiments.
2. **Reproducible Generation Pipeline (Phase 2):** Hugging Face causal LM abstraction (`src/models/loader.py`, `src/models/generation.py`), deterministic generation, chat templates, and overwrite-protected result writers (`results/raw/<experiment_id>/`).
3. **Activation Extraction Engine (Phase 3):** Memory-efficient hidden-state extraction across layers, token pooling (`last_token`, `mean_pool`), compressed `.npz` storage, and JSON manifests (`src/activations/extractor.py`, `src/activations/storage.py`).
4. **Persona Representation Construction (Phase 4):** Matched benign persona dataset (`data/prompts/persona_pilot.json`), training-only contrastive mean-difference direction construction, projection scores, Cohen's d with bootstrap confidence intervals, and layer sweep analysis (`src/activations/direction.py`, `src/activations/analysis.py`).
5. **Rigorous Representation Validation (Phase 5):** Validation-based layer selection, direction stability resampling ($B=100$), empirical random-direction null distribution ($K=100$), label-shuffling controls, and train-fitted PCA (`src/activations/validation.py`, `scripts/run_persona_validation.py`).
6. **Controlled Provenance Investigation (Phase 6):** 4-condition matched task dataset (`data/prompts/provenance_pilot.json`), paired representation deltas ($\Delta_{\text{persona}}$ vs $\Delta_{\text{surface}}$ vs $\Delta_{\text{neutral}}$), paired Wilcoxon and t-tests, paired task-unit bootstrap, multiple comparisons correction (FDR / Bonferroni), modular behavioral evaluation (`src/evaluation/behavioral.py`), and representation-to-behavior correlation (`src/experiments/provenance.py`, `scripts/run_provenance_experiment.py`).

**No research experiments have been executed yet.** The repository provides tested, reproducible infrastructure.

---

## Hardware Constraints

Designed for **consumer-grade hardware**:

| Resource | Specification |
|----------|---------------|
| Primary GPU | NVIDIA RTX 3050 (4 GB VRAM) |
| System RAM | 16 GB |
| Cloud Fallback | Google Colab free GPU tier |
| Development Model | `microsoft/Phi-2` (2.7B) or `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

- Extraction runs in `torch.inference_mode()` with small batch sizes and immediate CPU movement.
- Full CPU fallback supported.
- Quantization (4-bit/8-bit via `bitsandbytes`) optional and configurable.

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ProvenanceShift

# Create a virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

Verify your environment:
```bash
python scripts/environment_check.py
```

---

## Running Offline Tests

All unit tests run completely offline without downloading models or requiring a GPU:

```bash
pytest
```

---

## Complete Workflow Guide

### 1. Model Text Generation (Phase 2)
```bash
python scripts/run_generation.py \
    --config configs/model.yaml \
    --prompt "Explain what machine learning is." \
    --experiment-id GEN-001 \
    --seed 42
```

### 2. Activation Extraction (Phase 3)
```bash
python scripts/extract_activations.py \
    --config configs/model.yaml \
    --prompt-file data/prompts/persona_pilot.json \
    --experiment-id ACT-001 \
    --layers all \
    --pooling last_token \
    --batch-size 4
```

### 3. Persona Representation Validation (Phase 5)
```bash
python scripts/run_persona_validation.py \
    --activations-dir results/raw/ACT-001/activations \
    --config configs/persona_validation.yaml \
    --experiment-id PERSONA-VALIDATION-001
```

Generated validation artifacts:
- **Table:** `results/tables/PERSONA-VALIDATION-001/persona_validation.csv`
- **Figures:**
  - `results/figures/PERSONA-VALIDATION-001/direction_stability.png`
  - `results/figures/PERSONA-VALIDATION-001/random_direction_null.png`
  - `results/figures/PERSONA-VALIDATION-001/train_fitted_pca.png`
- **Metadata:** `results/raw/PERSONA-VALIDATION-001/persona_validation_metadata.json`

### 4. Controlled Provenance Experiment (Phase 6)
```bash
python scripts/run_provenance_experiment.py \
    --config configs/model.yaml \
    --provenance-config configs/provenance.yaml \
    --prompt-file data/prompts/provenance_pilot.json \
    --validation-report results/raw/PERSONA-VALIDATION-001/persona_validation_metadata.json \
    --experiment-id PROVENANCE-PILOT-001
```

Generated provenance artifacts:
- **Summary Table:** `results/tables/PROVENANCE-PILOT-001/provenance_pilot.csv`
- **Paired Deltas Table:** `results/tables/PROVENANCE-PILOT-001/provenance_paired_deltas.csv`
- **Figures:**
  - `results/figures/PROVENANCE-PILOT-001/provenance_delta_distribution.png`
  - `results/figures/PROVENANCE-PILOT-001/representation_vs_behavior.png`
- **Metadata:** `results/raw/PROVENANCE-PILOT-001/provenance_experiment_metadata.json`

---

## Repository Structure

```
ProvenanceShift/
├── configs/
│   ├── model.yaml                    # Model and generation settings
│   ├── persona.yaml                  # Persona extraction & analysis settings
│   ├── persona_validation.yaml       # Phase 5 validation settings
│   ├── provenance.yaml               # Phase 6 provenance settings
│   └── experiment.yaml               # Experiment template
│
├── data/
│   └── prompts/
│       ├── example_prompts.json      # Simple generation examples
│       ├── persona_pilot.json        # Matched benign persona dataset (train/val/test)
│       ├── persona_paraphrase.json   # Paraphrase robustness dataset
│       └── provenance_pilot.json     # 4-condition matched provenance dataset (32 tasks)
│
├── src/
│   ├── activations/                  # Activation & representation modules
│   │   ├── extractor.py              # Layer-wise hidden state extraction & pooling
│   │   ├── storage.py                # Compressed NPZ & manifest saving/loading
│   │   ├── direction.py              # Mean-difference directions, Cohen's d, bootstrap CIs
│   │   ├── analysis.py               # Layer sweep orchestration, tables, and plotting
│   │   ├── validation.py             # Phase 5 validation, stability, random null, PCA
│   │   └── schemas.py                # Activation metadata schemas
│   ├── evaluation/                   # Evaluation modules
│   │   ├── behavioral.py             # Phase 6 behavioral & stylistic metrics
│   │   └── schemas.py                # Evaluation result schemas
│   ├── experiments/                  # Experiment orchestration
│   │   ├── provenance.py             # Phase 6 matched deltas, paired tests, multiple comparisons
│   │   └── schemas.py                # Experiment configuration schemas
│   ├── models/                       # Model inference pipeline
│   │   ├── loader.py                 # Model & tokenizer loader with fallback
│   │   ├── generation.py             # Deterministic text generation
│   │   └── results.py                # Overwrite-protected result storage
│   ├── prompting/                    # Prompt schemas
│   └── utils/                        # Logging, reproducibility, system diagnosis
│
├── scripts/
│   ├── environment_check.py          # Environment verification
│   ├── run_generation.py             # Phase 2 CLI generation
│   ├── extract_activations.py        # Phase 3 CLI activation extraction
│   ├── run_persona_analysis.py       # Phase 4 CLI persona analysis
│   ├── run_persona_validation.py     # Phase 5 CLI persona validation
│   └── run_provenance_experiment.py  # Phase 6 CLI provenance experiment
│
├── tests/                            # Comprehensive offline test suite (135 tests)
│   ├── test_config.py
│   ├── test_schemas.py
│   ├── test_reproducibility.py
│   ├── test_generation.py
│   ├── test_activations.py
│   ├── test_persona_analysis.py
│   ├── test_persona_validation.py
│   ├── test_behavioral.py
│   ├── test_provenance_experiment.py
│   └── integration/                  # Model integration tests (excluded by default)
│
├── results/                          # Gitignored experiment outputs
│   ├── raw/
│   ├── tables/
│   └── figures/
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── reproducibility.md
│
├── README.md
├── RESEARCH_CONTEXT.md
├── EXPERIMENT_LOG.md
├── CHANGELOG.md
└── LICENSE
```

---

## Scientific Rigour & Ethics

- **No Overclaiming:** We test whether provenance manipulation is associated with a measurable change in candidate persona representations.
- **No Data Leakage:** Directions, thresholds, and classifier baselines are trained exclusively on the training split; layers and hyperparameters are selected via validation splits; test sets evaluate held-out semantic topics.
- **Consumer Hardware Friendly:** All pipelines are tested and sized for 4 GB VRAM (RTX 3050) without massive activation bloat.

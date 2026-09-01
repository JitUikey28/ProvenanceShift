#!/usr/bin/env python3
"""
Environment check script for ProvenanceShift.

Run this after installation to verify that all dependencies are available
and to print a summary of the hardware environment.

Usage:
    python scripts/environment_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `src` is importable.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main() -> None:
    # -- Core imports -------------------------------------------------
    print("Checking core dependencies...\n")

    errors: list[str] = []

    try:
        import torch
        print(f"  [OK] torch {torch.__version__}")
    except ImportError:
        errors.append("torch")

    try:
        import transformers
        print(f"  [OK] transformers {transformers.__version__}")
    except ImportError:
        errors.append("transformers")

    try:
        import numpy
        print(f"  [OK] numpy {numpy.__version__}")
    except ImportError:
        errors.append("numpy")

    try:
        import pandas
        print(f"  [OK] pandas {pandas.__version__}")
    except ImportError:
        errors.append("pandas")

    try:
        import scipy
        print(f"  [OK] scipy {scipy.__version__}")
    except ImportError:
        errors.append("scipy")

    try:
        import sklearn
        print(f"  [OK] scikit-learn {sklearn.__version__}")
    except ImportError:
        errors.append("scikit-learn")

    try:
        import yaml
        print(f"  [OK] pyyaml {yaml.__version__}")
    except ImportError:
        errors.append("pyyaml")

    try:
        import matplotlib
        print(f"  [OK] matplotlib {matplotlib.__version__}")
    except ImportError:
        errors.append("matplotlib")

    try:
        import tqdm
        print(f"  [OK] tqdm {tqdm.__version__}")
    except ImportError:
        errors.append("tqdm")

    # -- Optional dependencies ----------------------------------------
    print("\nOptional dependencies:")

    try:
        import bitsandbytes
        print(f"  [OK] bitsandbytes {bitsandbytes.__version__}")
    except ImportError:
        print("  [SKIP] bitsandbytes -- not installed (needed for quantization)")

    try:
        import accelerate
        print(f"  [OK] accelerate {accelerate.__version__}")
    except ImportError:
        print("  [SKIP] accelerate -- not installed (needed for quantization)")

    try:
        import pytest
        print(f"  [OK] pytest {pytest.__version__}")
    except ImportError:
        print("  [SKIP] pytest -- not installed (needed for running tests)")

    # -- Errors -------------------------------------------------------
    if errors:
        print(f"\n[FAIL] Missing required dependencies: {', '.join(errors)}")
        print("  Install them with: pip install -r requirements.txt")
        sys.exit(1)

    # -- System info --------------------------------------------------
    print()
    from src.utils.system import print_system_info
    print_system_info()

    # -- Package import check -----------------------------------------
    print("\nVerifying src package imports...")
    try:
        from src.models.loader import ModelConfig, load_model_config
        from src.prompting.schemas import PromptItem, PromptSet
        from src.experiments.schemas import ExperimentConfig
        from src.activations.schemas import ActivationMetadata
        from src.evaluation.schemas import EvaluationResult
        from src.utils.reproducibility import set_seed, capture_environment
        from src.utils.logging import get_logger
        print("  [OK] All src modules imported successfully.")
    except ImportError as exc:
        print(f"  [FAIL] Import error: {exc}")
        sys.exit(1)

    print("\n[OK] Environment check passed. Ready for experiments.")


if __name__ == "__main__":
    main()

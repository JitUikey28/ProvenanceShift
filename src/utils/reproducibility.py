"""
Reproducibility utilities.

Provides deterministic seed-setting, git commit retrieval, and an
experiment-metadata snapshot that captures every parameter required
to reproduce a run.
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Integer seed.  Must be non-negative.

    Raises
    ------
    ValueError
        If *seed* is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enforce deterministic behaviour where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit(repo_dir: Optional[Path] = None) -> Optional[str]:
    """Return the current HEAD commit hash, or ``None`` if unavailable.

    Parameters
    ----------
    repo_dir:
        Path to the repository root.  Defaults to the current working
        directory.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir) if repo_dir else None,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_git_dirty(repo_dir: Optional[Path] = None) -> Optional[bool]:
    """Return ``True`` if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir) if repo_dir else None,
            timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


@dataclass
class ExperimentSnapshot:
    """Immutable record of everything needed to reproduce an experiment.

    This is *not* the full experiment schema (see
    ``src.experiments.schemas.ExperimentConfig``).  It captures the *runtime*
    environment at execution time.
    """

    # Versions
    python_version: str = ""
    torch_version: str = ""
    transformers_version: str = ""
    numpy_version: str = ""

    # Hardware
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None

    # Reproducibility
    seed: Optional[int] = None
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None

    # Timestamp
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def capture_environment(
    seed: Optional[int] = None, repo_dir: Optional[Path] = None
) -> ExperimentSnapshot:
    """Capture a full snapshot of the current runtime environment.

    Parameters
    ----------
    seed:
        The seed that was (or will be) used for this experiment.
    repo_dir:
        Path to the repository root for git commit detection.

    Returns
    -------
    ExperimentSnapshot
    """
    import sys
    import transformers

    snap = ExperimentSnapshot(
        python_version=sys.version,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        numpy_version=np.__version__,
        cuda_available=torch.cuda.is_available(),
        seed=seed,
        git_commit=get_git_commit(repo_dir),
        git_dirty=get_git_dirty(repo_dir),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if torch.cuda.is_available():
        snap.cuda_version = torch.version.cuda
        snap.gpu_name = torch.cuda.get_device_name(0)
        snap.gpu_memory_mb = int(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        )

    return snap

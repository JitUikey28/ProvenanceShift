"""
Tests for reproducibility utilities.

Verifies seed-setting, git commit retrieval, and environment snapshot
capture.  No GPU or model downloads required.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.utils.reproducibility import (
    ExperimentSnapshot,
    capture_environment,
    get_git_commit,
    set_seed,
)


class TestSetSeed:

    def test_deterministic_python_random(self) -> None:
        """Same seed → same Python random output."""
        set_seed(0)
        a = random.random()
        set_seed(0)
        b = random.random()
        assert a == b

    def test_deterministic_numpy(self) -> None:
        """Same seed → same NumPy random output."""
        set_seed(0)
        a = np.random.rand(5)
        set_seed(0)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_torch(self) -> None:
        """Same seed → same PyTorch random output."""
        set_seed(0)
        a = torch.rand(5)
        set_seed(0)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_negative_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            set_seed(-1)

    def test_zero_seed(self) -> None:
        """Seed 0 should be valid."""
        set_seed(0)  # should not raise


class TestGetGitCommit:

    def test_returns_string_or_none(self) -> None:
        """Should return a commit hash string or None (never raise)."""
        result = get_git_commit()
        assert result is None or (isinstance(result, str) and len(result) == 40)


class TestCaptureEnvironment:

    def test_snapshot_fields(self) -> None:
        """Snapshot should populate version strings."""
        snap = capture_environment(seed=42)
        assert snap.python_version != ""
        assert snap.torch_version != ""
        assert snap.transformers_version != ""
        assert snap.timestamp != ""
        assert snap.seed == 42

    def test_snapshot_to_dict(self) -> None:
        """to_dict() should return a plain dictionary."""
        snap = capture_environment(seed=0)
        d = snap.to_dict()
        assert isinstance(d, dict)
        assert "python_version" in d
        assert "cuda_available" in d


class TestExperimentSnapshot:

    def test_dataclass_defaults(self) -> None:
        snap = ExperimentSnapshot()
        assert snap.cuda_available is False
        assert snap.seed is None

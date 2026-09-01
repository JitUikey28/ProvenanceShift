"""
Tests for configuration loading and validation.

These tests verify that YAML configs are parsed correctly and that
invalid configurations are rejected loudly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.models.loader import ModelConfig, load_model_config
from src.experiments.schemas import ExperimentConfig


# =====================================================================
# ModelConfig
# =====================================================================

class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_default_config(self) -> None:
        """Default ModelConfig should have model_name=None."""
        cfg = ModelConfig()
        assert cfg.model_name is None
        assert cfg.device == "auto"
        assert cfg.dtype == "auto"
        assert cfg.load_in_4bit is False

    def test_reject_both_quantization_flags(self) -> None:
        """Cannot set both load_in_4bit and load_in_8bit."""
        with pytest.raises(ValueError, match="Cannot set both"):
            ModelConfig(load_in_4bit=True, load_in_8bit=True)

    def test_resolve_device_cpu(self) -> None:
        """Explicit 'cpu' should resolve to cpu device."""
        cfg = ModelConfig(device="cpu")
        assert str(cfg.resolve_device()) == "cpu"

    def test_resolve_dtype_float16(self) -> None:
        """'float16' should resolve to torch.float16."""
        import torch
        cfg = ModelConfig(dtype="float16")
        assert cfg.resolve_dtype() == torch.float16

    def test_resolve_dtype_invalid(self) -> None:
        """Invalid dtype string should raise."""
        cfg = ModelConfig(dtype="invalid_dtype")
        with pytest.raises(ValueError, match="Unsupported dtype"):
            cfg.resolve_dtype()


class TestLoadModelConfig:
    """Tests for YAML config loading."""

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """Should parse a valid YAML config."""
        config_data = {
            "model_name": "test/model",
            "device": "cpu",
            "dtype": "float32",
            "load_in_4bit": False,
        }
        config_file = tmp_path / "model.yaml"
        with open(config_file, "w") as fh:
            yaml.dump(config_data, fh)

        cfg = load_model_config(config_file)
        assert cfg.model_name == "test/model"
        assert cfg.device == "cpu"

    def test_missing_file_raises(self) -> None:
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_model_config(Path("/nonexistent/path/model.yaml"))

    def test_empty_yaml(self, tmp_path: Path) -> None:
        """Empty YAML should produce default config (model_name=None)."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        cfg = load_model_config(config_file)
        assert cfg.model_name is None


# =====================================================================
# ExperimentConfig
# =====================================================================

class TestExperimentConfig:
    """Tests for ExperimentConfig validation."""

    def test_valid_config(self) -> None:
        """A fully specified config should validate without errors."""
        cfg = ExperimentConfig(
            experiment_id="test_001",
            experiment_name="Test Experiment",
            seed=42,
        )
        cfg.validate()  # should not raise

    def test_missing_experiment_id(self) -> None:
        """Missing experiment_id should fail validation."""
        cfg = ExperimentConfig(experiment_name="Test")
        with pytest.raises(ValueError, match="experiment_id"):
            cfg.validate()

    def test_missing_experiment_name(self) -> None:
        """Missing experiment_name should fail validation."""
        cfg = ExperimentConfig(experiment_id="test_001")
        with pytest.raises(ValueError, match="experiment_name"):
            cfg.validate()

    def test_negative_seed(self) -> None:
        """Negative seed should fail validation."""
        cfg = ExperimentConfig(
            experiment_id="test",
            experiment_name="test",
            seed=-1,
        )
        with pytest.raises(ValueError, match="seed"):
            cfg.validate()

    def test_invalid_top_p(self) -> None:
        """top_p outside (0, 1] should fail validation."""
        cfg = ExperimentConfig(
            experiment_id="test",
            experiment_name="test",
            top_p=0.0,
        )
        with pytest.raises(ValueError, match="top_p"):
            cfg.validate()

    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        """Config should survive YAML serialisation and deserialisation."""
        original = ExperimentConfig(
            experiment_id="test_001",
            experiment_name="Round Trip Test",
            seed=123,
            temperature=0.7,
        )
        path = tmp_path / "experiment.yaml"
        original.save_yaml(path)
        loaded = ExperimentConfig.from_yaml(path)
        assert loaded.experiment_id == original.experiment_id
        assert loaded.seed == original.seed
        assert loaded.temperature == original.temperature

    def test_stamp_sets_timestamp(self) -> None:
        """stamp() should set a non-empty ISO timestamp."""
        cfg = ExperimentConfig(
            experiment_id="test",
            experiment_name="test",
        )
        assert cfg.timestamp == ""
        cfg.stamp()
        assert cfg.timestamp != ""
        assert "T" in cfg.timestamp  # basic ISO format check

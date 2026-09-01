"""
Tests for the generation pipeline: configuration, results, experiment IDs,
result writer, device selection, and seed handling.

All tests are offline — no model downloads, no GPU required.
Mock objects are used for model and tokenizer.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.generation import GenerationConfig, GenerationResult, format_prompt
from src.models.results import (
    ResultWriter,
    build_result_record,
    generate_experiment_id,
    validate_experiment_id,
)
from src.models.loader import ModelConfig, ModelBundle


# =====================================================================
# GenerationConfig
# =====================================================================

class TestGenerationConfig:
    """Tests for GenerationConfig validation."""

    def test_defaults(self) -> None:
        """Default config should be valid."""
        cfg = GenerationConfig()
        cfg.validate()  # should not raise

    def test_negative_seed_fails(self) -> None:
        cfg = GenerationConfig(seed=-1)
        with pytest.raises(ValueError, match="seed"):
            cfg.validate()

    def test_negative_temperature_fails(self) -> None:
        cfg = GenerationConfig(temperature=-0.5)
        with pytest.raises(ValueError, match="temperature"):
            cfg.validate()

    def test_invalid_top_p_fails(self) -> None:
        cfg = GenerationConfig(top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            cfg.validate()

    def test_top_p_above_one_fails(self) -> None:
        cfg = GenerationConfig(top_p=1.5)
        with pytest.raises(ValueError, match="top_p"):
            cfg.validate()

    def test_negative_top_k_fails(self) -> None:
        cfg = GenerationConfig(top_k=-1)
        with pytest.raises(ValueError, match="top_k"):
            cfg.validate()

    def test_zero_max_new_tokens_fails(self) -> None:
        cfg = GenerationConfig(max_new_tokens=0)
        with pytest.raises(ValueError, match="max_new_tokens"):
            cfg.validate()

    def test_zero_repetition_penalty_fails(self) -> None:
        cfg = GenerationConfig(repetition_penalty=0.0)
        with pytest.raises(ValueError, match="repetition_penalty"):
            cfg.validate()

    def test_valid_sampling_config(self) -> None:
        cfg = GenerationConfig(
            seed=123,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_new_tokens=512,
            do_sample=True,
            repetition_penalty=1.1,
        )
        cfg.validate()  # should not raise

    def test_dict_round_trip(self) -> None:
        original = GenerationConfig(seed=99, temperature=0.5, top_p=0.8)
        d = original.to_dict()
        restored = GenerationConfig.from_dict(d)
        assert restored.seed == 99
        assert restored.temperature == 0.5
        assert restored.top_p == 0.8

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {"seed": 42, "unknown_key": "ignored"}
        cfg = GenerationConfig.from_dict(d)
        assert cfg.seed == 42


# =====================================================================
# GenerationResult
# =====================================================================

class TestGenerationResult:
    """Tests for GenerationResult schema."""

    def test_defaults(self) -> None:
        result = GenerationResult()
        assert result.prompt == ""
        assert result.text == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_dict_round_trip(self) -> None:
        original = GenerationResult(
            prompt="Hello",
            text="World",
            input_tokens=5,
            output_tokens=3,
            total_tokens=8,
            generation_seconds=1.23,
            tokens_per_second=2.44,
            seed=42,
            device="cpu",
            dtype="torch.float32",
        )
        d = original.to_dict()
        assert d["prompt"] == "Hello"
        assert d["text"] == "World"

        restored = GenerationResult.from_dict(d)
        assert restored.input_tokens == 5
        assert restored.device == "cpu"

    def test_to_dict_contains_all_fields(self) -> None:
        result = GenerationResult()
        d = result.to_dict()
        expected_keys = {
            "prompt", "input_tokens", "text", "output_tokens",
            "total_tokens", "generation_seconds", "tokens_per_second",
            "seed", "temperature", "top_p", "top_k", "max_new_tokens",
            "do_sample", "repetition_penalty", "device", "dtype",
        }
        assert expected_keys.issubset(set(d.keys()))


# =====================================================================
# Experiment ID
# =====================================================================

class TestExperimentId:
    """Tests for experiment ID generation and validation."""

    def test_auto_generation_format(self) -> None:
        eid = generate_experiment_id()
        assert eid.startswith("GEN-")
        # GEN-YYYYMMDD-HHMMSS → 4 + 8 + 1 + 6 = 19 chars
        assert len(eid) == 19

    def test_custom_prefix(self) -> None:
        eid = generate_experiment_id(prefix="TEST")
        assert eid.startswith("TEST-")

    def test_validate_valid_id(self) -> None:
        validate_experiment_id("GEN-001")  # should not raise
        validate_experiment_id("my-experiment_v2")  # should not raise

    def test_validate_empty_fails(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_experiment_id("")

    def test_validate_whitespace_only_fails(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_experiment_id("   ")

    def test_validate_unsafe_chars_fails(self) -> None:
        with pytest.raises(ValueError, match="unsafe"):
            validate_experiment_id("exp/001")

    def test_validate_colon_fails(self) -> None:
        with pytest.raises(ValueError, match="unsafe"):
            validate_experiment_id("exp:001")


# =====================================================================
# ResultWriter
# =====================================================================

class TestResultWriter:
    """Tests for ResultWriter overwrite protection and file I/O."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        writer = ResultWriter(
            base_dir=str(tmp_path),
            experiment_id="test-001",
        )
        assert (tmp_path / "test-001").is_dir()

    def test_write_metadata(self, tmp_path: Path) -> None:
        writer = ResultWriter(
            base_dir=str(tmp_path),
            experiment_id="test-002",
        )
        metadata = {"experiment_id": "test-002", "seed": 42}
        path = writer.write_metadata(metadata)
        assert path.exists()

        with open(path) as fh:
            loaded = json.load(fh)
        assert loaded["experiment_id"] == "test-002"

    def test_append_output(self, tmp_path: Path) -> None:
        writer = ResultWriter(
            base_dir=str(tmp_path),
            experiment_id="test-003",
        )
        record = {"text": "hello", "tokens": 5}
        path = writer.append_output(record)
        assert path.exists()

        with open(path) as fh:
            line = fh.readline()
        loaded = json.loads(line)
        assert loaded["text"] == "hello"

    def test_append_multiple_outputs(self, tmp_path: Path) -> None:
        writer = ResultWriter(
            base_dir=str(tmp_path),
            experiment_id="test-004",
        )
        writer.append_output({"i": 1})
        writer.append_output({"i": 2})

        path = tmp_path / "test-004" / "outputs.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_write_outputs_batch(self, tmp_path: Path) -> None:
        writer = ResultWriter(
            base_dir=str(tmp_path),
            experiment_id="test-005",
        )
        records = [{"i": 1}, {"i": 2}, {"i": 3}]
        path = writer.write_outputs(records)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_refuses_overwrite(self, tmp_path: Path) -> None:
        """Writing to an existing experiment should fail."""
        writer = ResultWriter(
            base_dir=str(tmp_path),
            experiment_id="test-006",
        )
        writer.write_metadata({"test": True})

        with pytest.raises(FileExistsError, match="already has results"):
            ResultWriter(
                base_dir=str(tmp_path),
                experiment_id="test-006",
            )

    def test_allows_overwrite_when_flag_set(self, tmp_path: Path) -> None:
        """Overwrite should work when explicitly enabled."""
        writer = ResultWriter(
            base_dir=str(tmp_path),
            experiment_id="test-007",
        )
        writer.write_metadata({"version": 1})

        # This should NOT raise
        writer2 = ResultWriter(
            base_dir=str(tmp_path),
            experiment_id="test-007",
            overwrite=True,
        )
        writer2.write_metadata({"version": 2})

        path = tmp_path / "test-007" / "metadata.json"
        with open(path) as fh:
            data = json.load(fh)
        assert data["version"] == 2


# =====================================================================
# Build result record
# =====================================================================

class TestBuildResultRecord:

    def test_structure(self) -> None:
        gen_result = GenerationResult(
            prompt="Hello",
            text="World",
            input_tokens=5,
            output_tokens=3,
            seed=42,
        )
        record = build_result_record(
            experiment_id="test-001",
            timestamp="2026-08-31T00:00:00Z",
            model_metadata={"name": "test/model", "revision": None,
                            "dtype": "float32", "device": "cpu",
                            "quantized": False},
            generation_result=gen_result,
        )
        assert record["experiment_id"] == "test-001"
        assert record["model"]["name"] == "test/model"
        assert record["generation"]["seed"] == 42
        assert record["input"]["prompt"] == "Hello"
        assert record["output"]["text"] == "World"
        assert record["timing"]["generation_seconds"] is not None

    def test_from_dict_input(self) -> None:
        """Should accept a dict as well as a GenerationResult."""
        gen_dict = {"prompt": "Hi", "text": "There", "input_tokens": 2,
                    "output_tokens": 1, "seed": 99}
        record = build_result_record(
            experiment_id="test-002",
            timestamp="2026-08-31T00:00:00Z",
            model_metadata={"name": "m", "revision": None, "dtype": "f32",
                            "device": "cpu", "quantized": False},
            generation_result=gen_dict,
        )
        assert record["generation"]["seed"] == 99


# =====================================================================
# Device selection logic (ModelConfig)
# =====================================================================

class TestDeviceSelection:
    """Tests for device resolution and fallback."""

    def test_explicit_cpu(self) -> None:
        cfg = ModelConfig(device="cpu")
        assert str(cfg.resolve_device()) == "cpu"

    def test_auto_without_cuda(self) -> None:
        cfg = ModelConfig(device="auto")
        with patch("torch.cuda.is_available", return_value=False):
            assert str(cfg.resolve_device()) == "cpu"

    def test_auto_with_cuda(self) -> None:
        cfg = ModelConfig(device="auto")
        with patch("torch.cuda.is_available", return_value=True):
            assert str(cfg.resolve_device()) == "cuda"

    def test_cuda_unavailable_with_fallback(self) -> None:
        cfg = ModelConfig(device="cuda", allow_cpu_fallback=True)
        with patch("torch.cuda.is_available", return_value=False):
            assert str(cfg.resolve_device()) == "cpu"

    def test_cuda_unavailable_without_fallback_raises(self) -> None:
        cfg = ModelConfig(device="cuda", allow_cpu_fallback=False)
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
                cfg.resolve_device()


# =====================================================================
# Dtype resolution
# =====================================================================

class TestDtypeResolution:

    def test_auto_on_cuda_returns_float16(self) -> None:
        cfg = ModelConfig(dtype="auto")
        result = cfg.resolve_dtype(device=torch.device("cuda"))
        assert result == torch.float16

    def test_auto_on_cpu_returns_float32(self) -> None:
        cfg = ModelConfig(dtype="auto")
        result = cfg.resolve_dtype(device=torch.device("cpu"))
        assert result == torch.float32

    def test_explicit_float16(self) -> None:
        cfg = ModelConfig(dtype="float16")
        assert cfg.resolve_dtype() == torch.float16

    def test_explicit_bfloat16(self) -> None:
        cfg = ModelConfig(dtype="bfloat16")
        assert cfg.resolve_dtype() == torch.bfloat16

    def test_invalid_dtype_raises(self) -> None:
        cfg = ModelConfig(dtype="float64")
        with pytest.raises(ValueError, match="Unsupported dtype"):
            cfg.resolve_dtype()


# =====================================================================
# Model metadata
# =====================================================================

class TestModelMetadata:

    def test_metadata_structure(self) -> None:
        cfg = ModelConfig(
            model_name="test/model",
            revision="abc123",
            load_in_4bit=True,
        )
        meta = cfg.get_model_metadata(
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert meta["name"] == "test/model"
        assert meta["revision"] == "abc123"
        assert meta["quantized"] is True
        assert meta["quantization_bits"] == 4
        assert meta["device"] == "cpu"

    def test_no_quantization(self) -> None:
        cfg = ModelConfig(model_name="test/model")
        meta = cfg.get_model_metadata(
            device=torch.device("cpu"),
            dtype=None,
        )
        assert meta["quantized"] is False
        assert meta["quantization_bits"] is None
        assert meta["dtype"] == "model_default"


# =====================================================================
# Prompt formatting
# =====================================================================

class TestFormatPrompt:

    def test_plain_text_without_chat_template(self) -> None:
        """Without a chat template, plain text should pass through."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        result = format_prompt(tokenizer, "Hello world")
        assert result == "Hello world"

    def test_messages_without_chat_template(self) -> None:
        """Without a template, messages should be joined."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = format_prompt(tokenizer, messages)
        assert "Hello" in result
        assert "Hi" in result

    def test_chat_template_used(self) -> None:
        """When available, the chat template should be called."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "some_template"
        tokenizer.apply_chat_template.return_value = "<formatted>"
        result = format_prompt(tokenizer, "Hello")
        assert result == "<formatted>"
        tokenizer.apply_chat_template.assert_called_once()

    def test_chat_template_failure_fallback(self) -> None:
        """If the chat template raises, fall back to plain text."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "some_template"
        tokenizer.apply_chat_template.side_effect = Exception("template error")
        result = format_prompt(tokenizer, "Hello")
        assert result == "Hello"


# =====================================================================
# Seed handling in config
# =====================================================================

class TestSeedHandling:

    def test_seed_in_model_config(self) -> None:
        cfg = ModelConfig(seed=123)
        assert cfg.seed == 123

    def test_seed_default(self) -> None:
        cfg = ModelConfig()
        assert cfg.seed == 42

    def test_seed_recorded_in_generation_config(self) -> None:
        gen = GenerationConfig(seed=99)
        d = gen.to_dict()
        assert d["seed"] == 99

    def test_seed_recorded_in_result(self) -> None:
        result = GenerationResult(seed=77)
        assert result.to_dict()["seed"] == 77


# =====================================================================
# ModelConfig new fields
# =====================================================================

class TestModelConfigNewFields:

    def test_allow_cpu_fallback_default(self) -> None:
        cfg = ModelConfig()
        assert cfg.allow_cpu_fallback is True

    def test_max_memory_default(self) -> None:
        cfg = ModelConfig()
        assert cfg.max_memory is None

    def test_max_memory_set(self) -> None:
        cfg = ModelConfig(max_memory="3500MiB")
        assert cfg.max_memory == "3500MiB"

"""
Integration tests for the generation pipeline.

These tests download a real model and run inference.  They are NOT
included in the default ``pytest`` run — use::

    pytest tests/integration/ -v

Prerequisites:
    - Internet access (for model download on first run)
    - Sufficient RAM / VRAM for the configured model
    - ``configs/model.yaml`` must specify a valid model

These tests verify the end-to-end pipeline:
    1. Model loads from config.
    2. Tokenizer loads.
    3. A prompt generates non-empty output.
    4. Metadata is written to disk.
    5. Output JSONL is valid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def model_bundle():
    """Load the model once for all integration tests."""
    from src.models.loader import load_model, load_model_config

    config_path = _PROJECT_ROOT / "configs" / "model.yaml"
    config = load_model_config(config_path)

    if not config.model_name:
        pytest.skip("No model_name configured in configs/model.yaml")

    bundle = load_model(config)
    return bundle, config


class TestModelLoading:
    """Verify model and tokenizer load correctly."""

    def test_model_loads(self, model_bundle) -> None:
        bundle, config = model_bundle
        assert bundle.model is not None

    def test_tokenizer_loads(self, model_bundle) -> None:
        bundle, config = model_bundle
        assert bundle.tokenizer is not None

    def test_model_in_eval_mode(self, model_bundle) -> None:
        bundle, _ = model_bundle
        assert not bundle.model.training

    def test_metadata_populated(self, model_bundle) -> None:
        bundle, _ = model_bundle
        assert "name" in bundle.metadata
        assert "device" in bundle.metadata
        assert "load_time_seconds" in bundle.metadata


class TestGeneration:
    """Verify that generation produces valid output."""

    def test_generates_non_empty_text(self, model_bundle) -> None:
        from src.models.generation import GenerationConfig, generate

        bundle, config = model_bundle
        gen_config = GenerationConfig(
            seed=42,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=32,
        )
        result = generate(bundle, "What is 2 + 2?", gen_config)
        assert len(result.text.strip()) > 0

    def test_token_counts_positive(self, model_bundle) -> None:
        from src.models.generation import GenerationConfig, generate

        bundle, config = model_bundle
        gen_config = GenerationConfig(
            seed=42,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=16,
        )
        result = generate(bundle, "Hello", gen_config)
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.total_tokens == result.input_tokens + result.output_tokens

    def test_generation_seconds_positive(self, model_bundle) -> None:
        from src.models.generation import GenerationConfig, generate

        bundle, config = model_bundle
        gen_config = GenerationConfig(seed=42, max_new_tokens=8)
        result = generate(bundle, "Test", gen_config)
        assert result.generation_seconds > 0


class TestResultStorage:
    """Verify that results are written correctly."""

    def test_full_pipeline_writes_valid_json(self, model_bundle, tmp_path) -> None:
        from src.models.generation import GenerationConfig, generate
        from src.models.results import (
            ResultWriter,
            build_result_record,
        )
        from datetime import datetime, timezone

        bundle, config = model_bundle
        gen_config = GenerationConfig(seed=42, max_new_tokens=16)

        result = generate(bundle, "Hello world", gen_config)

        writer = ResultWriter(
            base_dir=str(tmp_path),
            experiment_id="integration-test-001",
        )

        timestamp = datetime.now(timezone.utc).isoformat()
        record = build_result_record(
            experiment_id="integration-test-001",
            timestamp=timestamp,
            model_metadata=bundle.metadata,
            generation_result=result,
        )

        writer.write_metadata({
            "experiment_id": "integration-test-001",
            "model": bundle.metadata,
        })
        writer.append_output(record)

        # Verify metadata.json
        meta_path = tmp_path / "integration-test-001" / "metadata.json"
        assert meta_path.exists()
        with open(meta_path) as fh:
            meta = json.load(fh)
        assert meta["experiment_id"] == "integration-test-001"

        # Verify outputs.jsonl
        out_path = tmp_path / "integration-test-001" / "outputs.jsonl"
        assert out_path.exists()
        with open(out_path) as fh:
            line = fh.readline()
        output = json.loads(line)
        assert output["output"]["text"]  # non-empty
        assert output["experiment_id"] == "integration-test-001"

"""
Tests for data schemas: PromptItem, PromptSet, ActivationMetadata,
and EvaluationResult.

All tests are offline — no model downloads or GPU required.
"""

from __future__ import annotations

import pytest

from src.prompting.schemas import PromptItem, PromptSet
from src.activations.schemas import ActivationMetadata
from src.evaluation.schemas import EvaluationResult


# =====================================================================
# PromptItem
# =====================================================================

class TestPromptItem:

    def test_valid_prompt(self) -> None:
        p = PromptItem(prompt_id="p001", base_content="Hello world")
        assert p.prompt_id == "p001"
        assert p.role == "user"
        assert p.turn == 0

    def test_missing_prompt_id(self) -> None:
        with pytest.raises(ValueError, match="prompt_id"):
            PromptItem(base_content="Hello")

    def test_missing_base_content(self) -> None:
        with pytest.raises(ValueError, match="base_content"):
            PromptItem(prompt_id="p001")

    def test_dict_round_trip(self) -> None:
        original = PromptItem(
            prompt_id="p001",
            base_content="Test content",
            condition="baseline",
            provenance="system_message",
            role="system",
            turn=1,
            conversation_id="conv_01",
            metadata={"key": "value"},
        )
        d = original.to_dict()
        restored = PromptItem.from_dict(d)
        assert restored.prompt_id == original.prompt_id
        assert restored.provenance == original.provenance
        assert restored.metadata == original.metadata

    def test_condition_defaults_to_empty(self) -> None:
        p = PromptItem(prompt_id="p001", base_content="Hello")
        assert p.condition == ""


# =====================================================================
# PromptSet
# =====================================================================

class TestPromptSet:

    def test_valid_set(self) -> None:
        ps = PromptSet(
            set_id="set_001",
            prompts=[
                PromptItem(prompt_id="p1", base_content="A"),
                PromptItem(prompt_id="p2", base_content="B"),
            ],
        )
        assert len(ps) == 2
        assert ps.prompt_ids == ["p1", "p2"]

    def test_missing_set_id(self) -> None:
        with pytest.raises(ValueError, match="set_id"):
            PromptSet()

    def test_iterable(self) -> None:
        ps = PromptSet(
            set_id="s1",
            prompts=[PromptItem(prompt_id="p1", base_content="A")],
        )
        items = list(ps)
        assert len(items) == 1


# =====================================================================
# ActivationMetadata
# =====================================================================

class TestActivationMetadata:

    def test_valid_metadata(self) -> None:
        am = ActivationMetadata(
            model="test/model",
            layer=12,
            component="residual_stream",
            shape=(1, 768),
        )
        assert am.layer == 12
        assert am.shape == (1, 768)

    def test_missing_model(self) -> None:
        with pytest.raises(ValueError, match="model"):
            ActivationMetadata()

    def test_dict_round_trip(self) -> None:
        original = ActivationMetadata(
            model="test/model",
            layer=5,
            shape=(32, 2048),
            experiment_id="exp_001",
        )
        d = original.to_dict()
        assert isinstance(d["shape"], list)  # tuple → list for JSON
        restored = ActivationMetadata.from_dict(d)
        assert restored.shape == (32, 2048)  # list → tuple on load


# =====================================================================
# EvaluationResult
# =====================================================================

class TestEvaluationResult:

    def test_valid_result(self) -> None:
        er = EvaluationResult(
            experiment_id="exp_001",
            metric_name="refusal_rate",
            metric_value=0.15,
        )
        assert er.metric_value == 0.15

    def test_missing_experiment_id(self) -> None:
        with pytest.raises(ValueError, match="experiment_id"):
            EvaluationResult(metric_name="test")

    def test_missing_metric_name(self) -> None:
        with pytest.raises(ValueError, match="metric_name"):
            EvaluationResult(experiment_id="exp_001")

    def test_dict_round_trip(self) -> None:
        original = EvaluationResult(
            experiment_id="exp_001",
            metric_name="cosine_similarity",
            metric_value=0.92,
            condition="provenance_system",
            details={"n": 100},
        )
        d = original.to_dict()
        restored = EvaluationResult.from_dict(d)
        assert restored.metric_value == original.metric_value
        assert restored.details == original.details

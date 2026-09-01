"""
Offline unit tests for behavioral and stylistic evaluation (Phase 6).

Verifies lexical statistics, first-person pronoun rates, assistant disclaimer
detection, formality scoring, and role keyword adherence heuristics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.behavioral import BehavioralEvaluator, BehavioralMetrics


class TestBehavioralEvaluator:

    def test_empty_string(self) -> None:
        evaluator = BehavioralEvaluator()
        metrics = evaluator.evaluate_text("")
        assert metrics.word_count == 0
        assert metrics.char_count == 0
        assert metrics.first_person_count == 0

    def test_first_person_and_length(self) -> None:
        evaluator = BehavioralEvaluator()
        text = "I believe that my observations show a clear trend for me."
        metrics = evaluator.evaluate_text(text)
        assert metrics.word_count == 11
        # 3 first-person pronouns: "I", "my", "me"
        assert metrics.first_person_count == 3
        assert np.isclose(metrics.first_person_rate, 27.27, atol=0.01)

    def test_assistant_disclaimer_detection(self) -> None:
        evaluator = BehavioralEvaluator()
        text_with_disc = "As an AI language model, I do not possess personal beliefs."
        metrics_with = evaluator.evaluate_text(text_with_disc)
        assert metrics_with.assistant_disclaimer_count >= 1

        text_without = "Photosynthesis occurs primarily in the chloroplasts of plant cells."
        metrics_without = evaluator.evaluate_text(text_without)
        assert metrics_without.assistant_disclaimer_count == 0

    def test_formality_score(self) -> None:
        evaluator = BehavioralEvaluator()
        formal_text = "Furthermore, the data consequently supports the hypothesis; moreover, it is consistent."
        informal_text = "It's cool, but I don't think you're right, and we're not sure."

        m_formal = evaluator.evaluate_text(formal_text)
        m_informal = evaluator.evaluate_text(informal_text)

        assert m_formal.formality_score > m_informal.formality_score

    def test_role_keywords_adherence(self) -> None:
        evaluator = BehavioralEvaluator()
        text = "The clipper ship logged strong squalls along the southern maritime passage."
        keywords = ["clipper", "maritime", "squalls"]

        m = evaluator.evaluate_text(text, role_keywords=keywords)
        assert m.role_adherence_score == 1.0

    def test_evaluate_batch(self) -> None:
        evaluator = BehavioralEvaluator()
        texts = ["Text one.", "Text two with more words."]
        batch_res = evaluator.evaluate_batch(texts)
        assert len(batch_res) == 2
        assert batch_res[0].word_count == 2
        assert batch_res[1].word_count == 5

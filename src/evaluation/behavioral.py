"""
Behavioral evaluation — modular measurement of benign behavioral and stylistic shifts.

Measures:
    - Persona / role markers (e.g. self-referential phrases, assistant disclaimers, domain vocabulary).
    - Stylistic metrics (formality, first-person frequency, sentence length, vocabulary complexity).
    - Response length and lexical statistics.
    - Extensible interface for optional external LLM judge scoring with recorded metadata.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class BehavioralMetrics:
    """Quantitative behavioral and stylistic measurements for a generated response."""

    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    first_person_count: int = 0
    first_person_rate: float = 0.0  # per 100 words
    assistant_disclaimer_count: int = 0
    role_adherence_score: float = 0.0  # Normalized [0, 1] heuristic
    formality_score: float = 0.0       # Normalized [0, 1] heuristic
    custom_markers: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehavioralMetrics":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class BehavioralEvaluator:
    """Deterministic rubric-based evaluator for generated text."""

    FIRST_PERSON_PATTERNS = [
        r"\bi\b", r"\bmy\b", r"\bme\b", r"\bmine\b", r"\bmyself\b",
    ]

    ASSISTANT_DISCLAIMERS = [
        r"as an ai", r"as a language model", r"i am an ai", r"as an assistant",
        r"how can i assist you", r"how may i help",
    ]

    FORMAL_MARKERS = [
        r"\bfurthermore\b", r"\bmoreover\b", r"\bconsequently\b", r"\btherefore\b",
        r"\bthus\b", r"\bhence\b", r"\bnotably\b", r"\bspecifically\b",
    ]

    CONTRACTION_PATTERNS = [
        r"\bcan't\b", r"\bdon't\b", r"\bwon't\b", r"\bit's\b", r"\bi'm\b",
        r"\byou're\b", r"\bwe're\b", r"\bthey're\b", r"\bdidn't\b",
    ]

    def evaluate_text(
        self,
        text: str,
        role_keywords: Optional[Sequence[str]] = None,
    ) -> BehavioralMetrics:
        """Compute behavioral and stylistic metrics for a single text response.

        Parameters
        ----------
        text:
            The generated response string.
        role_keywords:
            Optional sequence of domain or character-specific keywords to measure role adherence.

        Returns
        -------
        BehavioralMetrics
        """
        if not text or not text.strip():
            return BehavioralMetrics()

        clean_text = text.strip()
        lower_text = clean_text.lower()
        words = re.findall(r"\b\w+\b", clean_text)
        word_count = len(words)
        char_count = len(clean_text)

        sentences = re.split(r"[.!?]+", clean_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(1, len(sentences))

        avg_word_length = (
            sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
        )

        # First-person pronouns count
        fp_count = 0
        for pat in self.FIRST_PERSON_PATTERNS:
            fp_count += len(re.findall(pat, lower_text))
        fp_rate = (fp_count / word_count * 100.0) if word_count > 0 else 0.0

        # Assistant disclaimer occurrences
        disc_count = 0
        for pat in self.ASSISTANT_DISCLAIMERS:
            disc_count += len(re.findall(pat, lower_text))

        # Formality heuristic: formal markers boost, contractions penalize
        formal_hits = sum(len(re.findall(pat, lower_text)) for pat in self.FORMAL_MARKERS)
        contraction_hits = sum(len(re.findall(pat, lower_text)) for pat in self.CONTRACTION_PATTERNS)
        formality_score = float(np_clip(
            0.5 + 0.1 * formal_hits - 0.1 * contraction_hits, 0.0, 1.0
        ))

        # Role adherence heuristic
        role_hits = 0
        if role_keywords:
            for kw in role_keywords:
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower_text):
                    role_hits += 1
            role_adherence_score = float(np_clip(role_hits / max(1, len(role_keywords)), 0.0, 1.0))
        else:
            role_adherence_score = 1.0 if disc_count == 0 else 0.0

        return BehavioralMetrics(
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
            avg_word_length=round(avg_word_length, 2),
            first_person_count=fp_count,
            first_person_rate=round(fp_rate, 2),
            assistant_disclaimer_count=disc_count,
            role_adherence_score=round(role_adherence_score, 2),
            formality_score=round(formality_score, 2),
        )

    def evaluate_batch(
        self,
        texts: Sequence[str],
        role_keywords: Optional[Sequence[str]] = None,
    ) -> List[BehavioralMetrics]:
        """Evaluate a batch of generated text responses."""
        return [self.evaluate_text(t, role_keywords=role_keywords) for t in texts]


def np_clip(val: float, min_val: float, max_val: float) -> float:
    """Helper clip function without requiring numpy in simple string tests."""
    return max(min_val, min(val, max_val))

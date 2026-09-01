"""
Prompt schemas — data structures for representing experimental prompts.

These schemas define the *structure* of prompts, not their content.
Actual prompt data lives in ``data/prompts/``.

Key design choice: prompts are plain dataclasses, not ORM models or
framework-specific objects.  This keeps the research code portable and
easy to serialise to/from YAML/JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptItem:
    """A single prompt used in an experiment.

    Attributes
    ----------
    prompt_id:
        Unique identifier for this prompt.  Required.
    base_content:
        The core semantic content shared across experimental conditions.
    condition:
        The experimental condition label (e.g. ``"baseline"``,
        ``"provenance_system"``).
    provenance:
        Description or label of the provenance framing applied.
        ``None`` for control/baseline conditions.
    role:
        The chat role this content occupies (e.g. ``"user"``,
        ``"system"``, ``"assistant"``).
    turn:
        Turn index within a multi-turn conversation (0-indexed).
    conversation_id:
        Groups prompts that belong to the same multi-turn conversation.
    metadata:
        Arbitrary additional key-value pairs for future extensions.
    """

    prompt_id: str = ""
    base_content: str = ""
    condition: str = ""
    provenance: Optional[str] = None
    role: str = "user"
    turn: int = 0
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id is required and must be non-empty.")
        if not self.base_content:
            raise ValueError("base_content is required and must be non-empty.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "base_content": self.base_content,
            "condition": self.condition,
            "provenance": self.provenance,
            "role": self.role,
            "turn": self.turn,
            "conversation_id": self.conversation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptItem":
        """Construct a ``PromptItem`` from a dictionary."""
        return cls(
            prompt_id=data.get("prompt_id", ""),
            base_content=data.get("base_content", ""),
            condition=data.get("condition", ""),
            provenance=data.get("provenance"),
            role=data.get("role", "user"),
            turn=data.get("turn", 0),
            conversation_id=data.get("conversation_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PromptSet:
    """An ordered collection of prompts for one experimental condition.

    Attributes
    ----------
    set_id:
        Unique identifier for this prompt set.
    description:
        Human-readable description.
    prompts:
        The prompts in this set.
    """

    set_id: str = ""
    description: str = ""
    prompts: List[PromptItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.set_id:
            raise ValueError("set_id is required and must be non-empty.")

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self):
        return iter(self.prompts)

    @property
    def prompt_ids(self) -> List[str]:
        return [p.prompt_id for p in self.prompts]

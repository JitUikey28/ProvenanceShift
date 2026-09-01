"""
Evaluation result schemas.

Defines data structures for storing evaluation/metric results.
Actual metrics and analysis logic will be implemented later.

IMPORTANT: A change in model output is NOT automatically "persona drift".
The evaluation framework will eventually need to distinguish:
    - Behavioral difference (surface-level output change)
    - Representation difference (internal activation change)
    - Persona-axis projection (directional change in a defined subspace)
    - Statistical significance
    - Effect size

These schemas lay the groundwork but do not implement metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationResult:
    """Container for a single evaluation measurement.

    Attributes
    ----------
    experiment_id:
        The experiment that produced the data being evaluated.
    metric_name:
        Name of the metric (e.g. ``"refusal_rate"``,
        ``"cosine_similarity"``, ``"persona_projection"``).
    metric_value:
        The computed metric value.
    condition:
        Experimental condition label.
    comparison_condition:
        The baseline or reference condition, if this is a comparative
        metric.
    details:
        Arbitrary additional information (e.g. per-prompt breakdowns,
        confidence intervals).
    """

    experiment_id: str = ""
    metric_name: str = ""
    metric_value: Optional[float] = None
    condition: Optional[str] = None
    comparison_condition: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required.")
        if not self.metric_name:
            raise ValueError("metric_name is required.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

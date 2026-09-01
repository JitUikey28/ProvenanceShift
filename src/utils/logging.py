"""
Lightweight structured logging for ProvenanceShift experiments.

Design goals:
    - Every log line includes a timestamp.
    - Experiment ID is attached when available.
    - Standard Python logging — no custom framework.
    - Console + optional file output.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_LOGGER_NAME = "provenance_shift"
_INITIALISED = False


class _ExperimentFormatter(logging.Formatter):
    """Formatter that prepends experiment ID when set on the logger."""

    def __init__(self, experiment_id: Optional[str] = None) -> None:
        super().__init__()
        self.experiment_id = experiment_id

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        exp_tag = f"[{self.experiment_id}] " if self.experiment_id else ""
        return f"{ts} [{record.levelname}] {exp_tag}{record.getMessage()}"


def get_logger(
    experiment_id: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Return the project-wide logger, initialising it on first call.

    Parameters
    ----------
    experiment_id:
        If provided, every log message will be tagged with this ID.
    level:
        Logging level (default ``logging.INFO``).
    log_file:
        If provided, logs are also written to this file.

    Returns
    -------
    logging.Logger
    """
    global _INITIALISED

    logger = logging.getLogger(_LOGGER_NAME)

    if _INITIALISED:
        # Update experiment ID on existing handlers if requested.
        if experiment_id is not None:
            for handler in logger.handlers:
                if isinstance(handler.formatter, _ExperimentFormatter):
                    handler.formatter.experiment_id = experiment_id
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = _ExperimentFormatter(experiment_id=experiment_id)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _INITIALISED = True
    return logger

"""Sentiment scorers behind one interface: ``score(texts) -> DataFrame`` with ``sentiment``, ``compound``, ``status``."""

from .base import LABELS, SCORERS, build_scorer

__all__ = ["LABELS", "SCORERS", "build_scorer"]

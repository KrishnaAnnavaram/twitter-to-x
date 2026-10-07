"""The scorer interface and the factory.

A scorer returns one row per text with:

* ``sentiment`` - ``negative``, ``neutral`` or ``positive`` (or ``None`` when ``status`` is ``error``)
* ``compound``  - a score in [-1, 1]
* ``status``    - ``ok`` or ``error``. Errors are COUNTED and EXCLUDED from the statistics. They are
  never relabelled as a fourth sentiment class (the prototype turned errors into "NEUTRAL").
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

LABELS = ("negative", "neutral", "positive")
SCORERS = ("lexicon", "vader", "transformer")


class Scorer(Protocol):
    name: str

    def score(self, texts: list[str]) -> pd.DataFrame: ...


def build_scorer(name: str, **options) -> Scorer:
    if name == "lexicon":
        from .lexicon import LexiconScorer

        return LexiconScorer()
    if name == "vader":
        from .lexicon import VaderScorer

        return VaderScorer()
    if name == "transformer":
        from .transformer import TransformerScorer

        return TransformerScorer(**options)
    raise ValueError(f"unknown scorer {name!r}; choose from {SCORERS}")


def label_from_compound(compound: float, threshold: float = 0.05) -> str:
    if compound >= threshold:
        return "positive"
    if compound <= -threshold:
        return "negative"
    return "neutral"

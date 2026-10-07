"""Validate a moderation detector against a labelled set: precision, recall, F1 with Wilson intervals.

``gold.csv`` (shipped with the package) has 60 hand-written sentences, 28 about moderation and 32 hard
negatives ("urban", "banana", "skill", "the bus was suspended"). It is a smoke check of the patterns.
For a study, label a random sample of your own corpus and pass it with ``--gold``.
"""

from __future__ import annotations

import math
from importlib import resources
from pathlib import Path
from typing import Callable

import pandas as pd


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_gold(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        with resources.files("twitter_to_x.moderation").joinpath("gold.csv").open(encoding="utf-8") as fh:
            df = pd.read_csv(fh)
    else:
        df = pd.read_csv(path)
    if not {"text", "label"} <= set(df.columns) or not df["label"].isin([0, 1]).all():
        raise ValueError("the gold file needs columns text and label (0 or 1)")
    return df


def evaluate_flags(gold: pd.DataFrame, flag: Callable[[str], bool]) -> dict:
    pred = gold["text"].map(flag).astype(int)
    y = gold["label"].astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"n": len(gold), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "precision_ci95": wilson(tp, tp + fp), "recall_ci95": wilson(tp, tp + fn),
            "false_positives": gold.loc[(pred == 1) & (y == 0), "text"].tolist(),
            "false_negatives": gold.loc[(pred == 0) & (y == 1), "text"].tolist()}

"""Random, comparable samples. Never a head or tail slice.

The prototype kept ``iloc[-300000:]`` of Sentiment140. That file is sorted by label, so the "Twitter"
sample was all positive. ``sample_period`` draws at random (optionally stratified), and
``sortedness`` measures how much a column is sorted, so a slice of a sorted file can be detected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sortedness(values) -> float:
    """Share of neighbour pairs with the same value. About ``sum(p_k^2)`` for a shuffled column, near 1 if sorted."""
    arr = np.asarray(list(values), dtype=object)
    if len(arr) < 2:
        return 0.0
    return float(np.mean(arr[1:] == arr[:-1]))


def sample_period(df: pd.DataFrame, n: int, seed: int, stratify: str | None = None) -> pd.DataFrame:
    """A random sample of ``n`` rows without replacement. With ``stratify``, each stratum keeps its share."""
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if stratify is None:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    shares = df[stratify].value_counts(normalize=True)
    parts = []
    for value, share in shares.items():
        group = df[df[stratify] == value]
        parts.append(group.sample(n=min(len(group), int(round(share * n))), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def balanced_periods(df: pd.DataFrame, n_per_period: int, seed: int, stratify: str | None = None) -> pd.DataFrame:
    """The same sample size for ``pre`` and ``post`` (the smaller period sets the size if it is smaller)."""
    sizes = df["period"].value_counts()
    if set(sizes.index) != {"pre", "post"}:
        raise ValueError(f"both periods are needed, found {sorted(sizes.index)}")
    n = min(n_per_period, int(sizes.min()))
    parts = [sample_period(df[df["period"] == p], n, seed, stratify) for p in ("pre", "post")]
    return pd.concat(parts, ignore_index=True)

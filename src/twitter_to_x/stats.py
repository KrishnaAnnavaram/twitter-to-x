"""Statistics on COUNTS, with effect sizes and intervals.

The prototype passed a table of proportions to ``chi2_contingency``. A chi-square test needs counts:
with proportions the statistic and the p-value have no meaning. ``chi_square`` refuses a table that is
not made of non-negative integers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


class NotCountsError(ValueError):
    """Raised when a contingency table holds proportions or other non-count values."""


def contingency(df: pd.DataFrame, row: str, col: str, col_order=None) -> pd.DataFrame:
    table = pd.crosstab(df[row], df[col])
    if col_order is not None:
        table = table.reindex(columns=list(col_order), fill_value=0)
    return table.astype(int)


def chi_square(table: pd.DataFrame) -> dict:
    values = np.asarray(table, dtype=float)
    if (values < 0).any() or not np.allclose(values, np.round(values)):
        raise NotCountsError("the chi-square test needs a table of counts, not proportions")
    if values.sum() < 1 or min(values.shape) < 2:
        raise NotCountsError("the table needs at least two rows, two columns and one count")
    values = values[:, values.sum(axis=0) > 0]
    chi2, p, dof, expected = stats.chi2_contingency(values, correction=False)
    n = values.sum()
    v = math.sqrt(chi2 / (n * (min(values.shape) - 1))) if n else float("nan")
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": v, "n": int(n),
            "min_expected": float(expected.min())}


def proportion_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def diff_ci(x1: int, n1: int, x2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """Difference p2 - p1 with a Newcombe (Wilson score) interval."""
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = proportion_ci(x1, n1, z)
    l2, u2 = proportion_ci(x2, n2, z)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return d, lo, hi


def standardized_share_diff(df: pd.DataFrame, outcome: str, stratum: str, n_boot: int = 500, seed: int = 0) -> dict:
    """Post-minus-pre difference of ``outcome`` (0/1) after direct standardization on ``stratum``.

    Each period's stratum-specific rate is weighted by the POOLED stratum distribution, so a change in
    topic mix does not show up as a change in sentiment.
    """
    def estimate(d: pd.DataFrame) -> float:
        weights = d[stratum].value_counts(normalize=True)
        rates = d.groupby(["period", stratum])[outcome].mean()
        total = 0.0
        for s, w in weights.items():
            if ("pre", s) in rates and ("post", s) in rates:
                total += w * (rates[("post", s)] - rates[("pre", s)])
        covered = sum(w for s, w in weights.items() if ("pre", s) in rates and ("post", s) in rates)
        return total / covered if covered else float("nan")

    point = estimate(df)
    rng = np.random.default_rng(seed)
    parts = {p: df[df["period"] == p] for p in ("pre", "post")}
    draws = []
    for _ in range(n_boot):
        sample = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in parts.values()])
        draws.append(estimate(sample))
    lo, hi = np.nanquantile(draws, [0.025, 0.975])
    crude = df[df.period == "post"][outcome].mean() - df[df.period == "pre"][outcome].mean()
    return {"crude_diff": float(crude), "standardized_diff": float(point), "ci95": [float(lo), float(hi)]}


def logistic_regression(y: np.ndarray, x: pd.DataFrame, max_iter: int = 50, ridge: float = 1e-6) -> pd.DataFrame:
    """Maximum-likelihood logistic regression (IRLS) with Wald standard errors and odds ratios.

    ``x`` must hold the predictors without an intercept. A tiny ridge keeps the solve stable.
    """
    names = ["intercept", *x.columns]
    xm = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
    y = np.asarray(y, dtype=float)
    beta = np.zeros(xm.shape[1])
    for _ in range(max_iter):
        eta = xm @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        hess = xm.T @ (xm * w[:, None]) + ridge * np.eye(len(beta))
        step = np.linalg.solve(hess, xm.T @ (y - mu) - ridge * beta)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = 1 / (1 + np.exp(-(xm @ beta)))
    cov = np.linalg.inv(xm.T @ (xm * (mu * (1 - mu))[:, None]) + ridge * np.eye(len(beta)))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    return pd.DataFrame({
        "term": names, "coef": beta, "se": se, "z": z, "p_value": 2 * stats.norm.sf(np.abs(z)),
        "odds_ratio": np.exp(beta), "or_ci_low": np.exp(beta - 1.96 * se), "or_ci_high": np.exp(beta + 1.96 * se),
    })


def standardized_mean_difference(a, b) -> float:
    """SMD between two groups (pooled SD). |SMD| > 0.1 is a common sign of imbalance."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else 0.0
    return float((b.mean() - a.mean()) / sd) if sd else 0.0

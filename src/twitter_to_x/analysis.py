"""The study pipeline: prepare -> score -> analyze -> report.

Both periods go through exactly the same functions, one time each.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .data.filters import apply_filters
from .data.sampling import balanced_periods, sortedness
from .data.sources import assign_period, validate_posts
from .moderation import ModerationDetector
from .sentiment.base import LABELS
from .stats import (
    chi_square,
    contingency,
    diff_ci,
    logistic_regression,
    proportion_ci,
    standardized_mean_difference,
    standardized_share_diff,
)
from .text import light_clean
from .topics import TopicModel, prevalence


def prepare(posts: pd.DataFrame, cutoff: date, n_per_period: int, seed: int) -> tuple[pd.DataFrame, dict]:
    df = assign_period(validate_posts(posts), cutoff)
    df, filt = apply_filters(df)
    df = balanced_periods(df, n_per_period, seed)
    df["text"] = df["text"].map(light_clean)
    manifest = {
        "cutoff": cutoff.isoformat(), "seed": seed, "n_per_period": int(df["period"].value_counts().min()),
        "rows_per_period": df["period"].value_counts().to_dict(),
        "sources_per_period": {p: g["source"].value_counts().to_dict() for p, g in df.groupby("period")},
        "date_range": {p: [str(g["created_at"].min()), str(g["created_at"].max())] for p, g in df.groupby("period")},
        "filters": {"rows_in": filt.rows_in, "removed": filt.removed, "rows_out": filt.rows_out},
    }
    if "source_label" in df.columns:
        manifest["source_label_sortedness_in_sample"] = sortedness(df["source_label"])
    return df, manifest


def score(df: pd.DataFrame, scorer, detector: ModerationDetector | None = None) -> pd.DataFrame:
    detector = detector or ModerationDetector()
    out = df.reset_index(drop=True).copy()
    scored = scorer.score(out["text"].tolist())
    out["sentiment"], out["compound"], out["status"] = scored["sentiment"], scored["compound"], scored["status"]
    matches = [detector.match(t) for t in out["text"]]
    out["moderation"] = [int(m.flagged) for m in matches]
    out["moderation_categories"] = [";".join(m.categories) for m in matches]
    out["scorer"] = scorer.name
    return out


def _shares(df: pd.DataFrame) -> dict:
    out = {}
    for period, g in df.groupby("period"):
        n = len(g)
        out[period] = {lab: {"count": int((g["sentiment"] == lab).sum()),
                             "share": round(float((g["sentiment"] == lab).mean()), 4),
                             "ci95": [round(x, 4) for x in proportion_ci(int((g["sentiment"] == lab).sum()), n)]}
                       for lab in LABELS}
    return out


def _confounders(df: pd.DataFrame) -> dict:
    feats = {
        "length_words": df["text"].str.split().str.len(),
        "has_link": df["text"].str.contains("http", regex=False).astype(int),
        "has_mention": df["text"].str.contains("@user", regex=False).astype(int),
        "hashtags": df["text"].str.count("#"),
    }
    pre, post = df["period"] == "pre", df["period"] == "post"
    return {k: round(standardized_mean_difference(v[pre], v[post]), 4) for k, v in feats.items()}


def analyze(scored: pd.DataFrame, n_topics: int = 8, seed: int = 7, n_boot: int = 500) -> dict:
    errors = scored[scored["status"] != "ok"]
    df = scored[scored["status"] == "ok"].copy()
    df["negative"] = (df["sentiment"] == "negative").astype(int)
    result: dict = {
        "scorer": str(scored["scorer"].iloc[0]),
        "posts": df["period"].value_counts().to_dict(),
        "scorer_errors_excluded": errors["period"].value_counts().to_dict(),
        "sentiment_shares": _shares(df),
        "sentiment_chi_square": chi_square(contingency(df, "period", "sentiment", LABELS)),
        "confounders_smd": _confounders(df),
    }
    n = df.groupby("period")["moderation"].agg(["sum", "count"])
    d, lo, hi = diff_ci(int(n.loc["pre", "sum"]), int(n.loc["pre", "count"]), int(n.loc["post", "sum"]),
                        int(n.loc["post", "count"]))
    result["moderation_share"] = {"pre": round(n.loc["pre", "sum"] / n.loc["pre", "count"], 4),
                                  "post": round(n.loc["post", "sum"] / n.loc["post", "count"], 4),
                                  "diff": round(d, 4), "diff_ci95": [round(lo, 4), round(hi, 4)]}
    mod = df[df["moderation"] == 1]
    if mod["period"].nunique() == 2:
        result["moderation_sentiment_chi_square"] = chi_square(contingency(mod, "period", "sentiment", LABELS))
        m = mod.groupby("period")["negative"].agg(["sum", "count"])
        d, lo, hi = diff_ci(int(m.loc["pre", "sum"]), int(m.loc["pre", "count"]), int(m.loc["post", "sum"]),
                            int(m.loc["post", "count"]))
        result["moderation_negative_share"] = {"pre": round(m.loc["pre", "sum"] / m.loc["pre", "count"], 4),
                                               "post": round(m.loc["post", "sum"] / m.loc["post", "count"], 4),
                                               "diff": round(d, 4), "diff_ci95": [round(lo, 4), round(hi, 4)]}

    tm = TopicModel(n_topics=n_topics, seed=seed).fit(df["text"].tolist())
    df["topic"] = tm.assign()
    result["topics"] = tm.describe()
    result["topic_prevalence"] = prevalence(df["topic"].to_numpy(), df["period"], n_boot=n_boot,
                                            seed=seed).to_dict(orient="records")
    std = standardized_share_diff(df, "negative", "topic", n_boot=n_boot, seed=seed)
    result["negative_share_topic_adjusted"] = {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v])
                                               for k, v in std.items()}

    design = pd.DataFrame({"post": (df["period"] == "post").astype(int), "moderation": df["moderation"]})
    design["post_x_moderation"] = design["post"] * design["moderation"]
    topics = pd.get_dummies(df["topic"].astype(str), prefix="topic", drop_first=True, dtype=int)
    reg = logistic_regression(df["negative"].to_numpy(), pd.concat([design, topics], axis=1))
    result["regression_negative"] = reg[~reg["term"].str.startswith("topic_")].round(4).to_dict(orient="records")

    if "true_moderation" in df.columns:
        t = pd.to_numeric(df["true_moderation"])
        tp = int(((df["moderation"] == 1) & (t == 1)).sum())
        result["check_against_truth"] = {
            "moderation_precision": round(tp / max(1, int(df["moderation"].sum())), 4),
            "moderation_recall": round(tp / max(1, int(t.sum())), 4),
            "sentiment_accuracy": round(float((df["sentiment"] == df["true_sentiment"]).mean()), 4),
        }
    return result


def to_markdown(r: dict) -> str:
    lines = [f"# Twitter vs X study report (scorer: `{r['scorer']}`)", "",
             f"Posts analyzed: {r['posts']}. Scorer errors excluded: {r['scorer_errors_excluded'] or 'none'}.", "",
             "## Sentiment by period", "", "| period | negative | neutral | positive |", "|---|---|---|---|"]
    for p in ("pre", "post"):
        s = r["sentiment_shares"][p]
        lines.append(f"| {p} | " + " | ".join(f"{s[k]['share']:.3f} ({s[k]['ci95'][0]:.3f}-{s[k]['ci95'][1]:.3f})"
                                             for k in LABELS) + " |")
    c = r["sentiment_chi_square"]
    lines += ["", f"Chi-square on counts: chi2 = {c['chi2']:.2f}, dof = {c['dof']}, p = {c['p_value']:.3g}, "
                  f"Cramer's V = {c['cramers_v']:.3f}.", ""]
    adj = r["negative_share_topic_adjusted"]
    lines += [f"Negative share, post minus pre: crude {adj['crude_diff']:+.3f}, topic-adjusted "
              f"{adj['standardized_diff']:+.3f} (95 % CI {adj['ci95'][0]:+.3f} to {adj['ci95'][1]:+.3f}).", ""]
    m = r["moderation_share"]
    lines += ["## Moderation talk", "", f"Share of posts: pre {m['pre']:.3f}, post {m['post']:.3f}, difference "
              f"{m['diff']:+.3f} (95 % CI {m['diff_ci95'][0]:+.3f} to {m['diff_ci95'][1]:+.3f}).", ""]
    if "moderation_negative_share" in r:
        mn = r["moderation_negative_share"]
        lines.append(f"Negative share inside moderation posts: pre {mn['pre']:.3f}, post {mn['post']:.3f}, difference "
                     f"{mn['diff']:+.3f} (95 % CI {mn['diff_ci95'][0]:+.3f} to {mn['diff_ci95'][1]:+.3f}).")
    lines += ["", "## Logistic regression: negative ~ post * moderation + topic", "",
              "| term | odds ratio | 95 % CI | p |", "|---|---|---|---|"]
    for row in r["regression_negative"]:
        lines.append(f"| {row['term']} | {row['odds_ratio']:.3f} | {row['or_ci_low']:.3f}-{row['or_ci_high']:.3f} | "
                     f"{row['p_value']:.3g} |")
    lines += ["", "## Topics (one model for both periods)", "", "| topic | words | pre | post | diff (95 % CI) |",
              "|---|---|---|---|---|"]
    words = {t["topic"]: t["label"] for t in r["topics"]}
    for row in r["topic_prevalence"]:
        lines.append(f"| {row['topic']} | {words.get(row['topic'], '(none)')} | {row['pre_share']:.3f} | "
                     f"{row['post_share']:.3f} | {row['diff']:+.3f} ({row['diff_ci95'][0]:+.3f} to "
                     f"{row['diff_ci95'][1]:+.3f}) |")
    lines += ["", "## Confounder check (standardized mean difference, post vs pre)", "",
              ", ".join(f"{k}: {v:+.3f}" for k, v in r["confounders_smd"].items())]
    if "check_against_truth" in r:
        lines += ["", "## Check against the synthetic truth", "", json.dumps(r["check_against_truth"])]
    return "\n".join(lines) + "\n"


def write_report(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(result), encoding="utf-8")
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))

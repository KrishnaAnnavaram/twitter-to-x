"""Loaders. Each returns the post schema: ``post_id, text, created_at, source, author_hash``.

Raw user names are hashed in the loader and then dropped. Handles in the text are masked.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .anonymize import hash_author, mask_text

COLUMNS = ["post_id", "text", "created_at", "source", "author_hash"]


class SchemaError(ValueError):
    """Raised when a frame does not match the post schema."""


def _frame(source: str, ids, texts, times, authors, salt: str, extra: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame({
        "post_id": [f"{source}:{i}" for i in ids],
        "text": [mask_text(t) for t in texts],
        "created_at": pd.to_datetime(pd.Series(list(times)), utc=True, errors="coerce", format="mixed"),
        "source": source,
        "author_hash": [hash_author(a, salt) for a in authors],
    })
    for k, v in (extra or {}).items():
        df[k] = list(v)
    return df


def validate_posts(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    out = df.copy()
    out = out[out["text"].astype(str).str.strip() != ""]
    if out["created_at"].isna().any():
        raise SchemaError(f"{int(out['created_at'].isna().sum())} posts have no valid timestamp")
    if out["post_id"].duplicated().any():
        raise SchemaError("post ids must be unique")
    return out.reset_index(drop=True)


def load_sentiment140(path: str | Path, salt: str) -> pd.DataFrame:
    """Sentiment140 (2009). The file is SORTED by label: never slice it, sample it (see ``sampling``)."""
    raw = pd.read_csv(path, header=None, names=["target", "id", "date", "flag", "user", "text"], dtype=str,
                      encoding="latin-1")
    # "Mon Apr 06 22:19:45 PDT 2009": the zone name is not parseable, PDT is UTC-7
    stamps = pd.to_datetime(raw["date"].str.replace(r" [A-Z]{3} ", " ", regex=True), format="%a %b %d %H:%M:%S %Y",
                            errors="coerce") + pd.Timedelta(hours=7)
    return _frame("sentiment140", raw["id"], raw["text"], stamps, raw["user"], salt,
                  {"source_label": raw["target"].map({"0": "negative", "2": "neutral", "4": "positive"})})


def load_x_parquet(path: str | Path, salt: str) -> pd.DataFrame:
    """A public X dataset chunk (parquet, needs the ``parquet`` extra): columns ``text``, ``datetime`` and an
    encoded user column (``username_encoded`` or ``username``)."""
    raw = pd.read_parquet(path)
    user_col = next((c for c in ("username_encoded", "username", "user") if c in raw.columns), None)
    authors = raw[user_col] if user_col else [""] * len(raw)
    ids = raw["uri"] if "uri" in raw.columns else range(len(raw))
    return _frame(f"x_{Path(path).stem}", ids, raw["text"], raw["datetime"], authors, salt)


def load_csv(path: str | Path, salt: str, source: str | None = None) -> pd.DataFrame:
    """A CSV with ``text`` and ``created_at`` and optional ``author`` and ``post_id`` columns."""
    raw = pd.read_csv(path, dtype=str)
    for col in ("text", "created_at"):
        if col not in raw.columns:
            raise SchemaError(f"{path} needs a {col!r} column")
    src = source or Path(path).stem
    ids = raw["post_id"] if "post_id" in raw.columns else range(len(raw))
    authors = raw["author"] if "author" in raw.columns else [""] * len(raw)
    extra = {c: raw[c] for c in raw.columns if c.startswith("true_")}
    return _frame(src, ids, raw["text"], raw["created_at"], authors, salt, extra)


def assign_period(df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """``pre`` before the cutoff day, ``post`` on or after it (UTC)."""
    out = df.copy()
    edge = pd.Timestamp(cutoff, tz="UTC")
    out["period"] = (out["created_at"] >= edge).map({True: "post", False: "pre"})
    return out


LOADERS = {"sentiment140": load_sentiment140, "x_parquet": load_x_parquet, "csv": load_csv}


def load_sources(specs: list[str], salt: str) -> pd.DataFrame:
    frames = []
    for spec in specs:
        kind, _, path = spec.partition("=")
        if kind not in LOADERS or not path:
            raise ValueError(f"source must be KIND=PATH with KIND in {sorted(LOADERS)}, got {spec!r}")
        frames.append(LOADERS[kind](path, salt))
    return validate_posts(pd.concat(frames, ignore_index=True))

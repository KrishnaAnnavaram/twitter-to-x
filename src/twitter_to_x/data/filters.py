"""Documented, scripted filters. Each rule has a name, and the report counts the posts that each rule removes.

The prototype removed spam by hand with no record. Here every removal is a rule in code, applied in
the same way to both periods.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

_WORD = re.compile(r"[A-Za-z']+")
_ENGLISH_HINTS = {"the", "a", "an", "is", "are", "was", "to", "and", "of", "in", "it", "i", "you", "this", "that",
                  "my", "for", "on", "with", "not", "be", "have", "just", "so", "me", "we", "they", "what", "no"}
SPAM_PATTERNS = [
    re.compile(r"\b(?:free|win)\b.{0,30}\b(?:crypto|bitcoin|nft|airdrop|giveaway|followers)\b", re.I),
    re.compile(r"\bfollow\s+(?:me|back)\b.{0,40}\b(?:follow|dm|retweet)\b", re.I),
    re.compile(r"\b(?:dm|click)\b.{0,20}\b(?:link|bio)\b.{0,30}\b(?:earn|profit|\$\d+)", re.I),
]


@dataclass
class FilterReport:
    rows_in: int = 0
    removed: dict[str, dict[str, int]] = field(default_factory=dict)  # rule -> period -> count
    rows_out: int = 0

    def add(self, rule: str, dropped: pd.DataFrame) -> None:
        if len(dropped):
            by_period = dropped["period"].value_counts().to_dict() if "period" in dropped else {"all": len(dropped)}
            self.removed[rule] = {k: int(v) for k, v in by_period.items()}


def is_english(text: str, min_hits: int = 1) -> bool:
    """A light English check: at least ``min_hits`` common English words and mostly Latin letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    latin = sum(c.isascii() for c in letters) / len(letters)
    words = [w.lower() for w in _WORD.findall(text)]
    return latin >= 0.8 and sum(w in _ENGLISH_HINTS for w in words) >= min_hits


def is_spam(text: str) -> bool:
    if any(p.search(text) for p in SPAM_PATTERNS):
        return True
    return text.count("#") >= 6 or text.lower().count("http") >= 3


def apply_filters(df: pd.DataFrame, max_posts_per_author_text: int = 3) -> tuple[pd.DataFrame, FilterReport]:
    """Rules, in order: empty, duplicate text, non-English, spam pattern, repeated author text (bots)."""
    rep = FilterReport(rows_in=len(df))
    work = df.copy()
    norm = work["text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()

    mask = norm == ""
    rep.add("empty", work[mask])
    work, norm = work[~mask], norm[~mask]

    # the same text by the same author many times is a bot signal: drop the author's copies
    if "author_hash" in work:
        key = work["author_hash"].where(work["author_hash"] != "", "anon") + "|" + norm
        counts = key.map(key.value_counts())
        bot = (counts > max_posts_per_author_text) & (work["author_hash"] != "")
        rep.add("repeated_author_text", work[bot])
        work, norm = work[~bot], norm[~bot]

    dup = norm.duplicated(keep="first")
    rep.add("duplicate_text", work[dup])
    work, norm = work[~dup], norm[~dup]

    eng = work["text"].map(is_english)
    rep.add("not_english", work[~eng])
    work = work[eng]

    spam = work["text"].map(is_spam)
    rep.add("spam_pattern", work[spam])
    work = work[~spam]

    rep.rows_out = len(work)
    return work.reset_index(drop=True), rep

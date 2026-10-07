"""A small rule-based scorer in the style of VADER (offline, no dependency), and an adapter for VADER itself.

Rules of ``LexiconScorer``:

1. Each word in ``VALENCE`` adds its valence.
2. A negator in the three words before a word multiplies its valence by -0.74.
3. An intensifier right before a word multiplies it by 1.3. A word in capitals (in a mixed-case text) by 1.5.
4. After "but", words count 1.5 times and words before it count 0.5 times.
5. Each "!" (up to 3) adds 0.29 in the direction of the sum.
6. ``compound = s / sqrt(s^2 + 15)``. The label threshold is 0.05.

It reads the LIGHT-cleaned text, so case, "!" and negations are still there.
"""

from __future__ import annotations

import math
import re

import pandas as pd

from .base import label_from_compound

VALENCE: dict[str, float] = {
    "good": 1.9, "great": 3.1, "excellent": 3.2, "amazing": 2.8, "awesome": 3.1, "love": 3.2, "loved": 2.9,
    "like": 1.5, "nice": 1.8, "happy": 2.7, "glad": 2.0, "fun": 2.3, "best": 3.2, "better": 1.9, "fair": 1.3,
    "win": 2.8, "won": 2.7, "fixed": 1.2, "helpful": 1.8, "useful": 1.9, "fantastic": 2.6, "enjoy": 2.2,
    "thanks": 1.9, "thank": 1.5, "beautiful": 2.9, "free": 1.0, "safe": 1.9, "improved": 1.9, "accurate": 1.6,
    "bad": -2.5, "terrible": -2.1, "awful": -2.0, "hate": -2.7, "hated": -3.2, "worst": -3.1, "worse": -2.1,
    "sad": -2.1, "angry": -2.3, "broken": -1.9, "unfair": -2.1, "lost": -1.3, "lose": -1.7, "boring": -1.3,
    "annoying": -1.8, "dead": -3.3, "scary": -2.2, "toxic": -2.4, "fake": -2.1, "wrong": -2.1, "ugly": -2.3,
    "suspended": -1.5, "banned": -1.8, "censored": -1.9, "censorship": -1.6, "abuse": -3.2, "harassment": -2.6,
    "spam": -1.5, "scam": -2.5, "lies": -1.8, "misinformation": -1.6, "disappointed": -1.9, "disappointing": -2.2,
    "mess": -1.5, "chaos": -1.9, "fail": -2.5, "failed": -2.3, "slow": -1.0, "crashed": -1.7, "missing": -1.2,
}
NEGATORS = {"not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "without", "cannot",
            "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "can't", "won't", "ain't",
            "dont", "doesnt", "didnt", "isnt", "cant", "wont"}
INTENSIFIERS = {"very", "so", "really", "extremely", "totally", "super", "absolutely", "incredibly", "too"}
_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*|!")


class LexiconScorer:
    name = "lexicon"

    def compound(self, text: str) -> float:
        tokens = _TOKEN.findall(str(text))
        words = [t for t in tokens if t != "!"]
        mixed_case = any(w.islower() for w in words) and any(w.isupper() and len(w) > 1 for w in words)
        lowered = [w.lower() for w in words]
        but_at = lowered.index("but") if "but" in lowered else None
        total = 0.0
        for i, w in enumerate(lowered):
            v = VALENCE.get(w)
            if v is None:
                continue
            if mixed_case and words[i].isupper() and len(words[i]) > 1:
                v *= 1.5
            if i > 0 and lowered[i - 1] in INTENSIFIERS:
                v *= 1.3
            if any(prev in NEGATORS for prev in lowered[max(0, i - 3) : i]):
                v *= -0.74
            if but_at is not None:
                v *= 1.5 if i > but_at else 0.5
            total += v
        bangs = min(3, tokens.count("!"))
        if total:
            total += math.copysign(0.292 * bangs, total)
        return total / math.sqrt(total * total + 15.0)

    def score(self, texts: list[str]) -> pd.DataFrame:
        comp = [self.compound(t) for t in texts]
        return pd.DataFrame({"sentiment": [label_from_compound(c) for c in comp], "compound": comp,
                             "status": "ok"})


class VaderScorer:
    """Adapter for the ``vaderSentiment`` package (extra ``vader``)."""

    name = "vader"

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("install the 'vader' extra: pip install -e '.[vader]'") from exc
        self._sia = SentimentIntensityAnalyzer()

    def score(self, texts: list[str]) -> pd.DataFrame:
        comp = [self._sia.polarity_scores(str(t))["compound"] for t in texts]
        return pd.DataFrame({"sentiment": [label_from_compound(c) for c in comp], "compound": comp,
                             "status": "ok"})

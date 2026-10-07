"""Topics in ONE shared space for both periods (NMF on TF-IDF, fixed seed).

Two separate topic models (one per period) give topics that cannot be compared. Here one model is
fit on the pooled posts, so topic k means the same thing in both periods, and the prevalence of
topic k can be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import topic_tokens


@dataclass
class TopicModel:
    n_topics: int = 8
    seed: int = 7
    top_words: int = 8

    def fit(self, texts: list[str]) -> "TopicModel":
        docs = [topic_tokens(t) for t in texts]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True)
        x = self.vectorizer.fit_transform(docs)
        k = min(self.n_topics, max(1, x.shape[1] - 1), x.shape[0])
        self.nmf = NMF(n_components=k, init="nndsvda", random_state=self.seed, max_iter=400)
        self.weights_ = self.nmf.fit_transform(x)
        return self

    def describe(self) -> list[dict]:
        vocab = np.array(self.vectorizer.get_feature_names_out())
        out = []
        for k, row in enumerate(self.nmf.components_):
            words = vocab[np.argsort(row)[::-1][: self.top_words]].tolist()
            out.append({"topic": k, "label": " / ".join(words[:3]), "top_words": words})
        return out

    def assign(self, texts: list[str] | None = None) -> np.ndarray:
        """Dominant topic per post. Posts with no topic weight get -1."""
        w = self.weights_ if texts is None else self.nmf.transform(
            self.vectorizer.transform([topic_tokens(t) for t in texts]))
        dominant = w.argmax(axis=1)
        dominant[w.max(axis=1) <= 0] = -1
        return dominant


def prevalence(topics: np.ndarray, periods: pd.Series, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Share of posts per topic and period, with bootstrap intervals and the post-minus-pre difference."""
    df = pd.DataFrame({"topic": topics, "period": periods.to_numpy()})
    rng = np.random.default_rng(seed)
    rows = []
    groups = {p: df[df["period"] == p]["topic"].to_numpy() for p in ("pre", "post")}
    for k in sorted(set(topics.tolist())):
        share = {p: float(np.mean(g == k)) for p, g in groups.items()}
        diffs = []
        for _ in range(n_boot):
            b = {p: rng.choice(g, len(g)) for p, g in groups.items()}
            diffs.append(np.mean(b["post"] == k) - np.mean(b["pre"] == k))
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        rows.append({"topic": int(k), "pre_share": round(share["pre"], 4), "post_share": round(share["post"], 4),
                     "diff": round(share["post"] - share["pre"], 4), "diff_ci95": [round(lo, 4), round(hi, 4)]})
    return pd.DataFrame(rows)

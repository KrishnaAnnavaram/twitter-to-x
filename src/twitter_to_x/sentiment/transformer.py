"""Batched transformer sentiment (extra ``hf``). Default: ``cardiffnlp/twitter-roberta-base-sentiment-latest``.

* Texts go in batches (``TTX_BATCH_SIZE``), not one call per tweet.
* The model's own ``id2label`` gives the label. ``LABEL_0/1/2`` of the older Cardiff model maps to
  negative / neutral / positive.
* A batch that fails gets ``status = "error"`` and no label. The analysis counts and excludes it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_GENERIC = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}


def normalize_label(raw: str) -> str:
    key = str(raw).strip().lower()
    if key in _GENERIC:
        return _GENERIC[key]
    for name in ("negative", "neutral", "positive"):
        if key.startswith(name[:3]):
            return name
    raise ValueError(f"cannot map model label {raw!r}")


class TransformerScorer:
    name = "transformer"

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", batch_size: int = 64,
                 max_len: int = 128, local_files_only: bool = False, device: str | None = None):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError("install the 'hf' extra: pip install -e '.[hf]'") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=local_files_only).to(self.device).eval()
        self.labels = [normalize_label(self.model.config.id2label[i]) for i in range(self.model.config.num_labels)]
        self.batch_size, self.max_len = batch_size, max_len

    def score(self, texts: list[str]) -> pd.DataFrame:
        rows = []
        for i in range(0, len(texts), self.batch_size):
            batch = [str(t) for t in texts[i : i + self.batch_size]]
            try:
                with self.torch.no_grad():
                    enc = self.tokenizer(batch, truncation=True, max_length=self.max_len, padding=True,
                                         return_tensors="pt").to(self.device)
                    probs = self.torch.softmax(self.model(**enc).logits, -1).cpu().numpy()
            except Exception:  # noqa: BLE001 - one bad batch must not stop the run
                rows += [{"sentiment": None, "compound": np.nan, "status": "error"} for _ in batch]
                continue
            for p in probs:
                by = dict(zip(self.labels, p))
                rows.append({"sentiment": self.labels[int(p.argmax())],
                             "compound": float(by.get("positive", 0.0) - by.get("negative", 0.0)), "status": "ok"})
        return pd.DataFrame(rows)

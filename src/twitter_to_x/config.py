"""Settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ACQUISITION_DATE = date(2022, 10, 27)  # the ownership change closed on this day


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    out_dir: Path = Path("results")
    seed: int = 7
    cutoff: date = ACQUISITION_DATE
    salt: str | None = None  # secret for author hashing. None = a new random salt per run (no linkage)
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    batch_size: int = 64
    hf_offline: bool = False


def load_settings(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)

    def _int(name: str, default: int) -> int:
        raw = env.get(name) or str(default)
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    cutoff_raw = env.get("TTX_CUTOFF")
    cutoff = date.fromisoformat(cutoff_raw) if cutoff_raw else ACQUISITION_DATE
    batch = _int("TTX_BATCH_SIZE", 64)
    if batch < 1:
        raise ValueError("TTX_BATCH_SIZE must be at least 1")
    return Settings(
        data_dir=Path(env.get("TTX_DATA_DIR") or "data"),
        out_dir=Path(env.get("TTX_OUT_DIR") or "results"),
        seed=_int("TTX_SEED", 7),
        cutoff=cutoff,
        salt=env.get("TTX_SALT") or None,
        sentiment_model=env.get("TTX_SENTIMENT_MODEL") or "cardiffnlp/twitter-roberta-base-sentiment-latest",
        batch_size=batch,
        hf_offline=(env.get("TTX_HF_OFFLINE") or "").lower() in {"1", "true", "yes"},
    )

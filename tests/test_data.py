from datetime import date

import pandas as pd
import pytest

from twitter_to_x.config import load_settings
from twitter_to_x.data.anonymize import hash_author, mask_text
from twitter_to_x.data.filters import apply_filters, is_english, is_spam
from twitter_to_x.data.sampling import balanced_periods, sample_period, sortedness
from twitter_to_x.data.sources import SchemaError, assign_period, load_csv, load_sentiment140, load_sources


def _sorted_s140(tmp_path, n=200):
    rows = []
    for i in range(n):
        target = "0" if i < n // 2 else "4"  # sorted by label, like the real file
        rows.append(f'"{target}","{i}","Mon Apr 06 22:{i % 60:02d}:45 PDT 2009","NO_QUERY","user{i}","tweet {i} text"')
    path = tmp_path / "s140.csv"
    path.write_text("\n".join(rows), encoding="latin-1")
    return path


def test_a_tail_slice_of_sentiment140_is_one_label_but_a_sample_is_not(tmp_path):
    # reference problem 1: iloc[-300000:] of the label-sorted file kept only positive tweets
    df = load_sentiment140(_sorted_s140(tmp_path), salt="s")
    tail = df.iloc[-60:]
    assert set(tail["source_label"]) == {"positive"}
    assert sortedness(df["source_label"]) > 0.95
    sample = sample_period(df, 60, seed=1)
    shares = sample["source_label"].value_counts(normalize=True)
    assert 0.3 < shares["positive"] < 0.7
    assert sortedness(sample["source_label"]) < 0.8
    strat = sample_period(df, 60, seed=1, stratify="source_label")
    assert strat["source_label"].value_counts().to_dict() == {"negative": 30, "positive": 30}


def test_sentiment140_times_are_utc_and_users_hashed(tmp_path):
    df = load_sentiment140(_sorted_s140(tmp_path, 4), salt="s")
    assert str(df["created_at"].dt.tz) == "UTC"
    assert df["created_at"].iloc[0].hour == 5  # 22:00 PDT is 05:00 UTC
    assert "user" not in df.columns and not df["author_hash"].str.contains("user").any()


def test_hashing_and_masking():
    assert hash_author("Bob", "k") == hash_author(" bob ", "k") != hash_author("bob", "other")
    assert hash_author("", "k") == "" and len(hash_author("x", "k")) == 16
    assert mask_text("hi @Jane_Doe see https://t.co/x NOT ok!") == "hi @user see http NOT ok!"


def test_period_assignment_uses_the_cutoff():
    df = pd.DataFrame({"created_at": pd.to_datetime(["2022-10-26 23:59", "2022-10-27 00:00"], utc=True)})
    assert assign_period(df, date(2022, 10, 27))["period"].tolist() == ["pre", "post"]


def test_filters_are_rules_with_counts_per_period():
    df = pd.DataFrame({
        "text": ["the game was fun", "The game was fun!", "Follow me for FREE crypto giveaway now",
                 "este juego es muy bueno", "spam spam", "spam spam", "spam spam", "spam spam", "", "it is ok"],
        "author_hash": ["a", "b", "c", "d", "bot", "bot", "bot", "bot", "e", "f"],
        "period": ["pre", "post", "post", "pre", "pre", "pre", "pre", "pre", "pre", "post"],
    })
    out, rep = apply_filters(df)
    assert rep.removed == {"empty": {"pre": 1}, "repeated_author_text": {"pre": 4}, "duplicate_text": {"post": 1},
                           "not_english": {"pre": 1}, "spam_pattern": {"post": 1}}
    assert out["text"].tolist() == ["the game was fun", "it is ok"]
    assert is_english("this is fine") and not is_english("das ist gut")
    assert is_spam("#a #b #c #d #e #f") and not is_spam("a normal post")


def test_balanced_periods_and_loader_errors(tmp_path):
    df = pd.DataFrame({"period": ["pre"] * 10 + ["post"] * 4, "x": range(14)})
    out = balanced_periods(df, 100, seed=0)
    assert out["period"].value_counts().to_dict() == {"pre": 4, "post": 4}
    with pytest.raises(ValueError):
        balanced_periods(df[df.period == "pre"], 5, seed=0)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["x"]}).to_csv(bad, index=False)
    with pytest.raises(SchemaError):
        load_csv(bad, "s")
    with pytest.raises(ValueError):
        load_sources(["twitter=x.csv"], "s")


def test_settings():
    s = load_settings({"TTX_CUTOFF": "2023-07-24", "TTX_SEED": "3", "TTX_BATCH_SIZE": "8", "TTX_HF_OFFLINE": "1"})
    assert s.cutoff == date(2023, 7, 24) and s.seed == 3 and s.batch_size == 8 and s.hf_offline
    with pytest.raises(ValueError):
        load_settings({"TTX_BATCH_SIZE": "0"})

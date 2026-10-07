import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression

from twitter_to_x.analysis import analyze, to_markdown, write_report
from twitter_to_x.cli import main
from twitter_to_x.stats import (
    NotCountsError,
    chi_square,
    contingency,
    diff_ci,
    logistic_regression,
    proportion_ci,
    standardized_mean_difference,
    standardized_share_diff,
)
from twitter_to_x.topics import TopicModel, prevalence


def test_chi_square_needs_counts():
    # reference problem 2: the prototype passed proportions to chi2_contingency
    counts = pd.DataFrame([[30, 50, 20], [45, 35, 20]], index=["pre", "post"])
    res = chi_square(counts)
    assert res["chi2"] == pytest.approx(chi2_contingency(counts.to_numpy(), correction=False)[0])
    assert 0 < res["cramers_v"] < 1 and res["n"] == 200
    with pytest.raises(NotCountsError):
        chi_square(counts.div(counts.sum(axis=1), axis=0))
    with pytest.raises(NotCountsError):
        chi_square(pd.DataFrame([[1, 2]]))


def test_contingency_is_integer_and_keeps_label_order():
    df = pd.DataFrame({"period": ["pre", "pre", "post"], "sentiment": ["positive", "negative", "positive"]})
    table = contingency(df, "period", "sentiment", ("negative", "neutral", "positive"))
    assert list(table.columns) == ["negative", "neutral", "positive"] and table.dtypes.unique()[0].kind == "i"


def test_proportion_intervals():
    lo, hi = proportion_ci(20, 100)
    assert lo < 0.2 < hi
    d, lo, hi = diff_ci(20, 100, 40, 100)
    assert d == pytest.approx(0.2) and lo > 0 and hi < 0.4


def test_standardization_removes_a_pure_topic_mix_effect():
    # within each topic the negative rate is the same in both periods, only the topic mix changes
    rows = []
    for period, n_pol in (("pre", 100), ("post", 300)):
        rows += [{"period": period, "topic": "politics", "negative": int(i < 0.6 * n_pol)} for i in range(n_pol)]
        rows += [{"period": period, "topic": "food", "negative": int(i < 0.2 * 300)} for i in range(300)]
    res = standardized_share_diff(pd.DataFrame(rows), "negative", "topic", n_boot=100)
    assert res["crude_diff"] > 0.08
    assert res["standardized_diff"] == pytest.approx(0.0, abs=1e-9)


def test_logistic_regression_matches_sklearn():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.integers(0, 2, 2000)})
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.5 + 1.2 * x["a"] - 0.8 * x["b"])))).astype(int)
    ours = logistic_regression(y.to_numpy(), x).set_index("term")
    ref = LogisticRegression(C=1e9, max_iter=1000).fit(x, y)
    assert ours.loc["a", "coef"] == pytest.approx(ref.coef_[0][0], abs=1e-3)
    assert ours.loc["intercept", "coef"] == pytest.approx(ref.intercept_[0], abs=1e-3)
    assert ours.loc["a", "p_value"] < 1e-6 and ours.loc["a", "or_ci_low"] < ours.loc["a", "odds_ratio"]


def test_smd():
    assert standardized_mean_difference([1, 2, 3], [1, 2, 3]) == 0
    assert standardized_mean_difference([0, 1, 0, 1], [1, 2, 1, 2]) > 1


def test_topics_share_one_space():
    texts = ["vote policy budget", "vote campaign polls", "goal derby season", "goal penalty season"] * 10
    tm = TopicModel(n_topics=2, seed=1).fit(texts)
    topics = tm.assign()
    assert topics[0] == topics[1] != topics[2] == topics[3]
    prev = prevalence(topics, pd.Series(["pre", "post"] * 20), n_boot=20)
    assert set(prev.columns) >= {"topic", "pre_share", "post_share", "diff", "diff_ci95"}


def test_analysis_recovers_the_built_in_effects(scored, prepared, tmp_path):
    _, manifest = prepared
    assert manifest["rows_per_period"]["pre"] == manifest["rows_per_period"]["post"]
    assert "repeated_author_text" in manifest["filters"]["removed"]
    r = analyze(scored, n_topics=6, n_boot=100)
    assert r["moderation_share"]["post"] > r["moderation_share"]["pre"] and r["moderation_share"]["diff_ci95"][0] > 0
    assert r["moderation_negative_share"]["diff"] > 0
    assert r["check_against_truth"]["moderation_precision"] > 0.95
    assert r["check_against_truth"]["sentiment_accuracy"] > 0.85
    adj = r["negative_share_topic_adjusted"]
    assert adj["crude_diff"] > adj["standardized_diff"]  # part of the crude change is the topic mix
    out = write_report(r, tmp_path)
    assert "Chi-square on counts" in to_markdown(r)
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["scorer"] == "lexicon"


def test_cli(synthetic_csv, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TTX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TTX_OUT_DIR", str(tmp_path / "results"))
    assert main(["prepare", "--source", f"csv={synthetic_csv}", "--n-per-period", "400"]) == 0
    assert main(["analyze", "--topics", "4", "--n-boot", "30"]) == 0
    scores = pd.read_csv(tmp_path / "results" / "lexicon" / "scores.csv")
    assert "text" not in scores.columns and len(scores) == 800
    assert main(["validate-moderation"]) == 0
    capsys.readouterr()
    assert main(["flag", "my account got suspended"]) == 0
    assert json.loads(capsys.readouterr().out)["moderation"] is True

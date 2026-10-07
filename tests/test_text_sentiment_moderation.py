import pandas as pd
import pytest

from twitter_to_x.analysis import analyze
from twitter_to_x.moderation import ModerationDetector, substring_flag
from twitter_to_x.moderation.validate import evaluate_flags, load_gold, wilson
from twitter_to_x.sentiment import build_scorer
from twitter_to_x.sentiment.lexicon import LexiconScorer
from twitter_to_x.sentiment.transformer import normalize_label
from twitter_to_x.text import light_clean, topic_tokens

LEX = LexiconScorer()


def test_sentiment_text_keeps_negations_case_and_punctuation():
    # reference problem 3: the sentiment models got lemmatized text with "not" removed
    assert light_clean("I do NOT like this @bob!!! https://x.y &amp; more") == "I do NOT like this @user!!! http & more"
    assert "not" not in topic_tokens("I do not like the new policy").split()
    assert topic_tokens("The new policy") == "new policy"


def test_lexicon_rules():
    assert LEX.compound("this is good") > 0 > LEX.compound("this is not good")
    assert LEX.compound("this is GOOD") > LEX.compound("this is good")
    assert LEX.compound("this is very good") > LEX.compound("this is good")
    assert LEX.compound("good!!!") > LEX.compound("good")
    assert LEX.compound("it was good but the ending was awful") < 0
    out = LEX.score(["great day", "the meeting is at noon", "worst update ever"])
    assert out["sentiment"].tolist() == ["positive", "neutral", "negative"] and set(out["status"]) == {"ok"}
    with pytest.raises(ValueError):
        build_scorer("bert")


def test_transformer_label_mapping():
    assert [normalize_label(x) for x in ("LABEL_0", "LABEL_1", "LABEL_2")] == ["negative", "neutral", "positive"]
    assert normalize_label("Positive") == "positive" and normalize_label("neg") == "negative"
    with pytest.raises(ValueError):
        normalize_label("joy")


def test_scorer_errors_are_counted_and_excluded(scored):
    # reference problem 7: errors became a fourth "NEUTRAL" label
    broken = scored.copy()
    broken.loc[:9, ["sentiment", "status"]] = [None, "error"]
    result = analyze(broken, n_topics=4, n_boot=50)
    assert sum(result["scorer_errors_excluded"].values()) == 10
    assert sum(result["posts"].values()) == len(scored) - 10
    assert set(result["sentiment_shares"]["pre"]) == {"negative", "neutral", "positive"}


@pytest.mark.parametrize("text", ["urban gardening", "banana bread", "a new skill", "grape juice", "Essex rain",
                                  "I removed the stain", "the band played", "banished to the kitchen"])
def test_word_boundaries_stop_false_matches(text):
    # reference problem 4: substring matching flagged "urban", "banana", "skill", "grape", "Essex"
    assert not ModerationDetector().flag(text)


def test_detector_categories_and_terms():
    m = ModerationDetector().match("They BANNED him and my post was removed. Free speech?")
    assert m.flagged and m.categories == ("account_action", "content_action", "policy_and_speech")
    assert "banned" in m.terms
    assert ModerationDetector(("verification",)).flag("blue checks are back")
    with pytest.raises(ValueError):
        ModerationDetector(("spam",))


def test_gold_set_validation_beats_the_substring_baseline():
    gold = load_gold()
    good = evaluate_flags(gold, ModerationDetector().flag)
    naive = evaluate_flags(gold, substring_flag)
    assert good["precision"] >= 0.85 and good["recall"] >= 0.9
    assert naive["f1"] < good["f1"] - 0.3
    lo, hi = wilson(9, 10)
    assert 0.5 < lo < 0.9 < hi <= 1.0
    with pytest.raises(ValueError):
        evaluate_flags(load_gold_frame(), ModerationDetector().flag)


def load_gold_frame():
    import tempfile
    from pathlib import Path

    folder = Path(tempfile.mkdtemp())
    pd.DataFrame({"text": ["x"], "label": [2]}).to_csv(folder / "g.csv", index=False)
    return load_gold(folder / "g.csv")

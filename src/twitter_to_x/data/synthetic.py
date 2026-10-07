"""Synthetic posts with a KNOWN truth, for the offline demo and the tests.

Built-in effects (all on purpose):

* ``post`` has more moderation talk (10 %) than ``pre`` (4 %).
* Moderation posts are more negative in ``post`` (65 %) than in ``pre`` (45 %).
* The topic mix changes: ``post`` has more politics, and politics posts are more negative. This is a
  confounder: the crude sentiment change is larger than the topic-adjusted change.
* Traps for keyword matching ("urban", "banana", "skill"), spam bots, duplicates and @handles.

Columns ``true_moderation`` and ``true_sentiment`` hold the truth, so tests can check each step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOPICS = {
    "sports": ["the match", "our striker", "the final", "the coach", "the league table"],
    "tech": ["the new phone", "this laptop", "the update", "my router", "the app store"],
    "politics": ["the election", "the senate vote", "the new tax plan", "the debate", "the minister"],
    "food": ["banana bread", "the urban cafe", "grape juice", "my sourdough", "the taco truck"],
    "life": ["my commute", "the weekend", "my new skill", "the weather", "my cat"],
}
POS = ["is great", "was amazing", "made me so happy", "is the best", "looks fantastic", "is really good"]
NEG = ["is terrible", "was awful", "made me so angry", "is the worst", "looks like a mess", "is not good"]
NEU = ["starts at noon", "is on channel four", "was moved to Friday", "has a new date", "is in the news today"]
MOD = {
    "negative": ["my account got suspended again and it is unfair", "censorship on this app is the worst",
                 "they banned another journalist, terrible call", "free speech is dead here, sad",
                 "my post was removed for nothing, awful", "hate speech stays up and nobody acts"],
    "positive": ["community notes fixed that fake claim, great", "good call to ban that scam network",
                 "the new moderation rules look fair", "glad they removed the spam bots",
                 "verification finally works, nice"],
    "neutral": ["the content moderation report is out", "blue checks change on Monday",
                "the terms of service update starts in May"],
}
CONTEXT = {
    "sports": ["goal", "season", "match day", "transfer window", "penalty", "derby"],
    "tech": ["battery", "software", "download", "chip", "screen", "bug fix"],
    "politics": ["vote", "policy", "campaign", "parliament", "budget", "polls"],
    "food": ["recipe", "dinner", "bakery", "brunch", "dessert", "spicy"],
    "life": ["morning", "traffic", "holiday", "rain", "gym", "neighbours"],
}
TAILS = ["", "", "today", "again", "honestly", "this week", "right now", "lol", "for real", "tonight", "as usual"]
SPAM = "Follow me for a FREE crypto giveaway, follow and retweet"
NAMES = ["ana", "bo", "cy", "dee", "eli", "fay", "gus", "hal", "ivy", "jo"]


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def make_posts(n_per_period: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mix = {"pre": [0.25, 0.25, 0.15, 0.15, 0.20], "post": [0.20, 0.20, 0.30, 0.15, 0.15]}
    neg_rate = {"sports": 0.30, "tech": 0.35, "politics": 0.55, "food": 0.15, "life": 0.30}
    mod_share = {"pre": 0.04, "post": 0.10}
    mod_neg = {"pre": 0.45, "post": 0.65}
    rows = []
    for period, (start, end) in {"pre": ("2021-06-01", "2022-10-20"), "post": ("2023-08-01", "2025-01-31")}.items():
        span = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
        for i in range(n_per_period):
            when = pd.Timestamp(start, tz="UTC") + pd.Timedelta(seconds=float(rng.random() * span))
            author = f"{_pick(rng, NAMES)}{int(rng.integers(1000))}"
            if rng.random() < mod_share[period]:
                r = rng.random()
                sent = "negative" if r < mod_neg[period] else ("positive" if r < mod_neg[period] + 0.25 else "neutral")
                text, topic, moderation = _pick(rng, MOD[sent]), "platform", 1
            else:
                topic = list(TOPICS)[int(rng.choice(5, p=mix[period]))]
                r = rng.random()
                sent = "negative" if r < neg_rate[topic] else ("neutral" if r < neg_rate[topic] + 0.3 else "positive")
                phrase = _pick(rng, {"negative": NEG, "positive": POS, "neutral": NEU}[sent])
                text = f"{_pick(rng, TOPICS[topic]).capitalize()} {phrase}, {_pick(rng, CONTEXT[topic])}"
                if rng.random() < 0.5:
                    text += f" and {_pick(rng, CONTEXT[topic])}"
                moderation = 0
            if rng.random() < 0.15:
                text = f"@{_pick(rng, NAMES)} {text}"
            tail = _pick(rng, TAILS)
            text = f"{text} {tail}".strip()
            if rng.random() < 0.5:
                text += " " + str(int(rng.integers(1, 99)))  # makes most texts unique
            rows.append({"post_id": f"{period}-{i}", "text": text, "created_at": when.isoformat(), "author": author,
                         "true_moderation": moderation, "true_sentiment": sent, "true_topic": topic})
        for j in range(int(0.02 * n_per_period)):  # one spam bot per period, many copies
            rows.append({"post_id": f"{period}-spam-{j}", "text": SPAM, "created_at": (pd.Timestamp(start, tz="UTC")
                         + pd.Timedelta(days=j)).isoformat(), "author": f"bot_{period}", "true_moderation": 0,
                         "true_sentiment": "neutral", "true_topic": "spam"})
    return pd.DataFrame(rows)

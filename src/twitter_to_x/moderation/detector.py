"""Find posts that talk about content moderation.

The prototype matched keywords as substrings: "ban" matched "urban" and "banana", "kill" matched
"skill", "rape" matched "grape". Here each pattern has word boundaries and an explicit set of word
forms. Generic words ("removed", "deleted") count only next to "tweet", "post" or "account".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CATEGORIES: dict[str, list[str]] = {
    "account_action": [
        r"suspend(?:ed|s|ing)?", r"suspensions?", r"shadow[- ]?bann?(?:ed|ing|s)?", r"bann?(?:ed|ing|s)?",
        r"deplatform(?:ed|ing|s)?", r"reinstat(?:ed|e|ing)", r"(?:account|profile)s? (?:was |got |is )?"
        r"(?:locked|restricted|limited|suspended)",
    ],
    "content_action": [
        r"community notes?", r"fact[- ]?check(?:ed|s|ing|ers?)?", r"(?:labell?ed|flagged) as (?:misleading|false)",
        r"(?:tweet|post|reply|thread|account)s? (?:was |were |got |has been )?(?:removed|deleted|taken down|hidden)",
        r"content warnings?",
    ],
    "policy_and_speech": [
        r"free speech", r"freedom of speech", r"censor(?:ship|ed|ing|s)?", r"content moderation",
        r"moderation (?:rules?|polic(?:y|ies)|team|reports?)",
        r"moderat(?:ors?|ion team)", r"hate speech", r"(?:mis|dis)information", r"trust and safety",
        r"terms of service", r"community guidelines",
    ],
    "verification": [r"blue (?:checks?|ticks?|checkmarks?)", r"verification", r"paid verification"],
}


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)


_COMPILED = {cat: _compile(p) for cat, p in CATEGORIES.items()}

# the prototype-style substring check, kept only to measure how much worse it is
_NAIVE_TERMS = ["ban", "suspend", "censor", "kill", "rape", "sex", "removed", "deleted", "illegal", "moderation",
                "free speech", "fact check", "misinformation", "hate", "verified", "blue check"]


@dataclass(frozen=True)
class Match:
    flagged: bool
    categories: tuple[str, ...]
    terms: tuple[str, ...]


class ModerationDetector:
    def __init__(self, categories: tuple[str, ...] | None = None):
        self.categories = categories or tuple(CATEGORIES)
        unknown = set(self.categories) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"unknown categories {sorted(unknown)}")

    def match(self, text: str) -> Match:
        cats, terms = [], []
        for cat in self.categories:
            found = _COMPILED[cat].findall(str(text))
            if found:
                cats.append(cat)
                terms += [f.lower() for f in found]
        return Match(bool(cats), tuple(cats), tuple(dict.fromkeys(terms)))

    def flag(self, text: str) -> bool:
        return self.match(text).flagged


def substring_flag(text: str) -> bool:
    lowered = str(text).lower()
    return any(term in lowered for term in _NAIVE_TERMS)

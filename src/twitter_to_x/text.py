"""Two text views, for two different jobs.

* ``light_clean`` - for the sentiment scorers. It keeps case, punctuation, emoji and negations.
  Both periods get exactly the same function, one time.
* ``topic_tokens`` - for the topic model only. It lower-cases, drops stop words and short tokens.

The prototype gave the sentiment models the topic view (no stop words, so no "not"), which flips
negated sentences.
"""

from __future__ import annotations

import re

from .data.anonymize import mask_text

_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z][a-z']{2,}")
STOP_WORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i
if in into is it its itself just me more most my no nor not of off on once only or other our out over own same she
should so some such than that the their them then there these they this those through to too under until up very was
we were what when where which while who whom why will with would you your yours rt amp http user im dont cant get got
one like
""".split())


def light_clean(text: str) -> str:
    """Mask handles and links, decode ``&amp;`` and collapse spaces. Nothing else changes."""
    t = mask_text(text).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _SPACE.sub(" ", t).strip()


def topic_tokens(text: str) -> str:
    """Lower-case content words for the topic model (negations are not needed for topics)."""
    words = _TOKEN.findall(light_clean(text).lower())
    return " ".join(w for w in words if w not in STOP_WORDS)

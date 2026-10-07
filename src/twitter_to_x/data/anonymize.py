"""Privacy steps: salted author hashes and masked handles. No raw user name leaves the loader."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

_HANDLE = re.compile(r"(?<![\w@])@\w{1,30}")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


def new_salt() -> str:
    return secrets.token_hex(16)


def hash_author(author: object, salt: str) -> str:
    """HMAC-SHA256 of the author id with a secret salt, cut to 16 hex characters. Empty for no author."""
    if author is None or (isinstance(author, float) and author != author) or str(author).strip() == "":
        return ""
    return hmac.new(salt.encode("utf-8"), str(author).strip().lower().encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def mask_text(text: str) -> str:
    """Replace @handles with ``@user`` and links with ``http``. Case, punctuation and emoji stay."""
    return _URL.sub("http", _HANDLE.sub("@user", str(text)))

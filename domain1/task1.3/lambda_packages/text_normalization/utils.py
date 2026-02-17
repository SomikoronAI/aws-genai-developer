# utils.py

import os, sys, json
import re
import hashlib
import datetime
import unicodedata


# ------------------------------
# Helpers Methods
# ------------------------------
def sha256_hex(s: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 encoded string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha256_path(p: str) -> str:
     """Return the SHA-256 hex digest of an absolute filesystem path."""
     return hashlib.sha256(os.path.abspath(p).encode("utf-8")).hexdigest()

def now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    """
    Normalizes text for downstream RAG processing:
    - Unicode normalization
    - Removes control characters
    - Normalizes quotes and dashes
    - Cleans whitespace while preserving paragraphs
    """

    if not text:
        return ""

    # 1. Unicode normalization
    t = unicodedata.normalize("NFC", text)

    # 2. Remove control characters (except newline)
    t = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", t)

    # 3. Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Normalize quotes and dashes (helps embeddings)
    replacements = {
        "“": '"', "”": '"',
        "‘": "'", "’": "'",
        "–": "-", "—": "-",
        "…": "..."
    }
    for k, v in replacements.items():
        t = t.replace(k, v)

    # 5. Normalize whitespace within lines
    t = re.sub(r"[ \t]+", " ", t)

    # 6. Collapse excessive blank lines (keep max 1)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # 7. Trim spaces around lines
    t = "\n".join(line.strip() for line in t.split("\n"))

    return t.strip()

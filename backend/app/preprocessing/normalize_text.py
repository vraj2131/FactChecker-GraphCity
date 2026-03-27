import html
import unicodedata
from typing import Optional


def normalize_claim_text(text: Optional[str]) -> Optional[str]:
    """
    Minimal, safe text normalization for FEVER claims.

    Steps:
    - Unicode normalization
    - Replace non-breaking spaces
    - Collapse repeated whitespace
    - Strip leading/trailing whitespace
    """
    if text is None:
        return None

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = " ".join(text.split())
    return text.strip()


def normalize_snippet_text(text: Optional[str]) -> Optional[str]:
    """
    Normalization for source snippet text.

    Extends normalize_claim_text with:
    - HTML entity decoding (&amp; → &, &nbsp; → space, &#39; → ', etc.)
    - Removal of leading/trailing quote characters (" ' \u201c \u201d \u2018 \u2019)
    """
    if text is None:
        return None

    # Decode HTML entities first
    text = html.unescape(text)

    # Standard Unicode + whitespace normalization
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = " ".join(text.split())

    # Strip surrounding quote characters
    text = text.strip('"\'\u201c\u201d\u2018\u2019')
    text = text.strip()

    return text or None
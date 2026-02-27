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
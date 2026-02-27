import re
from typing import Optional


def clean_page_title(page_title: Optional[str]) -> Optional[str]:
    """
    Normalize FEVER/Wikipedia page titles for joining and display.

    Examples:
    - "Soul_Food_-LRB-film-RRB-" -> "Soul_Food_(film)"
    - None -> None
    """
    if page_title is None:
        return None

    text = str(page_title).strip()
    if not text:
        return None

    text = text.replace("-LRB-", "(")
    text = text.replace("-RRB-", ")")
    text = text.replace("-LSB-", "[")
    text = text.replace("-RSB-", "]")
    text = text.replace("-LCB-", "{")
    text = text.replace("-RCB-", "}")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_sentence_text(sentence: Optional[str]) -> Optional[str]:
    """
    Light cleanup for extracted wiki sentence text.
    """
    if sentence is None:
        return None

    text = str(sentence).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def clean_source_text(text: Optional[str]) -> Optional[str]:
    """
    General cleanup for source/snippet text.
    """
    if text is None:
        return None

    cleaned = str(text).replace("\u00a0", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """
    Return current UTC datetime.
    """
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """
    Return current UTC datetime as ISO string.
    """
    return utc_now().isoformat()


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO-like datetime string safely.

    Supports values like:
    - 2026-03-13T12:34:56Z
    - 2026-03-13T12:34:56+00:00
    - 2026-03-13
    """
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def to_utc(dt: datetime) -> datetime:
    """
    Convert a datetime to UTC.
    If naive, assume UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_iso_utc(dt: datetime) -> str:
    """
    Format datetime as UTC ISO string ending with Z.
    """
    dt_utc = to_utc(dt)
    return dt_utc.isoformat().replace("+00:00", "Z")


def safe_date_string(value: Optional[str]) -> Optional[str]:
    """
    Convert an ISO-like datetime string to YYYY-MM-DD.
    Returns None if parsing fails.
    """
    dt = parse_iso_datetime(value)
    if dt is None:
        return None
    return dt.date().isoformat()


def is_recent(value: Optional[str], max_age_days: int) -> bool:
    """
    Return True if the given datetime string is within max_age_days from now.
    """
    dt = parse_iso_datetime(value)
    if dt is None:
        return False

    age = utc_now() - to_utc(dt)
    return age.days <= max_age_days


def days_between(start: Optional[str], end: Optional[str]) -> Optional[int]:
    """
    Return absolute number of days between two ISO-like datetime strings.
    """
    start_dt = parse_iso_datetime(start)
    end_dt = parse_iso_datetime(end)

    if start_dt is None or end_dt is None:
        return None

    delta = to_utc(end_dt) - to_utc(start_dt)
    return abs(delta.days)
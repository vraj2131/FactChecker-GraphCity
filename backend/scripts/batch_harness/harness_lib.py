"""Shared infrastructure for the batch harness: manifest I/O, error
classification, and quota-exhaustion detection.

The manifest is append-only JSONL, fsync'd per line. The previous approach
(`generate_factcheck_examples.py`) wrote a single summary.json at the very
end, so a crash at claim 190 lost everything. Here every claim is durable the
moment it finishes, and a torn final line is simply skipped on read.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import signal
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Overridable so an operator can give stubborn claims another pass after
# fixing the cause of their failure, without editing code.
MAX_ATTEMPTS = int(os.getenv("HARNESS_MAX_ATTEMPTS", "3"))

# Retrievers whose sustained absence signals a spent quota rather than a
# genuinely evidence-free claim.
WATCHED_RETRIEVERS = ["guardian", "gdelt", "factcheck", "livewiki", "duckduckgo", "bluesky"]
ZERO_STREAK_ALERT = 8    # flag the claim record
ZERO_STREAK_ABORT = 25   # two or more retrievers this dead => run is producing junk

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DEGRADED = 3


# --------------------------------------------------------------------------
# Timeouts
# --------------------------------------------------------------------------

class ClaimTimeout(Exception):
    """A single claim exceeded its wall-clock budget."""


@contextlib.contextmanager
def claim_timeout(seconds: int):
    """Hard wall-clock timeout that interrupts stuck blocking syscalls.

    socket.setdefaulttimeout doesn't cover every case (an SSL handshake can
    ignore it), so SIGALRM is the backstop. Main thread + Unix only.
    """
    def _handler(signum, frame):
        raise ClaimTimeout(f"claim exceeded {seconds}s timeout")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# --------------------------------------------------------------------------
# Identity & metadata
# --------------------------------------------------------------------------

def make_claim_id(index: int, claim: str) -> str:
    """Stable id: position + content hash.

    The hash half means reordering claims.py doesn't silently remap existing
    manifest records onto different claims.
    """
    digest = hashlib.sha1(claim.strip().encode("utf-8")).hexdigest()[:8]
    return f"c{index:03d}_{digest}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------

_PERMANENT_TYPES = {
    "ValidationError",      # pydantic — graph failed schema
    "KeyError",
    "TypeError",
    "AttributeError",
}

_RETRYABLE_TYPES = {
    "ClaimTimeout",
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ConnectionError",
    "ConnectionResetError",
    "TimeoutError",
    "ReadTimeout",
    "ConnectTimeout",
    "timeout",
    "RuntimeError",         # "Main retrieval failed: ..." — usually transient network
}

_PERMANENT_MESSAGE_HINTS = (
    "failed to parse json",
    "claim_text cannot be empty",
    "cannot be empty",
)

# Checked BEFORE the permanent hints. An empty completion is a model hiccup,
# not a defect in the claim: the reasoning-style models (gpt-oss-*) sometimes
# spend their whole token budget on internal reasoning and return no content.
# Another model, or the same one on a retry, handles it fine.
_RETRYABLE_MESSAGE_HINTS = (
    "no json object found in model output.\nraw output: ",
    "no json object found in model output. raw output: ",
)


def classify_error(exc: BaseException) -> str:
    """Return "retryable" or "permanent".

    Unknown errors default to retryable — one wasted retry is cheaper than
    silently dropping a claim from a 200-claim deliverable.
    """
    name = type(exc).__name__
    message = str(exc).lower()

    # An empty completion looks like a parse failure but is worth retrying.
    if any(message.rstrip().endswith(h.rstrip()) for h in _RETRYABLE_MESSAGE_HINTS):
        return "retryable"
    if any(hint in message for hint in _PERMANENT_MESSAGE_HINTS):
        return "permanent"
    if name in _PERMANENT_TYPES:
        return "permanent"
    if name in _RETRYABLE_TYPES:
        return "retryable"
    return "retryable"


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

class Manifest:
    """Append-only JSONL record store with crash-safe resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def append(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()

    @staticmethod
    def read_records(path: Path) -> List[Dict[str, Any]]:
        """Read all records, tolerating a torn final line from a hard kill."""
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("manifest: skipping malformed line (%.60s...)", line)
        return records

    @classmethod
    def load_state(cls, path: Path) -> tuple[Dict[str, Dict], Counter]:
        """Return (latest record per claim_id, attempt counts per claim_id)."""
        records = cls.read_records(path)
        latest: Dict[str, Dict] = {}
        attempts: Counter = Counter()
        for record in records:
            claim_id = record.get("claim_id")
            if not claim_id:
                continue
            latest[claim_id] = record
            attempts[claim_id] += 1
        return latest, attempts


def should_run(claim_id: str, latest: Dict[str, Dict], attempts: Counter) -> bool:
    """Decide whether a claim still needs work on this pass."""
    record = latest.get(claim_id)
    if record is None:
        return True

    status = record.get("status")
    if status == "ok":
        # Self-healing: if the graph file vanished, redo it.
        graph_path = record.get("graph_path")
        return not (graph_path and Path(graph_path).exists())
    if status in ("permanent", "abandoned"):
        return False
    return attempts[claim_id] < MAX_ATTEMPTS


# --------------------------------------------------------------------------
# Quota / degradation tracking
# --------------------------------------------------------------------------

class QuotaWatch:
    """Tracks consecutive claims where a retriever contributed zero sources.

    `retrieval_source_counts` is computed post-dedup/ranking, so a single zero
    is unremarkable — only a sustained run implies the retriever is actually
    dead (quota spent, key revoked, endpoint down).
    """

    def __init__(self, watched: Iterable[str] = WATCHED_RETRIEVERS) -> None:
        self.watched = list(watched)
        self.zero_streak: Dict[str, int] = defaultdict(int)
        self.flagged: set[str] = set()

    def update(self, source_counts: Dict[str, int]) -> List[str]:
        """Feed one claim's counts. Returns quota flags for that claim."""
        flags: List[str] = []
        for name in self.watched:
            if source_counts.get(name, 0) > 0:
                self.zero_streak[name] = 0
                continue
            self.zero_streak[name] += 1
            if self.zero_streak[name] >= ZERO_STREAK_ALERT:
                flags.append(f"{name}_suspected_exhausted")
                if name not in self.flagged:
                    self.flagged.add(name)
                    logger.warning(
                        "QUOTA: '%s' contributed 0 sources for %d consecutive claims",
                        name, self.zero_streak[name],
                    )
        return flags

    def is_degraded(self) -> bool:
        """True when enough retrievers are dead that results are junk."""
        dead = [n for n in self.watched if self.zero_streak[n] >= ZERO_STREAK_ABORT]
        return len(dead) >= 2

    def dead_retrievers(self) -> List[str]:
        return [n for n in self.watched if self.zero_streak[n] >= ZERO_STREAK_ABORT]


# --------------------------------------------------------------------------
# Pacing
# --------------------------------------------------------------------------

def sleep_to_floor(cycle_start: float, floor_s: float, elapsed_s: float,
                   cache_hit_threshold_s: float = 5.0) -> float:
    """Sleep so the claim cycle takes at least `floor_s`.

    Skipped entirely for cache hits — on resume, replaying already-completed
    claims costs no API tokens, so pacing them would waste hours for nothing.
    Returns the number of seconds actually slept.
    """
    if elapsed_s < cache_hit_threshold_s:
        return 0.0
    remaining = floor_s - (time.monotonic() - cycle_start)
    if remaining <= 0:
        return 0.0
    time.sleep(remaining)
    return remaining


def sleep_until_quota_reset(reset_seconds: float, marker_path: Optional[Path] = None,
                            safety_margin_s: float = 300.0) -> None:
    """Block until a daily API quota window rolls over.

    Writes a marker file so an external observer (or the watchdog) can tell
    the difference between "paused on purpose" and "hung".
    """
    total = max(60.0, reset_seconds + safety_margin_s)
    resume_at = datetime.now(timezone.utc).timestamp() + total
    resume_iso = datetime.fromtimestamp(resume_at, timezone.utc).isoformat(timespec="seconds")
    logger.warning(
        "DAILY QUOTA EXHAUSTED — sleeping %.0fs (%.1fh), resuming at %s",
        total, total / 3600.0, resume_iso,
    )
    if marker_path:
        marker_path.write_text(
            json.dumps({"paused_at": utc_now_iso(), "resume_at": resume_iso,
                        "sleep_seconds": round(total, 1)}, indent=2)
        )
    # Sleep in chunks so a Ctrl-C / SIGTERM isn't stuck behind one long call.
    deadline = time.monotonic() + total
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(60.0, remaining))
    if marker_path and marker_path.exists():
        with contextlib.suppress(Exception):
            marker_path.unlink()
    logger.warning("Quota pause complete — resuming run")

"""
Feature 13: concurrency helper for the retrieval layer.

Retrieval is the dominant cost in claim verification (~33 sequential
HTTP calls per claim across base retrieval, adversarial queries, query
decomposition, and the low-yield fallback — see plan.md "Round 3").
Each retriever call is already independent and exception-isolated, so
running them concurrently in a thread pool is a safe, local change that
doesn't touch retriever/ranking/confidence logic at all.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from typing import Callable, List, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_concurrent(
    tasks: List[Tuple[str, Callable[[], T]]],
    max_workers: int = 10,
    timeout: float = 8.0,
) -> List[Tuple[str, T]]:
    """
    Run (label, callable) pairs concurrently in a thread pool.

    A single failing or slow task never blocks the others — failures are
    logged and skipped, and the whole batch is bounded to ~`timeout`
    seconds of wall-clock time even if some underlying call hangs (the
    executor is not waited on at shutdown, so a stuck thread is abandoned
    rather than blocking the caller).

    Returns:
        List of (label, result) for tasks that completed successfully
        within the timeout. Order matches completion order, not
        submission order.
    """
    if not tasks:
        return []

    results: List[Tuple[str, T]] = []
    executor = ThreadPoolExecutor(max_workers=min(max_workers, len(tasks)))
    future_to_label = {executor.submit(fn): label for label, fn in tasks}

    try:
        for future in as_completed(future_to_label, timeout=timeout):
            label = future_to_label[future]
            try:
                results.append((label, future.result()))
            except Exception as exc:
                logger.warning("run_concurrent: task '%s' failed: %s", label, exc)
    except FutureTimeoutError:
        pending = sum(1 for f in future_to_label if not f.done())
        logger.warning(
            "run_concurrent: timed out after %.1fs with %d/%d tasks still "
            "pending — proceeding with %d completed results",
            timeout, pending, len(tasks), len(results),
        )
    finally:
        # wait=False: don't block the caller on stuck threads (e.g. a
        # retriever whose HTTP client ignores its own timeout). They'll
        # finish or die on their own; we've already moved on.
        executor.shutdown(wait=False)

    return results

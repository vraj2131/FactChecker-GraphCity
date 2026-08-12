"""Phase 1 — run every claim through the pipeline and save its graph JSON.

Runs the services in-process rather than over HTTP. That's deliberate: the
HTTP API's `live_news` source group bundles newsapi with guardian/gdelt/
livewiki, so there's no way to shed a single quota-exhausted retriever
through the API. In-process we can edit the registry directly.

Resumable — every claim is durably recorded in manifest.jsonl the moment it
finishes, so a crash costs one claim, not the run. Re-running skips completed
claims, and because retrieval/LLM results are cached, the replay is nearly
free (both in time and in API quota).

Usage:
    python -m backend.scripts.batch_harness.run_phase1_verify --run-dir <dir>
    python -m backend.scripts.batch_harness.run_phase1_verify --run-dir <dir> --limit 5
"""
from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
from pathlib import Path

# Retriever calls occasionally hang on a socket connect/read with no client
# timeout; a global default lets them fail fast instead of stalling the run.
socket.setdefaulttimeout(30)

from backend.app.models.llm_model import GroqDailyQuotaExhausted
from backend.app.services.graph_builder_service import GraphBuilderService
from backend.app.services.verify_claim_service import VerifyClaimService
import backend.app.services.retrieval_service as retrieval_service
from backend.scripts.batch_harness.claims import ordered_claims
from backend.scripts.batch_harness.harness_lib import (
    EXIT_DEGRADED,
    EXIT_ERROR,
    EXIT_OK,
    Manifest,
    QuotaWatch,
    ClaimTimeout,
    claim_timeout,
    classify_error,
    git_sha,
    make_claim_id,
    should_run,
    sleep_to_floor,
    sleep_until_quota_reset,
    utc_now_iso,
    MAX_ATTEMPTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase1")

# Per-claim wall-clock ceiling. Generous because Groq backoff can legitimately
# sleep for minutes across the two-call JSON-reparse path; a tighter bound
# would kill claims that were about to succeed.
CLAIM_TIMEOUT_S = 600

# Pacing floor per claim cycle, sized to stay under Groq's tokens-per-minute
# budget at ~5k tokens per claim.
PACING_FLOOR_S = 100.0

# Every source group. verify() defaults to only 5 of these — web_search and
# social are opt-in — but this run is meant to exercise the full retriever set.
ALL_SOURCE_GROUPS = [
    "wikipedia", "live_news", "factcheck", "scientific",
    "financial", "web_search", "social",
]


def configure_services(verify_svc: VerifyClaimService) -> None:
    """Trim per-claim API fan-out so daily quotas survive 200 claims.

    The retrieval service reads these module-level sets at call time, so
    reassigning them here changes behaviour for this process only — no
    production code is edited.
    """
    # Context expansion issues up to 3 extra queries x 4 retrievers per claim,
    # plus an extra Groq call. Far too expensive at this volume.
    verify_svc._context_expansion = None

    # Guardian's free tier is 500 calls/day. At base + 2 adversarial +
    # 2 decomposition it would need ~1000 for 200 claims, so it's cut to
    # base-only. Fact-check and GDELT keep their adversarial queries, which
    # are the ones that surface debunks.
    retrieval_service._ADVERSARIAL_RETRIEVER_SOURCES = {"factcheck", "gdelt"}
    retrieval_service._DECOMP_RETRIEVER_SOURCES = {"livewiki"}

    registry = verify_svc._retrieval._registry._retrievers
    # No REDDIT_CLIENT_ID/SECRET in .env — it returns [] on every call, so
    # keeping it just burns a thread per claim.
    registry.pop("reddit", None)

    logger.info(
        "Registry: %s | groups=%s | adversarial=%s | decomp=%s | context_expansion=off",
        sorted(registry.keys()),
        ALL_SOURCE_GROUPS,
        sorted(retrieval_service._ADVERSARIAL_RETRIEVER_SOURCES),
        sorted(retrieval_service._DECOMP_RETRIEVER_SOURCES),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: verify claims -> graph JSONs")
    parser.add_argument("--run-dir", required=True, help="Output directory for this run")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N claims")
    parser.add_argument("--pacing", type=float, default=PACING_FLOOR_S,
                        help=f"Seconds floor per claim cycle (default {PACING_FLOOR_S})")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    graphs_dir = run_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.jsonl"
    paused_marker = run_dir / "PAUSED_UNTIL.json"
    degraded_marker = run_dir / "RUN_DEGRADED"

    all_claims = ordered_claims()
    if args.limit:
        all_claims = all_claims[: args.limit]

    latest, attempts = Manifest.load_state(manifest_path)
    todo = [
        (i, domain, claim)
        for i, (domain, claim) in enumerate(all_claims, start=1)
        if should_run(make_claim_id(i, claim), latest, attempts)
    ]

    done_count = len(all_claims) - len(todo)
    logger.info(
        "Phase 1: %d claims total | %d already terminal | %d to process",
        len(all_claims), done_count, len(todo),
    )
    if not todo:
        logger.info("Nothing to do — all claims terminal.")
        return EXIT_OK

    logger.info("Loading models (NLI x2 + FAISS + Groq)...")
    t_load = time.monotonic()
    verify_svc = VerifyClaimService.build_default()
    configure_services(verify_svc)
    builder = GraphBuilderService()
    logger.info("Models ready in %.1fs", time.monotonic() - t_load)

    manifest = Manifest(manifest_path)
    quota = QuotaWatch()
    sha = git_sha()
    counts = {"ok": 0, "retryable": 0, "permanent": 0, "abandoned": 0}

    try:
        for n, (index, domain, claim) in enumerate(todo, start=1):
            claim_id = make_claim_id(index, claim)
            attempt = attempts[claim_id] + 1
            cycle_start = time.monotonic()
            started = utc_now_iso()

            logger.info("[%d/%d] %s (attempt %d) %s", n, len(todo), claim_id, attempt, claim[:70])

            record = {
                "schema_version": 1,
                "claim_id": claim_id,
                "index": index,
                "domain": domain,
                "claim": claim,
                "attempt": attempt,
                "started_at": started,
                "git_sha": sha,
            }

            try:
                with claim_timeout(CLAIM_TIMEOUT_S):
                    result = verify_svc.verify(
                        claim,
                        use_cache=True,
                        include_social=True,
                        enabled_source_groups=ALL_SOURCE_GROUPS,
                        deep_nli=True,
                    )
                    graph = builder.build(result)

                elapsed = time.monotonic() - cycle_start
                graph_path = graphs_dir / f"{claim_id}.json"
                graph_path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")

                meta = graph.metadata
                source_counts = dict(meta.retrieval_source_counts or {})
                flags = quota.update(source_counts)

                record.update({
                    "status": "ok",
                    "finished_at": utc_now_iso(),
                    "elapsed_s": round(elapsed, 1),
                    "verdict": meta.overall_verdict,
                    "confidence": meta.overall_confidence,
                    "num_sources": len(result.sources),
                    "total_nodes": meta.total_nodes,
                    "total_edges": meta.total_edges,
                    "support_node_count": meta.support_node_count,
                    "refute_node_count": meta.refute_node_count,
                    "factcheck_node_count": meta.factcheck_node_count,
                    "context_node_count": meta.context_node_count,
                    "retrieval_source_counts": source_counts,
                    "retrieval_notes": meta.retrieval_notes,
                    "graph_path": str(graph_path),
                    "quota_flags": flags,
                    "error_class": None,
                    "error_msg": None,
                })
                counts["ok"] += 1
                logger.info(
                    "    -> %s %.2f | %ds nodes=%d edges=%d | %.1fs%s",
                    meta.overall_verdict, meta.overall_confidence,
                    len(result.sources), meta.total_nodes, meta.total_edges, elapsed,
                    f" | FLAGS {flags}" if flags else "",
                )

            except GroqDailyQuotaExhausted as exc:
                # Not a claim failure — the run simply cannot proceed until the
                # token window rolls over. Record nothing terminal and retry
                # this same claim after the pause.
                manifest.append({**record, "status": "retryable", "finished_at": utc_now_iso(),
                                 "elapsed_s": round(time.monotonic() - cycle_start, 1),
                                 "error_class": "GroqDailyQuotaExhausted",
                                 "error_msg": str(exc)[:400], "quota_flags": ["groq_daily_quota"]})
                attempts[claim_id] += 1
                sleep_until_quota_reset(exc.reset_seconds, marker_path=paused_marker)
                continue

            except (Exception, ClaimTimeout) as exc:
                elapsed = time.monotonic() - cycle_start
                kind = classify_error(exc)
                if kind == "retryable" and attempt >= MAX_ATTEMPTS:
                    kind = "abandoned"
                record.update({
                    "status": kind,
                    "finished_at": utc_now_iso(),
                    "elapsed_s": round(elapsed, 1),
                    "error_class": type(exc).__name__,
                    "error_msg": str(exc)[:400],
                    "quota_flags": [],
                })
                counts[kind] += 1
                logger.warning("    -> %s: %s: %s", kind.upper(), type(exc).__name__, str(exc)[:160])

            manifest.append(record)
            attempts[claim_id] += 1

            if quota.is_degraded():
                dead = quota.dead_retrievers()
                logger.error(
                    "RUN DEGRADED — retrievers dead for %d+ consecutive claims: %s. Halting.",
                    25, dead,
                )
                degraded_marker.write_text(
                    f"halted_at={utc_now_iso()}\ndead_retrievers={dead}\n"
                    f"processed={n}/{len(todo)}\n"
                )
                return EXIT_DEGRADED

            slept = sleep_to_floor(cycle_start, args.pacing, record.get("elapsed_s", 0.0))
            if slept:
                logger.debug("    paced %.0fs", slept)

    except KeyboardInterrupt:
        logger.warning("Interrupted — progress is saved, re-run to resume.")
        return EXIT_ERROR
    finally:
        manifest.close()

    logger.info(
        "Phase 1 complete: ok=%d retryable=%d permanent=%d abandoned=%d",
        counts["ok"], counts["retryable"], counts["permanent"], counts["abandoned"],
    )
    # Non-zero only if work remains that a restart could still fix.
    return EXIT_OK if counts["retryable"] == 0 else EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

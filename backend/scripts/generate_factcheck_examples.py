"""Run the full verification pipeline for a fixed set of example claims
spanning different domains, and save graph JSON snapshots + a summary
for PDF report generation.
"""
import contextlib
import json
import signal
import socket
from pathlib import Path

from backend.app.services.graph_builder_service import GraphBuilderService
from backend.app.services.verify_claim_service import VerifyClaimService

# This sandbox's network occasionally hangs indefinitely on a socket connect/
# read (no client-level timeout kicks in). A global default timeout lets
# individual retriever calls fail fast and fall back gracefully instead of
# blocking the whole run.
socket.setdefaulttimeout(30)


class ClaimTimeout(Exception):
    pass


@contextlib.contextmanager
def claim_timeout(seconds: int):
    """Hard wall-clock timeout that can interrupt a stuck blocking syscall
    (e.g. an SSL handshake that ignores socket.setdefaulttimeout)."""

    def _handler(signum, frame):
        raise ClaimTimeout(f"claim exceeded {seconds}s timeout")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

CLAIMS = [
    ("Geography", "The Eiffel Tower is located in Paris, France."),
    ("Science", "Water boils at 100 degrees Celsius at sea level."),
    ("History", "Henri Christophe built a palace in Milot."),
    ("Politics", "Barack Obama was the 44th president of the United States."),
    ("Health / Myth", "Humans only use 10 percent of their brains."),
    ("Astronomy", "The Sun revolves around the Earth."),
    ("Technology", "The first iPhone was released in 2007."),
    ("Sports", "The FIFA World Cup is held every four years."),
    ("Economics", "Bitcoin has a maximum supply of 21 million coins."),
    ("Health / Nutrition", "Coffee is good for your health."),
]

OUT_DIR = Path("data/artifacts/graph_samples/pdf_examples")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building services (loading NLI + LLM models)...")
    verify_svc = VerifyClaimService.build_default()
    # Context expansion issues many extra retrieval sub-queries per claim and
    # is extremely slow under the rate-limited/flaky network in this run —
    # disable it for this report generation pass.
    verify_svc._context_expansion = None
    # DuckDuckGo's HTTP client repeatedly hangs on a slow SSL handshake under
    # this sandbox's network — drop it for this report generation pass.
    verify_svc._retrieval._registry._retrievers.pop("duckduckgo", None)
    builder = GraphBuilderService()

    summary = []

    for i, (domain, claim) in enumerate(CLAIMS, start=1):
        print(f"\n[{i}/{len(CLAIMS)}] ({domain}) {claim}")
        try:
            with claim_timeout(240):
                result = verify_svc.verify(claim, use_cache=True)
                graph = builder.build(result)

            safe_name = "".join(c if c.isalnum() else "_" for c in claim[:50]).strip("_")
            graph_path = OUT_DIR / f"{i:02d}_{safe_name}.json"
            graph_path.write_text(graph.model_dump_json(indent=2))

            print(
                f"  verdict={graph.metadata.overall_verdict} "
                f"conf={graph.metadata.overall_confidence:.2f} "
                f"sources={len(result.sources)} "
                f"nodes={graph.metadata.total_nodes} edges={graph.metadata.total_edges}"
            )

            summary.append(
                {
                    "index": i,
                    "domain": domain,
                    "claim": claim,
                    "verdict": graph.metadata.overall_verdict,
                    "confidence": graph.metadata.overall_confidence,
                    "graph_path": str(graph_path),
                    "num_sources": len(result.sources),
                    "retrieval_notes": graph.metadata.retrieval_notes,
                }
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            summary.append(
                {
                    "index": i,
                    "domain": domain,
                    "claim": claim,
                    "error": str(exc),
                }
            )

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[DONE] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

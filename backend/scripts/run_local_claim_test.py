import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from backend.app.retrieval.factcheck_retriever import FactCheckRetriever
from backend.app.retrieval.gdelt_retriever import GDELTRetriever
from backend.app.retrieval.guardian_retriever import GuardianRetriever
from backend.app.retrieval.livewiki_retriever import LiveWikiRetriever
from backend.app.retrieval.newsapi_retriever import NewsApiRetriever
from backend.app.retrieval.retriever_registry import RetrieverRegistry
from backend.app.retrieval.wikipedia_retriever import WikipediaRetriever
from backend.app.models.nli_model import NLIModel
from backend.app.services.cache_service import CacheService
from backend.app.services.evidence_expansion_service import EvidenceExpansionService
from backend.app.services.ranking_service import RankingService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.stance_service import StanceService
from backend.app.utils.constants import DEFAULT_SNIPPET_MAX_CHARS
from backend.app.utils.constants import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RETRIEVAL_CACHE_DIR,
    DEFAULT_RETRIEVER_MAX_RESULTS,
    FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME,
    NLI_CONFIRM_MODEL_NAME,
    NLI_MODEL_NAME,
)

# Fixed benchmark claims for --mode orchestrate
BENCHMARK_CLAIMS = [
    "The Eiffel Tower is located in Paris, France.",
    "The Great Wall of China is visible from space with the naked eye.",
    "Henri Christophe built a palace in Milot.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local retriever smoke test for a single claim or full orchestration."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "orchestrate", "snippet", "nli"],
        help=(
            "single: test one retriever against --query (original behavior). "
            "orchestrate: run full RetrievalService against 3 benchmark claims. "
            "nli: run full pipeline including NLI stance classification on --query."
        ),
    )

    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Claim or query text to test (required for --mode single).",
    )

    parser.add_argument(
        "--retriever",
        type=str,
        default="wikipedia",
        choices=["wikipedia", "factcheck", "guardian", "newsapi", "gdelt", "livewiki"],
        help="Retriever to test (used only for --mode single).",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_RETRIEVER_MAX_RESULTS,
        help="Maximum number of normalized sources to return.",
    )

    parser.add_argument(
        "--evidence-snippets-path",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR / FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME),
        help="Path to processed fever_evidence_snippets parquet.",
    )

    parser.add_argument(
        "--factcheck-api-key",
        type=str,
        default="",
        help="Optional explicit Google Fact Check Tools API key. If omitted, loads FACTCHECK_API_KEY from .env/environment.",
    )

    parser.add_argument(
        "--factcheck-language-code",
        type=str,
        default="en-US",
        help="Language code for Fact Check Tools API.",
    )
    
    parser.add_argument(
        "--guardian-api-key",
        type=str,
        default="",
        help="Optional explicit Guardian API key. Falls back to GUARDIAN_API_KEY from environment.",
    )

    parser.add_argument(
        "--guardian-section",
        type=str,
        default="",
        help="Optional Guardian section filter like business, world, politics, technology.",
    )
    
    parser.add_argument(
        "--newsapi-api-key",
        type=str,
        default="",
        help="Optional explicit NewsAPI key. Falls back to NEWSAPI_KEY from environment.",
    )

    parser.add_argument(
        "--newsapi-language",
        type=str,
        default="en",
        help="Language for NewsAPI.",
    )

    parser.add_argument(
        "--newsapi-sort-by",
        type=str,
        default="relevancy",
        choices=["relevancy", "popularity", "publishedAt"],
        help="NewsAPI sort order.",
    )
    
    parser.add_argument(
        "--gdelt-timespan",
        type=str,
        default="30d",
        help="GDELT timespan like 24h, 7d, 30d, 3months.",
    )

    parser.add_argument(
        "--nli-model",
        type=str,
        default=NLI_MODEL_NAME,
        help="HuggingFace model name for NLI fast model.",
    )

    parser.add_argument(
        "--use-cascade",
        action="store_true",
        default=False,
        help=(
            "Enable cascade NLI: fast model pre-filters, strong model confirms "
            "supports/refutes. Uses NLI_CONFIRM_MODEL_NAME from constants."
        ),
    )

    return parser.parse_args()


def build_registry(args: argparse.Namespace) -> RetrieverRegistry:
    load_dotenv()

    registry = RetrieverRegistry()

    wikipedia_retriever = WikipediaRetriever(
        evidence_snippets_path=Path(args.evidence_snippets_path)
    )
    registry.register(wikipedia_retriever)

    factcheck_api_key = args.factcheck_api_key.strip() or os.getenv("FACTCHECK_API_KEY", "").strip()
    factcheck_retriever = FactCheckRetriever(
        api_key=factcheck_api_key or None,
        language_code=args.factcheck_language_code,
    )
    registry.register(factcheck_retriever)
    
    guardian_api_key = args.guardian_api_key.strip() or os.getenv("GUARDIAN_API_KEY", "").strip()
    guardian_section = args.guardian_section.strip() or None
    guardian_retriever = GuardianRetriever(
        api_key=guardian_api_key or None,
        section=guardian_section,
    )
    registry.register(guardian_retriever)
    
    newsapi_api_key = args.newsapi_api_key.strip() or os.getenv("NEWSAPI_KEY", "").strip()
    newsapi_retriever = NewsApiRetriever(
        api_key=newsapi_api_key or None,
        language=args.newsapi_language,
        sort_by=args.newsapi_sort_by,
    )
    registry.register(newsapi_retriever)
    
    gdelt_retriever = GDELTRetriever(
        timespan=args.gdelt_timespan,
    )
    registry.register(gdelt_retriever)

    livewiki_retriever = LiveWikiRetriever()
    registry.register(livewiki_retriever)

    return registry


def print_results(query: str, retriever_name: str, results) -> None:
    print("\n" + "=" * 80)
    print("LOCAL CLAIM TEST")
    print("=" * 80)
    print(f"Retriever: {retriever_name}")
    print(f"Query: {query}")
    print(f"Results returned: {len(results)}")
    print("=" * 80)

    if not results:
        print("No results found.\n")
        return

    for i, source in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("-" * 80)
        print(f"source_id:       {source.source_id}")
        print(f"source_type:     {source.source_type}")
        print(f"title:           {source.title}")
        print(f"url:             {source.url}")
        print(f"publisher:       {source.publisher}")
        print(f"trust_score:     {source.trust_score}")
        print(f"relevance_score: {source.relevance_score}")
        print(f"stance_hint:     {source.stance_hint}")
        print(f"snippet:         {source.snippet}")

    print("\n[DONE] Local claim test completed.\n")


def print_orchestrate_results(claim: str, results, cache_hit: bool) -> None:
    print("\n" + "=" * 80)
    print(f"CLAIM: {claim}")
    print("=" * 80)
    print(f"Total sources returned : {len(results)}")
    print(f"Cache hit              : {cache_hit}")

    # Source breakdown by type
    type_counts: dict = {}
    for s in results:
        type_counts[s.source_type] = type_counts.get(s.source_type, 0) + 1
    print(f"Source breakdown       : {type_counts}")

    # Top 3
    print("\nTop 3 sources:")
    for i, s in enumerate(results[:3], start=1):
        print(f"  [{i}] {s.title}")
        print(f"       type={s.source_type}  trust={s.trust_score:.2f}  relevance={s.relevance_score:.2f}")
        print(f"       url={s.url}")


def _build_service(args: argparse.Namespace) -> RetrievalService:
    """Build a fully wired RetrievalService with Phase 6 expansion."""
    registry = build_registry(args)
    cache = CacheService(DEFAULT_RETRIEVAL_CACHE_DIR)
    ranking = RankingService()
    expansion = EvidenceExpansionService()
    return RetrievalService(registry=registry, cache=cache, ranking=ranking, expansion=expansion)


def run_snippet(args: argparse.Namespace) -> None:
    """Run RetrievalService on --query and inspect snippet quality."""
    if not args.query.strip():
        print("[ERROR] --query is required for --mode snippet.")
        sys.exit(1)

    service = _build_service(args)
    results = service.retrieve(args.query, max_results=args.max_results, use_cache=False)

    print("\n" + "=" * 80)
    print("PHASE 6 SNIPPET INSPECTION")
    print("=" * 80)
    print(f"Query      : {args.query}")
    print(f"Results    : {len(results)}")
    print(f"Max chars  : {DEFAULT_SNIPPET_MAX_CHARS}")
    print("=" * 80)

    issues = []
    for i, s in enumerate(results, start=1):
        snip = s.snippet or ""
        over_limit = len(snip) > DEFAULT_SNIPPET_MAX_CHARS
        empty = not snip.strip()
        flag = " [EMPTY]" if empty else (" [OVER LIMIT]" if over_limit else "")
        print(f"\n[{i}] {s.title}{flag}")
        print(f"     type={s.source_type}  len={len(snip)}")
        print(f"     snippet: {snip[:200]}")
        if empty or over_limit:
            issues.append(f"Source {s.source_id}: {flag.strip()}")

    print("\n" + "=" * 80)
    if issues:
        print(f"[WARN] {len(issues)} issue(s) found:")
        for iss in issues:
            print(f"  - {iss}")
    else:
        print("[PASS] All snippets are non-empty and within char limit.")
    print("=" * 80 + "\n")


def run_orchestrate(args: argparse.Namespace) -> None:
    """Run RetrievalService against all 3 benchmark claims and print summary."""
    service = _build_service(args)

    print("\n" + "=" * 80)
    print("PHASE 6 ORCHESTRATION SMOKE TEST")
    print("=" * 80)
    print(f"Benchmark claims : {len(BENCHMARK_CLAIMS)}")
    print(f"Max results      : {args.max_results}")
    print(f"Cache dir        : {DEFAULT_RETRIEVAL_CACHE_DIR}")

    all_passed = True

    for claim in BENCHMARK_CLAIMS:
        try:
            results = service.retrieve(claim, max_results=args.max_results)
            results_cached = service.retrieve(claim, max_results=args.max_results)
            cache_hit = len(results_cached) == len(results)

            print_orchestrate_results(claim, results, cache_hit)

            if not results:
                print("  [WARN] No results returned for this claim.")
        except Exception as exc:
            print(f"\n[ERROR] Claim failed: {claim!r}\n  {exc}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("[PASS] All benchmark claims completed without errors.")
    else:
        print("[FAIL] One or more benchmark claims raised errors (see above).")
    print("=" * 80 + "\n")


def run_nli(args: argparse.Namespace) -> None:
    """Run full pipeline including NLI stance classification on --query."""
    if not args.query.strip():
        print("[ERROR] --query is required for --mode nli.")
        sys.exit(1)

    cascade = args.use_cascade

    print("\n" + "=" * 80)
    print("PHASE 7 NLI STANCE CLASSIFICATION")
    print("=" * 80)
    print(f"Query        : {args.query}")
    print(f"Fast model   : {args.nli_model}")
    print(f"Cascade mode : {cascade}" + (f" ({NLI_CONFIRM_MODEL_NAME})" if cascade else ""))
    print(f"Max results  : {args.max_results}")
    print("Loading model(s)...")
    print("=" * 80)

    service = _build_service(args)
    sources = service.retrieve(args.query, max_results=args.max_results, use_cache=False)

    nli_model = NLIModel(model_name=args.nli_model)
    confirm_model = NLIModel(model_name=NLI_CONFIRM_MODEL_NAME) if cascade else None
    stance_svc = StanceService(
        model=nli_model,
        cache=CacheService(DEFAULT_RETRIEVAL_CACHE_DIR),
        confirm_model=confirm_model,
    )

    classified = stance_svc.classify(args.query, sources)

    counts: dict = {"supports": 0, "refutes": 0, "insufficient": 0, "none": 0}

    for i, source in enumerate(classified, start=1):
        stance = source.stance_hint or "none"
        counts[stance] = counts.get(stance, 0) + 1
        snip = (source.snippet or "")[:150]

        print(f"\n[{i}] {source.title}")
        print(f"     type={source.source_type}  stance={stance}")
        print(f"     snippet: {snip}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  supports    : {counts.get('supports', 0)}")
    print(f"  refutes     : {counts.get('refutes', 0)}")
    print(f"  insufficient: {counts.get('insufficient', 0)}")
    print("=" * 80 + "\n")


def main() -> None:
    args = parse_args()

    if args.mode == "orchestrate":
        run_orchestrate(args)
        return

    if args.mode == "snippet":
        run_snippet(args)
        return

    if args.mode == "nli":
        run_nli(args)
        return

    # --- mode == "single" (original behavior) ---
    if not args.query.strip():
        print("[ERROR] --query is required for --mode single.")
        sys.exit(1)

    registry = build_registry(args)
    retriever = registry.get(args.retriever)

    results = retriever.retrieve(
        query=args.query,
        max_results=args.max_results,
    )

    print_results(
        query=args.query,
        retriever_name=args.retriever,
        results=results,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

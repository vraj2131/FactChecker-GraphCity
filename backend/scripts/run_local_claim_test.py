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
from backend.app.models.llm_model import LLMModel, GroqLLMModel, get_llm_model, get_groq_llm_model
from backend.app.models.nli_model import NLIModel
from backend.app.services.cache_service import CacheService
from backend.app.services.confidence_service import ConfidenceService
from backend.app.services.evidence_expansion_service import EvidenceExpansionService
from backend.app.services.graph_builder_service import GraphBuilderService
from backend.app.services.ranking_service import RankingService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.stance_service import StanceService
from backend.app.services.verify_claim_service import VerifyClaimService
from backend.app.utils.constants import DEFAULT_SNIPPET_MAX_CHARS
from backend.app.utils.constants import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RETRIEVAL_CACHE_DIR,
    DEFAULT_RETRIEVER_MAX_RESULTS,
    FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME,
    GROQ_MODEL_NAME,
    LLM_FALLBACK_MODEL_NAME,
    LLM_MAX_INPUT_SOURCES,
    LLM_MODEL_NAME,
    NLI_CONFIRM_MODEL_NAME,
    NLI_MODEL_NAME,
)

# Fixed benchmark claims for --mode orchestrate
BENCHMARK_CLAIMS = [
    "The Eiffel Tower is located in Paris, France.",
    "The Great Wall of China is visible from space with the naked eye.",
    "Henri Christophe built a palace in Milot.",
]

# Confidence benchmark claims for --mode confidence
CONFIDENCE_BENCHMARK = [
    # Verified (high confidence expected)
    ("The Eiffel Tower is located in Paris, France.", "verified"),
    ("Water boils at 100 degrees Celsius at sea level.", "verified"),
    ("Barack Obama was the 44th president of the United States.", "verified"),
    # Rejected (high confidence expected)
    ("The Great Wall of China is visible from space with the naked eye.", "rejected"),
    ("Humans only use 10 percent of their brains.", "rejected"),
    # NEI (low confidence expected)
    ("A secret underground city exists beneath Denver Airport.", "not_enough_info"),
    ("The number of cats in Tokyo exceeded 5 million in 2025.", "not_enough_info"),
    # Mixed/contested
    ("Coffee is good for your health.", "not_enough_info"),
    ("Einstein failed math in school.", "rejected"),
    # Specific/historical
    ("Henri Christophe built a palace in Milot.", "verified"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local retriever smoke test for a single claim or full orchestration."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "orchestrate", "snippet", "nli", "llm", "confidence", "graph"],
        help=(
            "single: test one retriever against --query (original behavior). "
            "orchestrate: run full RetrievalService against 3 benchmark claims. "
            "nli: run full pipeline including NLI stance classification on --query. "
            "llm: run full pipeline including LLM source classification on --query. "
            "confidence: run 10 benchmark claims through full pipeline + confidence scoring. "
            "graph: run full pipeline + graph builder, save JSON to data/artifacts/graph_samples/."
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

    parser.add_argument(
        "--llm-model",
        type=str,
        default=LLM_MODEL_NAME,
        help=(
            f"Primary HuggingFace model for LLM classification (default: {LLM_MODEL_NAME}). "
            f"Automatically falls back to {LLM_FALLBACK_MODEL_NAME} if the primary fails to load. "
            "Ignored when --llm-backend groq is set."
        ),
    )

    parser.add_argument(
        "--llm-backend",
        type=str,
        default="groq",
        choices=["local", "groq"],
        help=(
            "local: run LLM inference on-device (Qwen/Llama via HuggingFace). "
            "groq: use Groq cloud API — instant inference, requires GROQ_API_KEY in .env. "
            f"Groq default model: {GROQ_MODEL_NAME}."
        ),
    )

    parser.add_argument(
        "--groq-model",
        type=str,
        default=GROQ_MODEL_NAME,
        help=f"Groq model ID (default: {GROQ_MODEL_NAME}). Only used with --llm-backend groq.",
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

    classified, _nli_results = stance_svc.classify(args.query, sources)

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


def run_llm(args: argparse.Namespace) -> None:
    """Run full pipeline (retrieval → NLI → LLM) and print per-source classifications."""
    if not args.query.strip():
        print("[ERROR] --query is required for --mode llm.")
        sys.exit(1)

    cascade = args.use_cascade
    backend = args.llm_backend

    print("\n" + "=" * 80)
    print("PHASE 8 LLM SOURCE CLASSIFICATION")
    print("=" * 80)
    print(f"Query        : {args.query}")
    print(f"Backend      : {backend.upper()}")
    if backend == "groq":
        print(f"Groq model   : {args.groq_model}")
    else:
        print(f"LLM model    : {args.llm_model}")
    print(f"NLI model    : {args.nli_model}")
    print(f"Cascade NLI  : {cascade}")
    print(f"Max results  : {args.max_results}")
    print(f"LLM sources  : {LLM_MAX_INPUT_SOURCES}")
    print("Loading models...")
    print("=" * 80)

    # 1. Retrieve sources
    service = _build_service(args)
    sources = service.retrieve(args.query, max_results=args.max_results, use_cache=False)
    print(f"\n[Retrieval] {len(sources)} sources retrieved.")

    if not sources:
        print("[WARN] No sources retrieved — cannot run LLM classification.")
        return

    # 2. NLI stance classification
    nli_model = NLIModel(model_name=args.nli_model)
    confirm_model = NLIModel(model_name=NLI_CONFIRM_MODEL_NAME) if cascade else None
    stance_svc = StanceService(
        model=nli_model,
        cache=CacheService(DEFAULT_RETRIEVAL_CACHE_DIR),
        confirm_model=confirm_model,
    )
    classified, nli_results = stance_svc.classify(args.query, sources)
    print(f"[NLI] Stance classification complete.")

    # 3. Slice to LLM_MAX_INPUT_SOURCES
    llm_input = classified[:LLM_MAX_INPUT_SOURCES]
    print(f"[LLM] Sending {len(llm_input)} sources to {backend.upper()}...")

    # 4. LLM classification — local or Groq
    if backend == "groq":
        llm = get_groq_llm_model(model_name=args.groq_model)
    else:
        llm = get_llm_model(model_name=args.llm_model, fallback_model_name=LLM_FALLBACK_MODEL_NAME)
    result = llm.classify(args.query, llm_input)

    # 5. Print per-source table
    print("\n" + "=" * 80)
    print("PER-SOURCE CLASSIFICATIONS")
    print("=" * 80)
    print(f"{'[i]':<5} {'classification':<22} {'nli_hint':<14} {'type':<12} title")
    print("-" * 80)

    # Build a lookup from index → classification
    class_by_index = {sc.index: sc for sc in result.sources}

    for i, source in enumerate(llm_input, start=1):
        sc = class_by_index.get(i)
        classification = sc.classification if sc else "insufficient"
        rationale = sc.rationale if sc else ""
        nli_hint = source.stance_hint or "none"
        title = source.title[:45] if source.title else ""
        print(f"[{i}]   {classification:<22} {nli_hint:<14} {source.source_type:<12} {title}")
        if rationale:
            print(f"       rationale: {rationale}")

    # 6. Print overall verdict
    print("\n" + "=" * 80)
    print("OVERALL VERDICT")
    print("=" * 80)
    print(f"  verdict      : {result.overall_verdict}")
    print(f"  confidence   : {result.confidence:.2f}")
    print(f"  best source  : [{result.best_source_index}]")
    print(f"  explanation  : {result.short_explanation}")
    print("=" * 80 + "\n")


def run_confidence(args: argparse.Namespace) -> None:
    """Run 10 benchmark claims through full pipeline + confidence scoring."""
    backend = args.llm_backend

    claims = CONFIDENCE_BENCHMARK
    # If --query is given, run a single claim instead
    if args.query.strip():
        claims = [(args.query.strip(), "unknown")]

    print("\n" + "=" * 80)
    print("PHASE 9 CONFIDENCE BENCHMARK")
    print("=" * 80)
    print(f"Backend      : {backend.upper()}")
    print(f"Claims       : {len(claims)}")
    print("Loading models...")
    print("=" * 80)

    # Build services
    service = _build_service(args)
    nli_model = NLIModel(model_name=args.nli_model)
    confirm_model = NLIModel(model_name=NLI_CONFIRM_MODEL_NAME) if args.use_cascade else None
    stance_svc = StanceService(
        model=nli_model,
        cache=CacheService(DEFAULT_RETRIEVAL_CACHE_DIR),
        confirm_model=confirm_model,
    )
    if backend == "groq":
        llm = get_groq_llm_model(model_name=args.groq_model)
    else:
        llm = get_llm_model(model_name=args.llm_model, fallback_model_name=LLM_FALLBACK_MODEL_NAME)
    confidence_svc = ConfidenceService()

    results_summary = []

    for claim_text, expected in claims:
        print(f"\n{'='*80}")
        print(f"CLAIM: {claim_text}")
        print(f"{'='*80}")
        print(f"  Expected       : {expected}")

        # 1. Retrieve
        sources = service.retrieve(claim_text, max_results=args.max_results, use_cache=False)
        if not sources:
            print("  [WARN] No sources retrieved — skipping.")
            results_summary.append((claim_text, expected, "skip", 0.0, False))
            continue

        # 2. NLI
        classified, nli_results = stance_svc.classify(claim_text, sources)

        # 3. LLM
        llm_input = classified[:LLM_MAX_INPUT_SOURCES]
        llm_result = llm.classify(claim_text, llm_input)
        print(f"  LLM verdict    : {llm_result.overall_verdict:<14} (conf={llm_result.confidence:.2f})")

        # 4. Confidence scoring
        conf = confidence_svc.compute_main_confidence(llm_result, llm_input, nli_results)

        print(f"  SUB-SCORES:")
        print(f"    directional  : {conf.debug.get('directional', 0):.2f}  "
              f"(support={conf.support_score:.2f}, refute={conf.refute_score:.2f})")
        print(f"    llm_conf     : {conf.debug.get('llm_conf', 0):.2f}")
        print(f"    quality      : {conf.evidence_quality:.2f}")
        print(f"    corroboration: {conf.corroboration:.2f}  ({len(conf.debug.get('agreeing_types', []))} types)")
        print(f"    coverage     : {conf.coverage:.2f}")
        print(f"  RAW -> CALIBRATED: {conf.raw_confidence:.2f} -> {conf.overall_confidence:.2f}")
        print(f"  FINAL          : {conf.overall_verdict} ({conf.overall_confidence:.2f})")

        # 5. Edge confidences
        class_by_idx = {sc.index: sc for sc in llm_result.sources}
        print(f"\n  EDGE CONFIDENCES:")
        for i, src in enumerate(llm_input, start=1):
            sc = class_by_idx.get(i)
            llm_class = sc.classification if sc else "insufficient"
            nli = nli_results.get(i - 1)  # 0-indexed in nli_results
            edge = confidence_svc.compute_edge_confidence(src, llm_class, nli)
            marker = "  <-- correctly lower" if llm_class == "correlated_context" else ""
            print(f"    [{i}] {llm_class:<22} edge={edge:.2f}  {src.source_type:<12}{marker}")

        # Check pass/fail
        passed = True
        if expected != "unknown":
            passed = conf.overall_verdict == expected
            if conf.overall_confidence == 0.0 or conf.overall_confidence == 1.0:
                passed = False
            label = "[PASS]" if passed else "[FAIL]"
            print(f"  {label} {'verdict matches' if passed else 'MISMATCH: got ' + conf.overall_verdict}")
        results_summary.append((claim_text, expected, conf.overall_verdict, conf.overall_confidence, passed))

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'#':<3} {'Expected':<16} {'Actual':<16} {'Conf':<8} {'Pass':<6} Claim")
    print("-" * 80)
    total_pass = 0
    for i, (claim_text, expected, actual, conf_val, passed) in enumerate(results_summary, 1):
        if passed:
            total_pass += 1
        status = "PASS" if passed else ("SKIP" if actual == "skip" else "FAIL")
        print(f"{i:<3} {expected:<16} {actual:<16} {conf_val:<8.2f} {status:<6} {claim_text[:50]}")

    print(f"\nTotal: {total_pass}/{len(results_summary)} passed")
    print("=" * 80 + "\n")


def run_graph(args: argparse.Namespace) -> None:
    """Run full pipeline + graph builder on --query, save JSON to graph_samples/."""
    claim = args.query.strip()
    if not claim:
        print("[ERROR] --query is required for --mode graph.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("PHASE 10 GRAPH BUILDER")
    print("=" * 80)
    print(f"Claim   : {claim}")
    print("Building services...")

    verify_svc = VerifyClaimService.build_default(groq_model=args.groq_model)
    builder = GraphBuilderService()

    print("Running pipeline...")
    result = verify_svc.verify(claim, use_cache=True)

    print(f"  Sources retrieved : {len(result.sources)}")
    print(f"  LLM verdict       : {result.llm_result.overall_verdict} (conf={result.llm_result.confidence:.2f})")
    print(f"  Final verdict     : {result.confidence_output.overall_verdict} ({result.confidence_output.overall_confidence:.2f})")

    graph = builder.build(result)

    print("\n" + "-" * 80)
    print("GRAPH OUTPUT")
    print("-" * 80)
    print(f"  Nodes : {graph.metadata.total_nodes}  "
          f"(support={graph.metadata.support_node_count}, "
          f"refute={graph.metadata.refute_node_count}, "
          f"context={graph.metadata.context_node_count}, "
          f"factcheck={graph.metadata.factcheck_node_count}, "
          f"insufficient={graph.metadata.insufficient_node_count})")
    print(f"  Edges : {graph.metadata.total_edges}")
    print(f"  Verdict: {graph.metadata.overall_verdict} ({graph.metadata.overall_confidence:.2f})")

    # Verify node fields
    print("\n  NODE FIELD CHECK:")
    issues = []
    for node in graph.nodes:
        if not node.node_id:
            issues.append(f"  [WARN] node missing node_id")
        if not node.color:
            issues.append(f"  [WARN] {node.node_id} missing color")
        if node.best_source_url is None and node.is_main_claim:
            issues.append(f"  [WARN] main node missing best_source_url")
        marker = " ← MAIN" if node.is_main_claim else ""
        print(f"    {node.node_id:<16} type={node.node_type:<22} verdict={node.verdict:<12} "
              f"conf={node.confidence:.2f}  color={node.color}  "
              f"url={'YES' if node.best_source_url else 'MISSING'}{marker}")

    print("\n  EDGE FIELD CHECK:")
    for edge in graph.edges:
        print(f"    {edge.source} → {edge.target:<16} type={edge.edge_type:<12} "
              f"weight={edge.weight:.2f}  width={edge.width:.1f}  "
              f"dashed={str(edge.dashed):<5}  color={edge.color}")

    if issues:
        print("\n  ISSUES:")
        for iss in issues:
            print(f"  {iss}")
    else:
        print("\n  [OK] All node/edge fields complete.")

    # Node type matches edge type check
    node_type_map = {n.node_id: n.node_type for n in graph.nodes}
    edge_ok = True
    for edge in graph.edges:
        target_type = node_type_map.get(edge.target, "")
        expected = {
            "direct_support": "supports",
            "direct_refute": "refutes",
            "factcheck_review": "refutes",
            "context_signal": "correlated",
            "insufficient_evidence": "insufficient",
        }.get(target_type, "insufficient")
        if edge.edge_type != expected:
            print(f"  [WARN] {edge.target}: node_type={target_type} but edge_type={edge.edge_type} (expected {expected})")
            edge_ok = False
    if edge_ok:
        print("  [OK] All node types match edge types.")

    # Save JSON
    out_dir = Path("data/artifacts/graph_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in claim[:40]).strip("_")
    out_path = out_dir / f"graph_{safe_name}.json"
    out_path.write_text(graph.model_dump_json(indent=2))
    print(f"\n  [SAVED] {out_path}")
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

    if args.mode == "llm":
        run_llm(args)
        return

    if args.mode == "confidence":
        run_confidence(args)
        return

    if args.mode == "graph":
        run_graph(args)
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
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)

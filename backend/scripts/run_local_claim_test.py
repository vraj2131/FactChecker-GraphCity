import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from backend.app.retrieval.factcheck_retriever import FactCheckRetriever
from backend.app.retrieval.gdelt_retriever import GDELTRetriever
from backend.app.retrieval.newsapi_retriever import NewsApiRetriever
from backend.app.retrieval.guardian_retriever import GuardianRetriever
from backend.app.retrieval.retriever_registry import RetrieverRegistry
from backend.app.retrieval.wikipedia_retriever import WikipediaRetriever
from backend.app.utils.constants import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RETRIEVER_MAX_RESULTS,
    FEVER_EVIDENCE_SNIPPETS_OUTPUT_FILENAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local retriever smoke test for a single claim."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Claim or query text to test.",
    )

    parser.add_argument(
        "--retriever",
        type=str,
        default="wikipedia",
        choices=["wikipedia", "factcheck", "guardian", "newsapi", "gdelt"],
        help="Retriever to test.",
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


def main() -> None:
    args = parse_args()

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

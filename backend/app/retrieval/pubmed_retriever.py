import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import DEFAULT_RETRIEVER_TIMEOUT_SECONDS, SOURCE_NAME_PUBMED

logger = logging.getLogger(__name__)

TRUST_SCORE_PUBMED = 0.92
_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_HEADERS = {"User-Agent": "FactCheckerBot/1.0 (mailto:factcheck@research.edu)"}


def _fetch_abstracts(pmids: List[str]) -> Dict[str, Dict]:
    """Fetch titles + abstracts for a list of PubMed IDs. Returns {pmid: {title, abstract, doi}}."""
    if not pmids:
        return {}
    try:
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        resp = requests.get(
            _EFETCH_URL,
            params=params,
            headers=_HEADERS,
            timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:
        logger.warning("PubMedRetriever: efetch failed: %s", exc)
        return {}

    results: Dict[str, Dict] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
        if not pmid:
            continue

        title_el = article.find(".//ArticleTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""

        abstract_parts = [
            (el.text or "").strip()
            for el in article.findall(".//AbstractText")
            if el.text
        ]
        abstract = " ".join(abstract_parts)[:500]

        doi = ""
        for aid in article.findall(".//ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip()
                break

        pub_year = ""
        year_el = article.find(".//PubDate/Year")
        if year_el is not None and year_el.text:
            pub_year = year_el.text.strip()

        results[pmid] = {
            "title": title,
            "abstract": abstract,
            "doi": doi,
            "year": pub_year,
        }
    return results


class PubMedRetriever(BaseRetriever):
    """
    Retrieves peer-reviewed biomedical abstracts from PubMed (NCBI E-utilities).
    No API key required (optional key raises rate limits).
    """

    def __init__(self) -> None:
        super().__init__(source_name=SOURCE_NAME_PUBMED)

    def fetch_raw(self, query: str, max_results: int = 5, **kwargs: Any) -> List[str]:
        """Returns a list of PubMed IDs matching the query."""
        try:
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": min(max_results, 10),
                "retmode": "json",
                "sort": "relevance",
            }
            resp = requests.get(
                _ESEARCH_URL,
                params=params,
                headers=_HEADERS,
                timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            logger.warning("PubMedRetriever: esearch failed: %s", exc)
            return []

    def normalize(
        self, raw_data: Any, query: str, max_results: int = 5, **kwargs: Any
    ) -> List[Source]:
        pmids: List[str] = raw_data or []
        if not pmids:
            return []

        metadata = _fetch_abstracts(pmids[:max_results])
        sources: List[Source] = []

        for pmid in pmids[:max_results]:
            meta = metadata.get(pmid, {})
            title = meta.get("title") or f"PubMed Article {pmid}"
            abstract = meta.get("abstract") or ""
            doi = meta.get("doi") or ""
            year = meta.get("year") or None

            if doi:
                url = f"https://doi.org/{doi}" if not doi.startswith("http") else doi
            else:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            snippet = abstract if abstract else f"PubMed PMID {pmid}: {title}"

            sources.append(
                Source(
                    source_id=f"pubmed::{pmid}",
                    source_type=SOURCE_NAME_PUBMED,
                    title=title[:500],
                    url=url,
                    publisher="PubMed / NCBI",
                    snippet=snippet,
                    published_at=year,
                    trust_score=TRUST_SCORE_PUBMED,
                    relevance_score=0.8,
                    stance_hint=None,
                )
            )
        return self.postprocess(sources, max_results)

import logging
import re
from typing import Any, Dict, List, Optional

import requests

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import DEFAULT_RETRIEVER_TIMEOUT_SECONDS, SOURCE_NAME_COINGECKO

logger = logging.getLogger(__name__)

TRUST_SCORE_COINGECKO = 0.88
_SEARCH_URL = "https://api.coingecko.com/api/v3/search"
_COIN_URL = "https://api.coingecko.com/api/v3/coins/{id}"
_HEADERS = {"Accept": "application/json"}

# Known coin names / symbols to extract from claim text
_KNOWN_COINS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "ripple", "xrp",
    "cardano", "ada", "dogecoin", "doge", "litecoin", "ltc", "polkadot", "dot",
    "chainlink", "link", "avalanche", "avax", "uniswap", "uni", "binance", "bnb",
    "tether", "usdt", "usdc", "monero", "xmr", "stellar", "xlm", "tron", "trx",
    "shiba", "shib", "matic", "polygon", "cosmos", "atom", "near", "filecoin", "fil",
]


def _extract_coin_query(text: str) -> str:
    """Extract the most specific coin name/symbol from claim text for CoinGecko search."""
    lower = text.lower()
    for coin in _KNOWN_COINS:
        if re.search(rf"\b{re.escape(coin)}\b", lower):
            return coin
    # Fallback: take the first word that looks like a ticker (2-5 uppercase letters)
    ticker = re.search(r"\b([A-Z]{2,5})\b", text)
    if ticker:
        return ticker.group(1)
    # Last resort: first non-stopword token
    tokens = [w for w in text.split() if len(w) > 3]
    return tokens[0] if tokens else text


def _fmt_number(n: Optional[float]) -> str:
    if n is None:
        return "N/A"
    if n >= 1_000_000_000_000:
        return f"${n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    return f"${n:,.2f}"


def _search_coins(query: str, max_results: int) -> List[Dict]:
    """Search CoinGecko for coin IDs matching the query."""
    try:
        resp = requests.get(
            _SEARCH_URL,
            params={"query": query},
            headers=_HEADERS,
            timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json().get("coins", [])[:max_results]
    except Exception as exc:
        logger.warning("CoinGeckoRetriever: search failed: %s", exc)
        return []


def _fetch_coin(coin_id: str) -> Optional[Dict]:
    """Fetch detailed market data for a single coin."""
    try:
        resp = requests.get(
            _COIN_URL.format(id=coin_id),
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
            },
            headers=_HEADERS,
            timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("CoinGeckoRetriever: coin fetch for %s failed: %s", coin_id, exc)
        return None


class CoinGeckoRetriever(BaseRetriever):
    """
    Retrieves live crypto market data from CoinGecko (free tier, no API key required).
    Produces fact-card snippets: price, market cap, supply.
    """

    def __init__(self) -> None:
        super().__init__(source_name=SOURCE_NAME_COINGECKO)

    def fetch_raw(self, query: str, max_results: int = 3, **kwargs: Any) -> List[Dict]:
        coin_query = _extract_coin_query(query)
        coins = _search_coins(coin_query, max_results)
        results = []
        for coin in coins:
            coin_id = coin.get("id")
            if not coin_id:
                continue
            data = _fetch_coin(coin_id)
            if data:
                results.append(data)
        return results

    def normalize(
        self, raw_data: Any, query: str, max_results: int = 3, **kwargs: Any
    ) -> List[Source]:
        if not raw_data:
            return []
        sources: List[Source] = []
        for coin in raw_data:
            name = coin.get("name") or "Unknown"
            symbol = (coin.get("symbol") or "").upper()
            coin_id = coin.get("id") or name.lower()
            md = coin.get("market_data") or {}

            price_usd = (md.get("current_price") or {}).get("usd")
            market_cap = (md.get("market_cap") or {}).get("usd")
            circ_supply = md.get("circulating_supply")
            max_supply = md.get("max_supply")
            total_supply = md.get("total_supply")
            change_24h = md.get("price_change_percentage_24h")

            parts = [f"{name} ({symbol}) — Live CoinGecko data:"]
            if price_usd is not None:
                parts.append(f"Price: {_fmt_number(price_usd)} USD.")
            if market_cap:
                parts.append(f"Market cap: {_fmt_number(market_cap)}.")
            if circ_supply is not None:
                parts.append(f"Circulating supply: {circ_supply:,.0f} {symbol}.")
            if max_supply is not None:
                parts.append(f"Max supply: {max_supply:,.0f} {symbol}.")
            elif total_supply is not None:
                parts.append(f"Total supply: {total_supply:,.0f} {symbol}.")
            if change_24h is not None:
                parts.append(f"24h change: {change_24h:+.2f}%.")
            snippet = " ".join(parts)

            url = f"https://www.coingecko.com/en/coins/{coin_id}"
            title = f"{name} ({symbol}) — CoinGecko"

            sources.append(
                Source(
                    source_id=f"coingecko::{coin_id}",
                    source_type=SOURCE_NAME_COINGECKO,
                    title=title[:500],
                    url=url,
                    publisher="CoinGecko",
                    snippet=snippet[:500],
                    published_at=None,
                    trust_score=TRUST_SCORE_COINGECKO,
                    relevance_score=0.9,
                    stance_hint=None,
                )
            )
        return self.postprocess(sources, max_results)

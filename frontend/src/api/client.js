const BASE_URL = 'http://localhost:8000';

/**
 * POST /api/v1/verify-claim
 * Runs the full pipeline: retrieval → NLI → LLM → confidence → graph.
 * Returns the GraphResponse JSON (same shape as sampleGraph.json).
 */
export async function verifyClaim(claimText) {
  const res = await fetch(`${BASE_URL}/api/v1/verify-claim`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ claim_text: claimText }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Server error (HTTP ${res.status})`);
  }
  return res.json();
}

/**
 * GET /api/v1/sources?claim=...&max_results=10
 * Runs retrieval only (no NLI / LLM). Returns SourcesResponse.
 */
export async function fetchSources(claimText, maxResults = 10) {
  const params = new URLSearchParams({ claim: claimText, max_results: maxResults });
  const res = await fetch(`${BASE_URL}/api/v1/sources?${params}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Server error (HTTP ${res.status})`);
  }
  return res.json();
}

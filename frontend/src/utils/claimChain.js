// Feature 15: claim-chain navigation helpers.
//
// The chain keeps a short, token-safe trail of recently verified claims so
// the LLM gets "prior context" on follow-up claims, and so the user can
// click back into an earlier claim in the session. See plan.md Feature 15.

// Total claims kept in the visible chain / sent as context.
export const CONTEXT_CHAIN_MAX_LEN = 6;

// How many of the most recent claims keep their full evidence snippets when
// sent to the LLM as context. Older claims are decayed down to just
// claim text + verdict + confidence — a one-line summary costs ~15-20
// tokens vs ~150-300 with snippets, so a 6-claim chain still stays well
// under the model's context limit even at full length.
const CONTEXT_FULL_DETAIL_COUNT = 2;

/**
 * Build the next history array: removes any existing entry for the same
 * claim (so revisiting a claim moves it to the front, not duplicates it),
 * appends the new entry, and caps the total length.
 */
export function pushClaimHistory(prevHistory, entry, maxLen = CONTEXT_CHAIN_MAX_LEN) {
  const deduped = prevHistory.filter((h) => h.claim_text !== entry.claim_text);
  return [...deduped, entry].slice(-maxLen);
}

/**
 * Returns the history to actually send as `context_claims` to the backend —
 * the most recent CONTEXT_FULL_DETAIL_COUNT entries keep their snippets,
 * older entries are stripped down to claim text + verdict + confidence.
 */
export function decayContextForLLM(history) {
  const n = history.length;
  return history.map((h, i) => {
    const isRecent = i >= n - CONTEXT_FULL_DETAIL_COUNT;
    return isRecent ? h : { ...h, top_snippets: [] };
  });
}

/**
 * Extracts up to 3 short evidence snippets from a graph's top-confidence
 * evidence nodes, for use as a history entry's `top_snippets`.
 */
export function extractTopSnippets(graph) {
  return (graph.nodes || [])
    .filter((n) => !n.is_main_claim && n.top_sources?.length > 0)
    .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
    .slice(0, 3)
    .flatMap((n) => n.top_sources?.slice(0, 1).map((s) => s.snippet || '') || [])
    .filter(Boolean)
    .slice(0, 3);
}

/** Builds a claim-history entry from a verified claim's graph. */
export function buildHistoryEntry(claimText, graph) {
  return {
    claim_text: claimText,
    verdict: graph.metadata.overall_verdict,
    confidence: graph.metadata.overall_confidence,
    top_snippets: extractTopSnippets(graph),
  };
}

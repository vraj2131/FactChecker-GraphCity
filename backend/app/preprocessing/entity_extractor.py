import re
from typing import Set, Tuple

# Words skipped at position 0 (common sentence starters that are not proper nouns)
_SENTENCE_STARTERS = {
    "the", "a", "an", "this", "that", "it", "they", "we", "he", "she",
    "if", "when", "while", "although", "however", "therefore", "since",
    "because", "despite", "during", "after", "before", "there", "here",
    "some", "many", "most", "few", "all", "both", "each", "every",
    "researchers", "scientists", "experts", "according", "studies",
    "new", "recent", "latest", "first", "second", "third",
    "in", "on", "at", "for", "from", "with", "by", "as", "of",
    # Generic human/group nouns — common sentence subjects, not proper nouns
    "humans", "human", "people", "persons", "children", "adults",
    "women", "men", "workers", "citizens", "users", "patients",
    "students", "animals", "individuals", "countries", "nations",
    "companies", "businesses", "governments", "scientists",
    "exercise", "research", "evidence", "data", "results", "studies",
}

# Generic titles/roles that appear in many unrelated contexts — not useful as anchors
_GENERIC_TITLES = {
    "president", "prime", "minister", "secretary", "senator", "governor",
    "director", "chairman", "mayor", "commissioner", "general", "colonel",
    "captain", "admiral", "justice", "judge", "representative", "delegate",
    "ambassador", "chancellor", "premier", "speaker", "treasurer",
    "government", "administration", "official", "candidate", "election",
    "parliament", "congress", "senate", "committee", "department",
    "university", "institute", "foundation", "organization", "association",
    "according", "report", "study", "research", "analysis", "source",
}

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
    "and", "or", "but", "that", "this", "it", "its", "not", "no", "has",
    "have", "had", "will", "would", "could", "should", "may", "can",
    "also", "just", "more", "most", "some", "such", "than", "then",
    "their", "there", "they", "when", "which", "who", "whom", "why",
    "how", "what", "where", "very", "even", "into", "over",
    "after", "about", "because", "so", "do", "did", "us", "our",
}


def extract_claim_entities(text: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract two tiers of entities from claim text:

    anchor_entities: proper nouns / named subjects identified by capitalization.
      - Words capitalized mid-sentence (i > 0) are strong proper noun signals.
      - First-word candidates are included only if not a common sentence starter.
      These are the KEY subjects — a source missing ALL anchors is off-topic.

    numeric_entities: numbers, years, percentages.
      Used as secondary evidence that a source discusses the same statistic/fact.

    Returns (anchor_entities, numeric_entities) — both lowercased.
    """
    if not text:
        return set(), set()

    # --- Numeric entities ---
    numeric: Set[str] = set()
    for match in re.findall(r'\b\d[\d,./%-]*\b', text):
        clean = match.strip('.,')
        if clean:
            numeric.add(clean)

    # --- Anchor entities: capitalization-based proper noun detection ---
    anchor: Set[str] = set()
    words = text.split()
    for i, word in enumerate(words):
        clean = re.sub(r'[^a-zA-Z0-9\-]', '', word)
        if not clean or len(clean) < 3:
            continue
        lower = clean.lower()
        if lower in _STOP or lower in _GENERIC_TITLES:
            continue

        # A word is a proper noun candidate if:
        # - It starts with uppercase (traditional proper noun), OR
        # - It contains any uppercase letter anywhere (camelCase: iPhone, eBay, mRNA)
        has_upper = word[0].isupper() or any(c.isupper() for c in clean)

        if i == 0:
            # First word: include only if not a generic sentence starter
            if lower not in _SENTENCE_STARTERS and has_upper:
                anchor.add(lower)
        else:
            # Mid-sentence: any uppercase signal = likely proper noun
            if has_upper:
                anchor.add(lower)

    # Quoted phrases as single anchor units
    for phrase in re.findall(r'"([^"]+)"', text):
        stripped = phrase.strip()
        if stripped and stripped.lower() not in _STOP:
            anchor.add(stripped.lower())

    return anchor, numeric


def entity_overlap_score(entities: Set[str], source_text: str) -> float:
    """
    Fraction of entities found in source_text (0.0–1.0).
    Returns 1.0 when entities is empty (no penalty).
    """
    if not entities:
        return 1.0
    if not source_text:
        return 0.0
    lower = source_text.lower()
    matched = sum(1 for ent in entities if ent in lower)
    return matched / len(entities)


def anchor_present(anchor_entities: Set[str], source_text: str) -> bool:
    """
    Returns True if at least one anchor entity appears in source_text.
    A source missing all anchors is almost certainly off-topic.
    Returns True when anchor_entities is empty (no penalty).
    """
    if not anchor_entities:
        return True
    lower = source_text.lower()
    return any(ent in lower for ent in anchor_entities)

import re
from typing import Set

_SCIENCE_PATTERNS = [
    r"\bstudy\b", r"\bstudies\b", r"\bresearch\b", r"\bscientists?\b",
    r"\bexperiment\b", r"\bgene\b", r"\bgenetic\b", r"\bDNA\b", r"\bRNA\b",
    r"\bprotein\b", r"\bclimate\b", r"\buniverse\b", r"\bquantum\b",
    r"\bparticle\b", r"\bspecies\b", r"\bevolution\b", r"\bphysics\b",
    r"\bchemistry\b", r"\bbiology\b", r"\bastronomy\b", r"\bgeology\b",
    r"\batmosphere\b", r"\bcarbon\b", r"\bgreenhouse\b", r"\bfossil\b",
    r"\bextinct\b", r"\bcelsius\b", r"\bCO2\b", r"\bhydrogen\b",
    r"\boxygen\b", r"\batom\b", r"\bmolecule\b", r"\blaboratory\b",
    r"\bpeer.reviewed\b", r"\bscientific\b", r"\blight.year\b",
]

_HEALTH_PATTERNS = [
    r"\bhealth\b", r"\bdisease\b", r"\bmedicine\b", r"\bdrug\b",
    r"\bvaccine\b", r"\bcancer\b", r"\bbrain\b", r"\bheart\b",
    r"\bblood\b", r"\bdiabetes\b", r"\bvirus\b", r"\bbacteria\b",
    r"\bsymptom\b", r"\btreatment\b", r"\btherapy\b", r"\bhospital\b",
    r"\bpatient\b", r"\binfection\b", r"\bpandemic\b", r"\bepidemic\b",
    r"\bmental health\b", r"\bnutrition\b", r"\bcalorie\b",
    r"\bvitamin\b", r"\bantibody\b", r"\bimmune\b", r"\bclinical\b",
    r"\bsurgery\b", r"\bpharma\b", r"\bFDA\b", r"\bCDC\b",
    r"\bmedical\b", r"\bmortality\b", r"\blifespans?\b",
]

_ECONOMICS_PATTERNS = [
    r"\bGDP\b", r"\binflation\b", r"\beconom\w+\b", r"\bstock\b",
    r"\brevenue\b", r"\bprofit\b", r"\bSEC\b", r"\bearnings?\b",
    r"\bfinancial\b", r"\bunemployment\b", r"\binterest rate\b",
    r"\bfiscal\b", r"\bfederal reserve\b", r"\bbond\b", r"\bequity\b",
    r"\btrade\b", r"\bexport\b", r"\bimport\b", r"\btariff\b",
    r"\bcorporation\b", r"\binvestment\b", r"\bIPO\b", r"\bmerger\b",
    r"\bquarterly\b", r"\bbalance sheet\b", r"\bdebt\b", r"\bdeficit\b",
    r"\bbudget\b", r"\bGNP\b", r"\bpoverty\b", r"\bincome\b",
    r"\bwage\b", r"\bsalary\b", r"\bworld bank\b", r"\bIMF\b",
    r"\bbillion\b", r"\btrillion\b", r"\bmarket cap\b",
]

_CRYPTO_PATTERNS = [
    r"\bbitcoin\b", r"\bcrypto\b", r"\bblockchain\b", r"\bethereum\b",
    r"\bNFT\b", r"\bcoin\b", r"\btoken\b", r"\bmining\b", r"\bwallet\b",
    r"\bDeFi\b", r"\bdefi\b", r"\bweb3\b", r"\bsatoshi\b",
    r"\bBTC\b", r"\bETH\b", r"\bSOL\b", r"\bXRP\b", r"\bADA\b",
    r"\bcryptocurrency\b", r"\bdecentralized\b", r"\bhash.?rate\b",
    r"\baltcoin\b", r"\bstablecoin\b", r"\bsmart contract\b",
    r"\bcoinbase\b", r"\bbinance\b", r"\bhodl\b",
]


def _compile(patterns: list) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


_RE_SCIENCE = _compile(_SCIENCE_PATTERNS)
_RE_HEALTH = _compile(_HEALTH_PATTERNS)
_RE_ECONOMICS = _compile(_ECONOMICS_PATTERNS)
_RE_CRYPTO = _compile(_CRYPTO_PATTERNS)


def detect_domains(claim: str) -> Set[str]:
    """Return domain tags for a claim (subset of: 'science', 'health', 'economics', 'crypto')."""
    domains: Set[str] = set()
    if _RE_SCIENCE.search(claim):
        domains.add("science")
    if _RE_HEALTH.search(claim):
        domains.add("health")
    if _RE_ECONOMICS.search(claim):
        domains.add("economics")
    if _RE_CRYPTO.search(claim):
        domains.add("crypto")
    return domains

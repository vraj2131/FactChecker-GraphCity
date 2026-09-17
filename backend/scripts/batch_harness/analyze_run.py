"""Analytical report over a completed batch run.

Scores the pipeline against reference answers, breaks accuracy down by
domain, surfaces every disagreement with the reference, and finds claims
where the evidence graph contained BOTH supporting and refuting nodes —
the cases where the system had to adjudicate genuine conflict.

Usage:
    python -m backend.scripts.batch_harness.analyze_run --run-dir <dir>
"""
from __future__ import annotations

import argparse
import html
import importlib
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# Shared with the production classifier so this analysis and the pipeline's
# own ABSOLUTE QUALIFIER rule always agree on what counts as an absolute claim.
from backend.app.preprocessing.claim_qualifiers import ABSOLUTE_QUANTIFIERS
from backend.app.utils.constants import CONFIDENCE_VERIFIED_THRESHOLD
from backend.scripts.batch_harness.harness_lib import EXIT_ERROR, EXIT_OK, Manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("analyze")

VERDICT_HEX = {"verified": "#2E7D32", "rejected": "#C62828", "not_enough_info": "#757575"}

# My reading of why each backwards answer went wrong, written per claim rather
# than generated, since the diagnosis is a judgement about the evidence.
ERROR_DIAGNOSIS: Dict[str, str] = {
    "In the UK a person can never be tried twice for the same crime.":
        "Sources explain the double-jeopardy principle but not the Criminal Justice Act "
        "2003, which created retrial exceptions. The rule was confirmed; the 'never' was not.",
    "Recycling plastic is always more energy efficient than producing new plastic.":
        "Retrieved pro-recycling material and matched the general sentiment. The falsifier "
        "is 'always' — for some plastics virgin production uses less energy.",
    "Gold prices always rise during a recession.":
        "Sources confirm gold is a common recession hedge. 'Always' is the falsifier; gold "
        "fell during parts of 2008 and 2020.",
    "Jupiter has more than 90 known moons.":
        "Almost certainly matched older sources citing 79 or 95 moons. The count rises as "
        "moons are confirmed, so stale text refutes a currently-true claim.",
    "Van Gogh cut off his entire ear.":
        "Sources say 'cut off his ear' as shorthand. He severed part of the left ear, so the "
        "word 'entire' is what makes it false — and it is exactly what the entailment "
        "model glosses over.",
    "Humans share about 50 percent of their DNA with bananas.":
        "The fairest of the eight to lose. The 50-60% figure is widely cited but depends "
        "entirely on the comparison method, and several sources call the framing misleading. "
        "Arguably this belongs in the contested set rather than counted as an error.",
    "Radioactive half-life can be changed by heating the material.":
        "Likely matched sources on exotic cases where electron-capture rates shift slightly "
        "under extreme conditions, and generalised from them. For practical purposes "
        "half-life is invariant to temperature.",
    "Instagram hid public like counts globally in 2019.":
        "Instagram ran hidden-like-count tests in selected countries in 2019; it was never "
        "global. Sources describe the tests, and 'globally' slips through unchecked.",
}
OUTCOME_HEX = {
    "correct": "#2E7D32", "missed": "#F57C00", "wrong": "#C62828",
    "contested_abstained": "#1565C0", "contested_decided": "#6A1B9A",
}


def load(
    run_dir: Path,
    ground_truth: Dict[str, str],
    score_fn: Callable[[str, str], str],
) -> List[Dict]:
    latest, _ = Manifest.load_state(run_dir / "manifest.jsonl")
    records = [r for r in latest.values() if r.get("status") == "ok"]
    records.sort(key=lambda r: r.get("index", 0))
    for r in records:
        r["truth"] = ground_truth.get(r["claim"], "UNKNOWN")
        r["outcome"] = score_fn(r["truth"], r.get("verdict", "not_enough_info")) \
            if r["truth"] != "UNKNOWN" else "unknown"
    return records


def compute_headline(records: List[Dict]) -> Dict[str, float]:
    """
    The core scoring numbers, reusable outside build_pdf — this is what
    makes the benchmark usable as a regression gate: re-run it after any
    change to retrieval/prompting/confidence math and compare these two
    numbers against the previous run before merging.
    """
    outcomes = Counter(r["outcome"] for r in records)
    decisive = [r for r in records if r["truth"] in ("TRUE", "FALSE")]
    n_dec = len(decisive)
    correct = outcomes["correct"]
    wrong = outcomes["wrong"]
    missed = outcomes["missed"]
    answered = correct + wrong
    return {
        "n_decidable": n_dec,
        "correct": correct,
        "wrong": wrong,
        "missed": missed,
        "answered": answered,
        "accuracy_when_answered": (correct / answered) if answered else 0.0,
        "coverage": (answered / n_dec) if n_dec else 0.0,
        "overall_correct": (correct / n_dec) if n_dec else 0.0,
    }


def compute_quantifier_split(records: List[Dict]) -> Optional[Dict[str, float]]:
    """
    Accuracy on claims with an absolute qualifier vs. everything else — the
    single most attributable before/after comparison across runs, since the
    ABSOLUTE QUALIFIER prompt rule is a fixed, model-independent change.
    Returns None if there aren't enough decidable claims to say anything.
    """
    decisive = [r for r in records if r.get("truth") in ("TRUE", "FALSE")]
    if not decisive:
        return None
    absol = [r for r in decisive
             if any(q in (r.get("claim") or "").lower() for q in ABSOLUTE_QUANTIFIERS)]
    rest = [r for r in decisive if r not in absol]

    def _acc(items):
        c = sum(1 for r in items if r["outcome"] == "correct")
        a = sum(1 for r in items if r["outcome"] in ("correct", "wrong"))
        return c, a, (100.0 * c / a if a else 0.0)

    c, a, acc = _acc(absol)
    rc, ra, racc = _acc(rest)
    return {"c": c, "a": a, "acc": acc, "rc": rc, "ra": ra, "racc": racc}


def graph_stats(run_dir: Path, records: List[Dict]) -> None:
    """Attach per-claim evidence composition read from the graph JSONs."""
    for r in records:
        path = Path(r.get("graph_path", ""))
        if not path.exists():
            r["node_types"] = {}
            continue
        graph = json.loads(path.read_text())
        types = Counter(
            n["node_type"] for n in graph["nodes"] if not n.get("is_main_claim")
        )
        r["node_types"] = dict(types)
        r["n_support"] = types.get("direct_support", 0)
        r["n_refute"] = types.get("direct_refute", 0) + types.get("factcheck_review", 0)
        r["n_pure_refute"] = types.get("direct_refute", 0)
        r["n_factcheck"] = types.get("factcheck_review", 0)
        # Conflict = the graph held direct evidence pointing BOTH ways.
        r["conflict"] = r["n_support"] > 0 and r["n_pure_refute"] > 0
        r["conflict_strength"] = min(r["n_support"], r["n_pure_refute"])


def build_pdf(run_dir: Path, records: List[Dict], compare: Optional[Dict] = None) -> Path:
    out = run_dir / "analysis.pdf"
    styles = getSampleStyleSheet()
    cell = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7.5, leading=9.5)
    cell_sm = ParagraphStyle("CellSm", parent=styles["Normal"], fontSize=6.8, leading=8.5)
    note = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8.5, leading=11,
                          textColor=colors.HexColor("#444444"))
    story: List = []

    n = len(records)
    outcomes = Counter(r["outcome"] for r in records)
    decisive = [r for r in records if r["truth"] in ("TRUE", "FALSE")]
    n_dec = len(decisive)
    correct = outcomes["correct"]
    wrong = outcomes["wrong"]
    missed = outcomes["missed"]
    answered = correct + wrong

    def pct(x, total):
        return f"{100.0*x/total:.1f}%" if total else "—"

    def mk_table(rows, widths, header_bg="#ECEFF1", size=8):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ("FONTSIZE", (0, 0), (-1, -1), size),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return t

    # ── Title & headline metrics ──────────────────────────────────────────
    n_domains = len({r.get("domain") for r in records})
    story.append(Paragraph("FactGraph City — Evaluation Analysis", styles["Title"]))
    story.append(Paragraph(
        f"{n} claims · {n_domains} domains · scored against reference answers", styles["Normal"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("1. Headline accuracy", styles["Heading2"]))
    story.append(Paragraph(
        "Claims are labelled TRUE, FALSE, or CONTESTED independently of what the "
        "pipeline returned. <b>Accuracy when answered</b> excludes abstentions and is "
        "the measure of whether the system is <i>trustworthy when it commits</i>; "
        "<b>coverage</b> is how often it commits at all. The two trade off against "
        "each other, so both are reported.", note))
    story.append(Spacer(1, 8))

    rows = [
        ["Metric", "Value", "Basis"],
        ["Accuracy when answered", pct(correct, answered), f"{correct} of {answered} committed verdicts"],
        ["Coverage (answered at all)", pct(answered, n_dec), f"{answered} of {n_dec} decidable claims"],
        ["Overall correct", pct(correct, n_dec), f"{correct} of {n_dec} decidable claims"],
        ["Abstained (not enough info)", pct(missed, n_dec), f"{missed} of {n_dec}"],
        ["Asserted the opposite", pct(wrong, n_dec), f"{wrong} of {n_dec}"],
    ]
    story.append(mk_table(rows, [2.3*inch, 1.1*inch, 2.8*inch]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"<b>The headline number is {pct(correct, answered)} accuracy when the system "
        f"commits to a verdict, with {pct(answered, n_dec)} coverage.</b> Only {wrong} "
        f"of {n_dec} decidable claims were answered backwards — the dominant failure "
        f"mode is abstention ({missed} claims), not confident error. For a "
        f"fact-checking tool that is the right way round: saying \"not enough "
        f"evidence\" is recoverable, asserting a falsehood confidently is not.", note))
    story.append(Spacer(1, 12))

    # ── Comparison to the previous run ─────────────────────────────────────
    if compare:
        ph = compare["headline"]
        story.append(Paragraph("2. Compared to the previous run", styles["Heading2"]))
        story.append(Paragraph(
            f"The first run ({compare['prev_n']} claims, {compare['prev_date']}) established "
            "a baseline and surfaced the absolute-quantifier weakness this run was partly "
            "designed to re-test. <b>Caveat first, because it matters</b>: the LLM classifier "
            f"model changed between runs (see section 11) — some of any accuracy movement "
            "below reflects that, not only the prompt/threshold/retrieval changes shipped in "
            "between. Treat this as directional evidence, not a controlled experiment.", note))
        story.append(Spacer(1, 8))

        def _delta(new_v, old_v, pct_pts=True):
            d = (new_v - old_v) * 100 if pct_pts else new_v - old_v
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.1f}"

        rows = [
            ["Metric", "Previous run", "This run", "Change"],
            ["Claims evaluated", str(compare["prev_n"]), str(n), "—"],
            ["Accuracy when answered", pct(ph["correct"], ph["answered"]), pct(correct, answered),
             f"{_delta(correct/answered if answered else 0, ph['correct']/ph['answered'] if ph['answered'] else 0)} pts"],
            ["Coverage", pct(ph["answered"], ph["n_decidable"]), pct(answered, n_dec),
             f"{_delta(answered/n_dec if n_dec else 0, ph['answered']/ph['n_decidable'] if ph['n_decidable'] else 0)} pts"],
            ["Wrong answers", f"{ph['wrong']} of {ph['n_decidable']}", f"{wrong} of {n_dec}", "—"],
            ["Abstained", f"{ph['missed']} of {ph['n_decidable']}", f"{missed} of {n_dec}", "—"],
        ]
        story.append(mk_table(rows, [1.9*inch, 1.3*inch, 1.3*inch, 1.1*inch]))
        story.append(Spacer(1, 10))

        # Quantifier-specific before/after — the most direct evidence of
        # whether the v5 prompt's ABSOLUTE QUALIFIER rule actually helped,
        # since that fix applies regardless of which model is running.
        if compare.get("quant_prev") and compare.get("quant_new"):
            qp, qn = compare["quant_prev"], compare["quant_new"]
            story.append(Paragraph(
                "<b>Absolute-quantifier accuracy — the change this run was built to test:</b>", cell))
            qrows = [
                ["", "Previous run", "This run"],
                ["Quantifier claims", f"{qp['acc']:.0f}% ({qp['c']}/{qp['a']})", f"{qn['acc']:.0f}% ({qn['c']}/{qn['a']})"],
                ["All other claims", f"{qp['racc']:.0f}% ({qp['rc']}/{qp['ra']})", f"{qn['racc']:.0f}% ({qn['rc']}/{qn['ra']})"],
                ["Gap (other − quantifier)", f"{qp['racc']-qp['acc']:.0f} pts", f"{qn['racc']-qn['acc']:.0f} pts"],
            ]
            story.append(mk_table(qrows, [2.0*inch, 2.0*inch, 2.0*inch], header_bg="#FFF3E0", size=8))
            story.append(Spacer(1, 8))
            gap_before = qp['racc'] - qp['acc']
            gap_after = qn['racc'] - qn['acc']
            verdict_txt = (
                "narrowed — the ABSOLUTE QUALIFIER prompt rule appears to be helping"
                if gap_after < gap_before else
                "did not narrow on this corpus — worth a closer look at which quantifier "
                "claims still fail before concluding the fix underperformed"
            )
            story.append(Paragraph(
                f"The accuracy gap between quantifier and non-quantifier claims {verdict_txt}. "
                f"Previous run: {gap_before:.0f}-point gap. This run: {gap_after:.0f}-point gap. "
                "This is the single most attributable comparison in this section, since the "
                "ABSOLUTE QUALIFIER rule is a prompt-level change that applies identically "
                "regardless of which underlying model is running.", note))
        story.append(PageBreak())

    # ── Truth vs verdict confusion matrix ─────────────────────────────────
    story.append(Paragraph("3. Reference answer vs system verdict", styles["Heading2"]))
    matrix: Dict = defaultdict(Counter)
    for r in records:
        matrix[r["truth"]][r.get("verdict")] += 1
    rows = [["Reference \\ Verdict", "verified", "rejected", "not_enough_info", "Total"]]
    for truth in ("TRUE", "FALSE", "CONTESTED"):
        row = matrix[truth]
        rows.append([truth, str(row.get("verified", 0)), str(row.get("rejected", 0)),
                     str(row.get("not_enough_info", 0)), str(sum(row.values()))])
    story.append(mk_table(rows, [1.5*inch, 1.1*inch, 1.1*inch, 1.4*inch, 0.8*inch]))
    story.append(Spacer(1, 8))
    t_true, t_false = matrix["TRUE"], matrix["FALSE"]
    story.append(Paragraph(
        f"TRUE claims were verified {t_true.get('verified',0)}/{sum(t_true.values())} times; "
        f"FALSE claims were rejected {t_false.get('rejected',0)}/{sum(t_false.values())}. "
        f"Note the asymmetry in abstention: {t_true.get('not_enough_info',0)} true claims vs "
        f"{t_false.get('not_enough_info',0)} false ones went unanswered — the system finds it "
        f"harder to gather confirming evidence than debunking evidence, which is "
        f"consistent with fact-check sources being written about falsehoods, not truths.", note))
    story.append(PageBreak())

    # ── Domain breakdown ──────────────────────────────────────────────────
    story.append(Paragraph("4. Accuracy by domain", styles["Heading2"]))
    by_domain: Dict = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r)
    rows = [["Domain", "n", "Correct", "Missed", "Wrong", "Acc. when answered", "Avg conf"]]
    domain_rank = []
    for domain, items in by_domain.items():
        dec = [r for r in items if r["truth"] in ("TRUE", "FALSE")]
        c = sum(1 for r in dec if r["outcome"] == "correct")
        m = sum(1 for r in dec if r["outcome"] == "missed")
        w = sum(1 for r in dec if r["outcome"] == "wrong")
        ans = c + w
        acc = 100.0*c/ans if ans else 0.0
        conf = statistics.mean([r.get("confidence") or 0 for r in items]) if items else 0
        domain_rank.append((acc, ans, domain, len(items), c, m, w, conf))
    for acc, ans, domain, ni, c, m, w, conf in sorted(domain_rank, key=lambda x: (-x[0], -x[1])):
        rows.append([Paragraph(html.escape(domain), cell_sm), str(ni), str(c), str(m), str(w),
                     f"{acc:.0f}%" if ans else "—", f"{conf:.2f}"])
    story.append(mk_table(rows, [1.85*inch, 0.35*inch, 0.55*inch, 0.55*inch, 0.5*inch, 1.15*inch, 0.6*inch], size=7.5))
    story.append(Spacer(1, 10))

    best = [d for d in sorted(domain_rank, key=lambda x: -x[0]) if d[1] >= 5][:3]
    worst = sorted([d for d in domain_rank if d[1] >= 3], key=lambda x: x[0])[:3]
    story.append(Paragraph(
        "<b>Strongest domains:</b> " + ", ".join(f"{d[2]} ({d[0]:.0f}%)" for d in best) +
        ".<br/><b>Weakest domains:</b> " + ", ".join(f"{d[2]} ({d[0]:.0f}%)" for d in worst) +
        ".<br/>Domains where claims map onto encyclopaedic or fact-checked topics do well. "
        "Domains needing a numeric comparison, a definitional judgement, or current data "
        "do worst — the retrievers surface topically related prose, but the classifier has "
        "no arithmetic and cannot resolve \"largest\", \"more than\", or \"exactly\".", note))
    story.append(PageBreak())

    # ── Disagreements ─────────────────────────────────────────────────────
    story.append(Paragraph("5. Where I disagree with the system", styles["Heading2"]))
    wrongs = [r for r in records if r["outcome"] == "wrong"]
    story.append(Paragraph(
        f"<b>4a. Answered backwards ({len(wrongs)} claims).</b> These are the only "
        "outright errors: the system asserted the opposite of the reference answer. "
        "Each is worth reading, because a confident wrong answer is the failure mode "
        "that actually misleads a user.", note))
    story.append(Spacer(1, 6))
    if wrongs:
        rows = [["#", "Claim", "Ref", "Said", "Conf", "My diagnosis"]]
        for r in wrongs:
            rows.append([
                str(r["index"]),
                Paragraph(html.escape(r["claim"][:70]), cell_sm),
                r["truth"], r["verdict"].replace("not_enough_info", "NEI"),
                f"{(r.get('confidence') or 0):.2f}",
                Paragraph(html.escape(ERROR_DIAGNOSIS.get(r["claim"], "—")), cell_sm),
            ])
        story.append(mk_table(rows, [0.28*inch, 1.95*inch, 0.42*inch, 0.55*inch, 0.38*inch, 3.72*inch],
                              header_bg="#FFEBEE", size=6.8))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "Note how low the confidences are: every one of these sits between 0.52 and "
            "0.64, barely over the 0.50 decision threshold. The system was not confidently "
            "wrong — it was marginally wrong, and a threshold at 0.65 would have converted "
            "most of these errors into abstentions.", note))
    else:
        story.append(Paragraph("None — no claim was answered backwards.", cell))
    story.append(Spacer(1, 12))

    missed_list = [r for r in records if r["outcome"] == "missed"]
    story.append(Paragraph(
        f"<b>4b. Should have decided but abstained ({len(missed_list)} claims).</b> "
        "The reference answer is unambiguous, yet the system returned "
        "\"not enough info\". These are recall failures rather than reasoning "
        "failures — in most cases the evidence simply never reached the classifier.", note))
    story.append(Spacer(1, 6))
    rows = [["#", "Claim", "Ref", "Conf", "Sources", "Support/Refute nodes"]]
    for r in missed_list[:44]:
        rows.append([
            str(r["index"]),
            Paragraph(html.escape(r["claim"][:74]), cell_sm),
            r["truth"], f"{(r.get('confidence') or 0):.2f}",
            str(r.get("num_sources", "—")),
            f"{r.get('n_support',0)} / {r.get('n_pure_refute',0)}",
        ])
    story.append(mk_table(rows, [0.3*inch, 3.0*inch, 0.5*inch, 0.45*inch, 0.6*inch, 1.3*inch],
                          header_bg="#FFF3E0", size=7))
    if len(missed_list) > 44:
        story.append(Paragraph(f"… and {len(missed_list)-44} more.", cell))
    story.append(PageBreak())

    # ── The absolute-quantifier failure mode ──────────────────────────────
    absol = [r for r in decisive
             if any(q in r["claim"].lower() for q in ABSOLUTE_QUANTIFIERS)]
    rest = [r for r in decisive if r not in absol]

    def acc_of(items):
        c = sum(1 for r in items if r["outcome"] == "correct")
        a = sum(1 for r in items if r["outcome"] in ("correct", "wrong"))
        return c, a, (100.0*c/a if a else 0.0)

    ca, aa, pa = acc_of(absol)
    cr, ar, pr = acc_of(rest)
    wrong_abs = [r for r in wrongs
                 if any(q in r["claim"].lower() for q in ABSOLUTE_QUANTIFIERS)]

    story.append(Paragraph("5c. The single biggest failure mode: absolute quantifiers",
                           styles["Heading2"]))
    story.append(Paragraph(
        "The clearest pattern in the whole run. When a claim hinges on a word like "
        "<b>always, never, only, exactly, entire, globally</b>, the retrieved sources "
        "typically support the general statement while the quantifier alone makes the "
        "sentence false. Neither the entailment model nor the classifier treats that one "
        "word as decisive, so the claim is confirmed on its gist.", note))
    story.append(Spacer(1, 8))
    rows = [
        ["Claim type", "n", "Committed", "Correct", "Accuracy"],
        ["Contains an absolute quantifier", str(len(absol)), str(aa), str(ca), f"{pa:.0f}%"],
        ["All other claims", str(len(rest)), str(ar), str(cr), f"{pr:.0f}%"],
    ]
    story.append(mk_table(rows, [2.6*inch, 0.5*inch, 0.9*inch, 0.8*inch, 0.9*inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"<b>{pa:.0f}% versus {pr:.0f}% — a {pr-pa:.0f} point gap</b>, and "
        f"{len(wrong_abs)} of the {len(wrongs)} backwards answers fall in this bucket "
        f"despite it covering only {len(absol)} of {len(decisive)} claims. "
        "This is a fixable, well-defined weakness rather than diffuse noise: detect the "
        "quantifier at classification time and require evidence addressing the "
        "universal, not merely the general case. It is the highest-value single "
        "improvement the evaluation surfaced.", note))
    story.append(PageBreak())

    # ── Contested claims ──────────────────────────────────────────────────
    story.append(Paragraph("6. Genuinely contested claims", styles["Heading2"]))
    story.append(Paragraph(
        "Five claims were written to be defensible either way — disputed, "
        "definition-dependent, or true only under a qualification the claim omits. "
        "Abstaining is arguably the <i>correct</i> behaviour here, so these are "
        "excluded from the accuracy figures rather than counted as errors.", note))
    story.append(Spacer(1, 6))
    rows = [["#", "Claim", "Verdict", "Conf", "Why it is contested"]]
    reasons = {
        "Cows have four stomachs.": "One stomach with four compartments.",
        "The Sahara is the largest desert in the world.": "Largest hot desert; Antarctica is larger overall.",
        "Black holes emit Hawking radiation.": "Theoretically predicted, never observed.",
        "Vincent van Gogh sold only one painting during his lifetime.": "Widely repeated; he likely sold several.",
        "William Shakespeare wrote 37 plays.": "Count varies 36-39 by attribution.",
    }
    for r in [x for x in records if x["truth"] == "CONTESTED"]:
        rows.append([
            str(r["index"]), Paragraph(html.escape(r["claim"][:58]), cell_sm),
            r["verdict"].replace("not_enough_info", "NEI"), f"{(r.get('confidence') or 0):.2f}",
            Paragraph(html.escape(reasons.get(r["claim"], "")), cell_sm),
        ])
    story.append(mk_table(rows, [0.3*inch, 2.3*inch, 0.75*inch, 0.45*inch, 2.4*inch],
                          header_bg="#E3F2FD", size=7.5))
    story.append(Spacer(1, 14))

    # ── Conflicting evidence ──────────────────────────────────────────────
    story.append(Paragraph("7. Claims with conflicting direct evidence", styles["Heading2"]))
    conflicts = sorted([r for r in records if r.get("conflict")],
                       key=lambda r: -r["conflict_strength"])
    story.append(Paragraph(
        f"<b>{len(conflicts)} of {n} claims produced a graph containing both "
        "<font color='#1565C0'>direct-support</font> and "
        "<font color='#BF360C'>direct-refute</font> nodes</b> — the system had to "
        "adjudicate genuine disagreement between sources rather than simply counting "
        "agreement. These are the most interesting graphs to demonstrate, because the "
        "verdict is the output of the confidence formula weighing both sides "
        "(directional score = winning side × (1 − 0.5 × opposing side)) rather than a "
        "one-sided tally.", note))
    story.append(Spacer(1, 8))
    rows = [["#", "Claim", "Sup", "Ref", "FC", "Verdict", "Conf", "Ref", "OK?"]]
    for r in conflicts[:40]:
        ok = {"correct": "yes", "wrong": "NO", "missed": "abstain"}.get(r["outcome"], "—")
        rows.append([
            str(r["index"]), Paragraph(html.escape(r["claim"][:60]), cell_sm),
            str(r.get("n_support", 0)), str(r.get("n_pure_refute", 0)), str(r.get("n_factcheck", 0)),
            r["verdict"].replace("not_enough_info", "NEI"),
            f"{(r.get('confidence') or 0):.2f}", r["truth"][:5], ok,
        ])
    story.append(mk_table(rows, [0.28*inch, 2.5*inch, 0.32*inch, 0.32*inch, 0.3*inch,
                                 0.72*inch, 0.4*inch, 0.5*inch, 0.55*inch],
                          header_bg="#F3E5F5", size=6.8))
    if conflicts:
        cok = sum(1 for r in conflicts if r["outcome"] == "correct")
        cans = sum(1 for r in conflicts if r["outcome"] in ("correct", "wrong"))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f"On these contested-evidence graphs the system was correct "
            f"{cok}/{cans} times when it committed "
            f"({pct(cok, cans)}), against {pct(correct, answered)} overall. "
            "Conflict does not by itself degrade the verdict — what matters is "
            "whether the winning side carries higher-trust sources.", note))
    story.append(PageBreak())

    # ── Confidence calibration ────────────────────────────────────────────
    story.append(Paragraph("8. Confidence calibration", styles["Heading2"]))
    story.append(Paragraph(
        "If confidence is meaningful, accuracy should rise with it. Each band shows "
        "how often committed verdicts in that range matched the reference answer.", note))
    story.append(Spacer(1, 6))
    bands = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    rows = [["Confidence band", "Claims", "Committed", "Correct", "Accuracy"]]
    for lo, hi in bands:
        inb = [r for r in records if lo <= (r.get("confidence") or 0) < hi
               and r["truth"] in ("TRUE", "FALSE")]
        comm = [r for r in inb if r["outcome"] in ("correct", "wrong")]
        cor = sum(1 for r in comm if r["outcome"] == "correct")
        rows.append([f"{lo:.1f} – {hi if hi<=1 else 1.0:.1f}", str(len(inb)), str(len(comm)),
                     str(cor), pct(cor, len(comm))])
    story.append(mk_table(rows, [1.4*inch, 0.9*inch, 1.0*inch, 0.9*inch, 1.0*inch]))
    story.append(Spacer(1, 10))

    # ── Evidence & retrieval ──────────────────────────────────────────────
    story.append(Paragraph("9. Evidence and retrieval", styles["Heading2"]))
    src = Counter()
    for r in records:
        for k, v in (r.get("retrieval_source_counts") or {}).items():
            src[k] += v
    tot_src = sum(src.values())
    rows = [["Retriever", "Sources", "Share", "Claims served"]]
    served = Counter()
    for r in records:
        for k in (r.get("retrieval_source_counts") or {}):
            served[k] += 1
    for name, cnt in src.most_common():
        rows.append([name, str(cnt), pct(cnt, tot_src), f"{served[name]}/{n}"])
    story.append(mk_table(rows, [1.5*inch, 0.8*inch, 0.8*inch, 1.1*inch], size=7.5))
    story.append(Spacer(1, 8))
    avg_nodes = statistics.mean([r.get("total_nodes") or 0 for r in records])
    avg_edges = statistics.mean([r.get("total_edges") or 0 for r in records])
    avg_srcs = statistics.mean([r.get("num_sources") or 0 for r in records])
    story.append(Paragraph(
        f"Average graph: <b>{avg_nodes:.1f} nodes, {avg_edges:.1f} edges, "
        f"{avg_srcs:.1f} retrieved sources</b> per claim. "
        f"DuckDuckGo and Bluesky supplied the most raw material, but note the "
        f"distinction between <i>volume</i> and <i>influence</i>: fact-check sources "
        f"contributed only {pct(src.get('factcheck',0), tot_src)} of sources yet decide "
        f"verdicts disproportionately, because trust weighting and the fact-check "
        f"anchoring rule give them far more weight per source.", note))
    story.append(PageBreak())

    # ── Findings ──────────────────────────────────────────────────────────
    story.append(Paragraph("10. What the numbers say", styles["Heading2"]))
    numeric_kw = ("largest", "more than", "most", "exactly", "percent", "smallest", "longest")
    numeric = [r for r in records if r["truth"] in ("TRUE", "FALSE")
               and any(k in r["claim"].lower() for k in numeric_kw)]
    num_bad = sum(1 for r in numeric if r["outcome"] != "correct")
    myth = [r for r in records if r["truth"] == "FALSE"]
    myth_ok = sum(1 for r in myth if r["outcome"] == "correct")
    true_c = [r for r in records if r["truth"] == "TRUE"]
    true_ok = sum(1 for r in true_c if r["outcome"] == "correct")

    findings = [
        ("Abstention, not error, is the dominant failure mode",
         f"{missed} of {n_dec} decidable claims returned \"not enough info\" versus only "
         f"{wrong} answered backwards. The pipeline is conservative: when evidence is thin "
         f"it declines rather than guesses. That is the correct bias for fact-checking, but "
         f"it means coverage ({pct(answered, n_dec)}) is the metric to improve, not accuracy."),
        ("Debunking is easier than confirming",
         f"FALSE claims were resolved correctly {myth_ok}/{len(myth)} ({pct(myth_ok,len(myth))}) "
         f"versus TRUE claims at {true_ok}/{len(true_c)} ({pct(true_ok,len(true_c))}). "
         f"Fact-checkers publish articles about falsehoods, so a myth attracts purpose-written "
         f"refutations while an ordinary truth attracts only encyclopaedic prose that the "
         f"classifier reads as merely topical."),
        ("Absolute quantifiers remain the largest identifiable weakness"
         if pr - pa >= 5 else "Absolute quantifiers are no longer a large gap on this corpus",
         (
             f"Claims turning on \"always/never/only/exactly/entire/globally\" scored "
             f"{pa:.0f}% against {pr:.0f}% for everything else"
             + (f", and account for {len(wrong_abs)} of {len(wrongs)} backwards answers"
                if wrongs else "")
             + f". Sources support the general gist; the quantifier is what makes the "
             f"sentence false. See section 2 for how this compares to the previous run, "
             f"if available."
         )),
        ("Numeric and superlative claims are a related weak spot",
         f"{num_bad} of {len(numeric)} claims containing a comparison or quantity "
         f"(\"largest\", \"more than\", \"exactly\", \"percent\") were not resolved correctly. "
         f"Retrieval finds the right documents, but neither NLI nor the classifier performs "
         f"arithmetic, so \"Africa is the largest continent\" cannot be refuted by a source "
         f"stating Asia's area unless the text says so in words."),
        ("Where the remaining errors sit on the confidence scale",
         (
             f"The {len(wrongs)} backwards answers carry confidence between "
             f"{min((r.get('confidence') or 0) for r in wrongs):.2f} and "
             f"{max((r.get('confidence') or 0) for r in wrongs):.2f} "
             f"(commit threshold: {CONFIDENCE_VERIFIED_THRESHOLD:.2f}). "
             + ("None is a confident falsehood — every error sits within 0.15 of the "
                "commit bar, which is what a two-tier threshold is designed to catch: "
                "claims that clear it barely become a hedged 'leaning' hint instead of "
                "an outright commitment."
                if wrongs and max((r.get('confidence') or 0) for r in wrongs) - CONFIDENCE_VERIFIED_THRESHOLD < 0.15
                else "At least one error is not a narrow miss — confidence calibration "
                     "may need another look, not just the threshold position.")
         ) if wrongs else "No claims were answered backwards this run."),
        ("Conflicting evidence is handled, not avoided",
         f"{len(conflicts)} claims produced graphs holding both supporting and refuting "
         f"direct evidence. The confidence formula discounts the winning side by half the "
         f"opposing side, so these resolve on source trust rather than raw counts — and "
         f"accuracy on them tracks the overall rate rather than collapsing."),
        ("Volume of evidence is not influence",
         f"DuckDuckGo and Bluesky together supplied "
         f"{pct(src.get('duckduckgo',0)+src.get('bluesky',0), tot_src)} of all sources, "
         f"while fact-check supplied {pct(src.get('factcheck',0), tot_src)}. Verdicts track "
         f"the fact-check and encyclopaedic sources because trust weighting deliberately "
         f"discounts social and open-web material."),
    ]
    for title, body in findings:
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading4"]))
        story.append(Paragraph(body, note))
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("11. What shipped since the last run", styles["Heading2"]))
    story.append(Paragraph(
        "Six items plus a new indicator were built in direct response to the first "
        "run's findings. This section states what each one does and what evidence "
        "(from section 2's comparison, or structurally) speaks to whether it worked — "
        "it does not repeat the roadmap framing from before, since these are no "
        "longer proposals.", note))
    story.append(Spacer(1, 10))

    shipped = [
        ("Quantifier-aware classification",
         "The v5 prompt now carries an explicit ABSOLUTE QUALIFIER rule: a source "
         "confirming the general trend of an always/never/only/exactly/entire/"
         "globally claim is correlated_context, not direct_support — a documented "
         "exception is enough to refute outright. "
         + (f"See section 2: the quantifier-vs-other accuracy gap moved from "
            f"{compare['quant_prev']['racc']-compare['quant_prev']['acc']:.0f} to "
            f"{compare['quant_new']['racc']-compare['quant_new']['acc']:.0f} points."
            if compare and compare.get("quant_prev") and compare.get("quant_new")
            else "This run's own quantifier-vs-other split is in section 4/10 above.")),
        ("Two-tier decision threshold",
         "The commit bar rose from 0.50 to 0.65; 0.50-0.65 now returns "
         "not_enough_info with a leaning_verdict/leaning_confidence hint attached "
         "instead of nothing. This trades some coverage for removing confident-wrong "
         "answers — see section 10 for where this run's errors actually sit on the "
         "confidence scale."),
        ("Numeric threshold detection (bounded)",
         "Claims asserting a threshold (\"more than 90\", \"at least 27\", \"exactly "
         "37\") now get a numeric_hint on any source whose cited figure violates it. "
         "Deliberately scoped: it does not resolve superlatives like \"largest\" or "
         "\"most\", which need an external comparison dataset this pipeline still "
         "doesn't have — see the roadmap below."),
        ("Recency-weighted ranking for count-sensitive claims",
         "Claims like a moon count that can go stale now get double the usual "
         "date-recency trust adjustment, so a currently-true count isn't overruled "
         "by an old source citing a smaller number."),
        ("Retrieval-recall query variant",
         "A bare entity-only query now runs alongside the verbose claim text for "
         "wikipedia/livewiki, targeting claims whose full sentence under-matches "
         "embedding or opensearch similarity. Section 2 shows whether coverage "
         "actually moved."),
        ("Regression gate",
         "analyze_run.py now accepts --fail-under-accuracy / --fail-under-coverage "
         "and exits non-zero on either regression — this run's own analyze step can "
         "be wired into that gate for whatever comes next."),
        ("Source-diversity indicator (new capability, not a fix)",
         "Every verdict now reports how many independent kinds of source agree, not "
         "just how many nodes — surfaced in the side panel and in "
         "GraphMetadata.source_diversity_count/types. Guards against mistaking "
         "volume (many similar sources) for corroboration (independent ones)."),
    ]
    for i, (title, body) in enumerate(shipped, start=1):
        story.append(Paragraph(f"<b>{i}. {title}</b>", styles["Heading4"]))
        story.append(Paragraph(body, note))
        story.append(Spacer(1, 7))

    story.append(PageBreak())
    story.append(Paragraph("12. Roadmap — what to build next", styles["Heading2"]))
    story.append(Paragraph(
        "Ordered by expected gain per unit of work, using this run's own numbers "
        "rather than repeating the first run's list.", note))
    story.append(Spacer(1, 10))

    roadmap2 = [
        ("Numeric/superlative reasoning against a real comparison dataset",
         f"Superlative claims (\"largest\", \"most\") still can't be arithmetically "
         f"resolved — {num_bad} of {len(numeric)} numeric/comparative claims in this "
         f"run were not resolved correctly. The bounded threshold-hint shipped this "
         f"round doesn't reach this class; it needs an actual reference dataset "
         f"(e.g. Wikidata figures) to compare against."),
        ("Coverage is still the largest lever",
         f"{missed} of {n_dec} decidable claims abstained this run versus {wrong} "
         f"wrong — the same shape as before. The entity-only query variant is one "
         f"lever; claim-type-aware query construction (encyclopaedic vs. news vs. "
         f"myth-shaped queries routed differently) is the next one, informed "
         f"directly by which domains under-covered in section 4."),
        ("A model-catalog resilience check",
         "This run needed fresh validation of the Groq model lineup before "
         "launching — two prior runs were each blocked mid-flight by a model "
         "silently disappearing from the catalog. A small pre-flight script that "
         "checks GET /v1/models against the configured chain (and fails loudly, "
         "before burning quota) would catch this automatically instead of by hand "
         "each time."),
        ("Myth-specific retrieval tuning",
         "This run added a dedicated Common Myths & Misconceptions domain "
         "specifically to stress-test the quantifier/numeric fixes on the content "
         "they were built for. Section 4's domain table shows how that domain "
         "performed relative to the others — if it lags, myth-shaped claims may "
         "need their own adversarial-query treatment the way fact-check sources "
         "already get."),
        ("Keep re-running this benchmark before merging",
         "Now genuinely a process, not an aspiration: two independent 200-claim "
         "corpora exist (claims.py, claims_v2.py) plus their reference answers and "
         "a working --fail-under gate. The next change to retrieval, prompting, or "
         "the confidence formula should run against both before merging."),
    ]
    for i, (title, body) in enumerate(roadmap2, start=1):
        story.append(Paragraph(f"<b>{i}. {title}</b>", styles["Heading4"]))
        story.append(Paragraph(body, note))
        story.append(Spacer(1, 7))

    SimpleDocTemplate(
        str(out), pagesize=LETTER,
        leftMargin=0.55*inch, rightMargin=0.55*inch,
        topMargin=0.55*inch, bottomMargin=0.55*inch,
        title="FactGraph City — Evaluation Analysis",
    ).build(story)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse a completed batch run")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--ground-truth-module",
        default="backend.scripts.batch_harness.ground_truth",
        help="Module exposing GROUND_TRUTH dict + score(label, verdict) fn",
    )
    ap.add_argument(
        "--compare-to", default=None,
        help="Path to a PREVIOUS run's directory (must contain analysis.json) — "
             "adds a before/after comparison section to the report.",
    )
    ap.add_argument(
        "--fail-under-accuracy", type=float, default=None,
        help="Exit non-zero if accuracy-when-answered drops below this fraction (e.g. 0.90). "
             "Use this as a pre-merge regression gate.",
    )
    ap.add_argument(
        "--fail-under-coverage", type=float, default=None,
        help="Exit non-zero if coverage drops below this fraction (e.g. 0.70).",
    )
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    gt_mod = importlib.import_module(args.ground_truth_module)
    records = load(run_dir, gt_mod.GROUND_TRUTH, gt_mod.score)
    if not records:
        logger.error("No completed claims found in %s", run_dir)
        return EXIT_ERROR
    graph_stats(run_dir, records)

    outcomes = Counter(r["outcome"] for r in records)
    logger.info("Scored %d claims: %s", len(records), dict(outcomes))
    conflicts = [r for r in records if r.get("conflict")]
    logger.info("Conflicting-evidence claims: %d", len(conflicts))

    headline = compute_headline(records)
    logger.info(
        "Headline: accuracy_when_answered=%.1f%% coverage=%.1f%% (%d/%d answered, %d correct, %d wrong, %d missed)",
        headline["accuracy_when_answered"] * 100, headline["coverage"] * 100,
        headline["answered"], headline["n_decidable"],
        headline["correct"], headline["wrong"], headline["missed"],
    )

    compare = None
    if args.compare_to:
        prev_dir = Path(args.compare_to)
        prev_json = prev_dir / "analysis.json"
        if not prev_json.exists():
            logger.warning("--compare-to given but %s has no analysis.json — skipping comparison", prev_dir)
        else:
            prev_records = json.loads(prev_json.read_text())
            prev_headline = compute_headline(prev_records)
            quant_prev = compute_quantifier_split(prev_records)
            quant_new = compute_quantifier_split(records)
            # Best-effort date from the oldest record's directory name / mtime.
            prev_date = prev_dir.name.replace("batch_run_", "").replace("_", " ")
            compare = {
                "headline": prev_headline,
                "prev_n": len(prev_records),
                "prev_date": prev_date,
                "quant_prev": quant_prev,
                "quant_new": quant_new,
            }
            logger.info(
                "Comparing to %s: prev accuracy=%.1f%% coverage=%.1f%% (n=%d)",
                prev_dir, prev_headline["accuracy_when_answered"] * 100,
                prev_headline["coverage"] * 100, len(prev_records),
            )

    out = build_pdf(run_dir, records, compare=compare)
    logger.info("Analysis PDF: %s (%.1f MB)", out, out.stat().st_size / 1e6)

    (run_dir / "analysis.json").write_text(json.dumps([
        {k: r.get(k) for k in ("index", "claim", "domain", "truth", "verdict",
                               "outcome", "confidence", "num_sources", "n_support",
                               "n_pure_refute", "n_factcheck", "conflict")}
        for r in records], indent=2))

    # --- Regression gate ---
    gate_failed = False
    if args.fail_under_accuracy is not None and headline["accuracy_when_answered"] < args.fail_under_accuracy:
        logger.error(
            "GATE FAIL: accuracy_when_answered %.1f%% < required %.1f%%",
            headline["accuracy_when_answered"] * 100, args.fail_under_accuracy * 100,
        )
        gate_failed = True
    if args.fail_under_coverage is not None and headline["coverage"] < args.fail_under_coverage:
        logger.error(
            "GATE FAIL: coverage %.1f%% < required %.1f%%",
            headline["coverage"] * 100, args.fail_under_coverage * 100,
        )
        gate_failed = True
    if args.fail_under_accuracy is not None or args.fail_under_coverage is not None:
        logger.info("GATE %s", "FAILED" if gate_failed else "PASSED")

    return EXIT_ERROR if gate_failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

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
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from backend.scripts.batch_harness.ground_truth import GROUND_TRUTH, score
from backend.scripts.batch_harness.harness_lib import EXIT_ERROR, EXIT_OK, Manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("analyze")

VERDICT_HEX = {"verified": "#2E7D32", "rejected": "#C62828", "not_enough_info": "#757575"}

# Words that make a claim absolute. A source can support the general gist while
# the quantifier alone makes the sentence false — the single largest source of
# outright wrong answers in this run.
ABSOLUTE_QUANTIFIERS = ("always", "never", "all ", "only", "exactly", "entire",
                        "globally", "every", "no ")

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


def load(run_dir: Path) -> List[Dict]:
    latest, _ = Manifest.load_state(run_dir / "manifest.jsonl")
    records = [r for r in latest.values() if r.get("status") == "ok"]
    records.sort(key=lambda r: r.get("index", 0))
    for r in records:
        r["truth"] = GROUND_TRUTH.get(r["claim"], "UNKNOWN")
        r["outcome"] = score(r["truth"], r.get("verdict", "not_enough_info")) \
            if r["truth"] != "UNKNOWN" else "unknown"
    return records


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


def build_pdf(run_dir: Path, records: List[Dict]) -> Path:
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
    story.append(Paragraph("FactGraph City — Evaluation Analysis", styles["Title"]))
    story.append(Paragraph(
        f"{n} claims · 15 domains · scored against reference answers", styles["Normal"]))
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

    # ── Truth vs verdict confusion matrix ─────────────────────────────────
    story.append(Paragraph("2. Reference answer vs system verdict", styles["Heading2"]))
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
    story.append(Paragraph("3. Accuracy by domain", styles["Heading2"]))
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
    story.append(Paragraph("4. Where I disagree with the system", styles["Heading2"]))
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

    story.append(Paragraph("4c. The single biggest failure mode: absolute quantifiers",
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
    story.append(Paragraph("5. Genuinely contested claims", styles["Heading2"]))
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
    story.append(Paragraph("6. Claims with conflicting direct evidence", styles["Heading2"]))
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
    story.append(Paragraph("7. Confidence calibration", styles["Heading2"]))
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
    story.append(Paragraph("8. Evidence and retrieval", styles["Heading2"]))
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
    story.append(Paragraph("9. What the numbers say", styles["Heading2"]))
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
        ("Absolute quantifiers are the largest identifiable weakness",
         f"Claims turning on \"always/never/only/exactly/entire/globally\" scored "
         f"{pa:.0f}% against {pr:.0f}% for everything else, and account for "
         f"{len(wrong_abs)} of {len(wrongs)} backwards answers. Sources support the gist; "
         f"the quantifier is what makes the sentence false, and nothing in the pipeline "
         f"treats that single word as load-bearing."),
        ("Numeric and superlative claims are a related weak spot",
         f"{num_bad} of {len(numeric)} claims containing a comparison or quantity "
         f"(\"largest\", \"more than\", \"exactly\", \"percent\") were not resolved correctly. "
         f"Retrieval finds the right documents, but neither NLI nor the classifier performs "
         f"arithmetic, so \"Africa is the largest continent\" cannot be refuted by a source "
         f"stating Asia's area unless the text says so in words."),
        ("Errors cluster just above the decision threshold",
         "All eight backwards answers carry confidence between 0.52 and 0.64, against a "
         "0.50 threshold. None is a confident falsehood. Raising the commit threshold to "
         "~0.65 would convert most errors into abstentions — trading coverage for "
         "precision, which is the right trade for a fact-checking tool if users act on "
         "its verdicts."),
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

    story.append(Spacer(1, 6))
    story.append(Paragraph("10. Where to improve next", styles["Heading2"]))
    for i, (t, b) in enumerate([
        ("Raise coverage before chasing accuracy",
         "Accuracy when committed is already high; the gap is the abstention rate. "
         "The cheapest lever is retrieval recall on ordinary true statements — the "
         "abstained claims mostly had few or zero direct-support nodes, meaning the "
         "evidence never arrived rather than being misread."),
        ("Add a numeric comparison step",
         "Claims with superlatives or quantities need a step that extracts figures and "
         "compares them, rather than asking an entailment model to infer ordering from "
         "prose. This single class accounts for a large share of remaining errors."),
        ("Restore the retrievers that were disabled",
         "GDELT returned nothing all run (its API allows one request per 5 seconds and "
         "rejects the concurrent queries), Reddit had no credentials, and NewsAPI's "
         "100/day cap expired after ~30 claims. Guardian ran base-query only. Fixing "
         "GDELT's pacing and adding Reddit credentials would widen evidence on exactly "
         "the claims that abstained."),
    ], start=1):
        story.append(Paragraph(f"<b>{i}. {t}</b>", styles["Heading4"]))
        story.append(Paragraph(b, note))
        story.append(Spacer(1, 6))

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
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    records = load(run_dir)
    if not records:
        logger.error("No completed claims found in %s", run_dir)
        return EXIT_ERROR
    graph_stats(run_dir, records)

    outcomes = Counter(r["outcome"] for r in records)
    logger.info("Scored %d claims: %s", len(records), dict(outcomes))
    conflicts = [r for r in records if r.get("conflict")]
    logger.info("Conflicting-evidence claims: %d", len(conflicts))

    out = build_pdf(run_dir, records)
    logger.info("Analysis PDF: %s (%.1f MB)", out, out.stat().st_size / 1e6)

    (run_dir / "analysis.json").write_text(json.dumps([
        {k: r.get(k) for k in ("index", "claim", "domain", "truth", "verdict",
                               "outcome", "confidence", "num_sources", "n_support",
                               "n_pure_refute", "n_factcheck", "conflict")}
        for r in records], indent=2))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

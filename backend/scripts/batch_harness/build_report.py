"""Phase 3 — build the deliverables from the run's manifest and screenshots.

Produces two things:
  report.pdf  — 2 pages per claim (graph shot, then panel shot + detail),
                front matter with run stats, and a failure appendix
  index.html  — a filterable contact sheet for fast spot-checking, with
                images referenced by relative path (not base64) so it stays
                small and opens instantly

Screenshots are downscaled to JPEG first: 400 PNGs at ~200KB would make a
~90MB PDF, which is unwieldy to share.

Usage:
    python -m backend.scripts.batch_harness.build_report --run-dir <dir>
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.scripts.batch_harness.harness_lib import EXIT_ERROR, EXIT_OK, Manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("report")

VERDICT_COLORS = {
    "verified": colors.HexColor("#2E7D32"),
    "rejected": colors.HexColor("#C62828"),
    "not_enough_info": colors.HexColor("#9E9E9E"),
}
VERDICT_HEX = {"verified": "#2E7D32", "rejected": "#C62828", "not_enough_info": "#757575"}

DERIVED_W, DERIVED_H = 1200, 675
JPEG_QUALITY = 72
IMG_W = 6.6 * inch
IMG_H = IMG_W * (DERIVED_H / DERIVED_W)


def _truncate(text: Optional[str], length: int = 70) -> str:
    text = text or ""
    return text if len(text) <= length else text[: length - 1] + "…"


def derive_images(run_dir: Path, records: List[Dict], shots: Dict[str, Dict]) -> Dict[str, Dict]:
    """Downscale PNG screenshots to JPEG. Returns {claim_id: {a: path, b: path}}."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        logger.error("Pillow is required to build the report (pip install Pillow)")
        raise

    derived_dir = run_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict] = {}

    for record in records:
        claim_id = record["claim_id"]
        shot = shots.get(claim_id)
        if not shot or shot.get("status") != "ok":
            continue
        pair = {}
        for key, src_key in (("a", "shot_graph"), ("b", "shot_panel")):
            src = shot.get(src_key)
            if not src or not Path(src).exists():
                continue
            dst = derived_dir / f"{claim_id}_{key}.jpg"
            if not dst.exists():
                with PILImage.open(src) as img:
                    img.convert("RGB").resize((DERIVED_W, DERIVED_H), PILImage.LANCZOS).save(
                        dst, "JPEG", quality=JPEG_QUALITY, optimize=True
                    )
            pair[key] = dst
        if pair:
            out[claim_id] = pair

    total_mb = sum(p.stat().st_size for d in out.values() for p in d.values()) / 1e6
    logger.info("Derived %d image pairs (%.1f MB)", len(out), total_mb)
    return out


def build_pdf(run_dir: Path, records: List[Dict], shots: Dict[str, Dict],
              derived: Dict[str, Dict]) -> Path:
    out_pdf = run_dir / "report.pdf"
    styles = getSampleStyleSheet()
    cell = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7.5, leading=9.5)
    claim_style = ParagraphStyle("ClaimBig", parent=styles["Heading2"], fontSize=13, leading=16)
    note_style = ParagraphStyle("Note", parent=styles["Normal"], fontSize=8.5,
                                leading=11, textColor=colors.HexColor("#555555"))

    story: List = []
    ok_records = [r for r in records if r.get("status") == "ok"]
    verdicts = Counter(r.get("verdict") for r in ok_records)
    retriever_totals: Counter = Counter()
    for r in ok_records:
        for name, count in (r.get("retrieval_source_counts") or {}).items():
            retriever_totals[name] += count

    # ── Front matter ──────────────────────────────────────────────────────
    story.append(Paragraph("FactGraph City — 200-Claim Evaluation Report", styles["Title"]))
    story.append(Spacer(1, 6))
    started = min((r.get("started_at") or "" for r in records), default="?")
    finished = max((r.get("finished_at") or "" for r in records), default="?")
    story.append(Paragraph(
        f"Run window: {started} → {finished} &nbsp;|&nbsp; git {records[0].get('git_sha','?') if records else '?'}"
        f" &nbsp;|&nbsp; {len(ok_records)} of {len(records)} claims completed",
        styles["Normal"]))
    story.append(Spacer(1, 12))

    summary_rows = [["Verdict", "Count", "%"]]
    for verdict in ("verified", "rejected", "not_enough_info"):
        n = verdicts.get(verdict, 0)
        pct = (100.0 * n / len(ok_records)) if ok_records else 0.0
        summary_rows.append([verdict.replace("_", " ").title(), str(n), f"{pct:.1f}%"])
    table = Table(summary_rows, colWidths=[2.4 * inch, 1.0 * inch, 1.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECEFF1")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(Paragraph("Verdict distribution", styles["Heading3"]))
    story.append(table)
    story.append(Spacer(1, 12))

    ret_rows = [["Retriever", "Sources contributed"]]
    for name, count in retriever_totals.most_common():
        ret_rows.append([name, str(count)])
    ret_table = Table(ret_rows, colWidths=[2.4 * inch, 2.0 * inch])
    ret_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECEFF1")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
    ]))
    story.append(Paragraph("Retriever contribution (surviving sources, all claims)", styles["Heading3"]))
    story.append(ret_table)
    story.append(Spacer(1, 12))

    # Which LLM classified which claims. Records written before model rotation
    # existed have no llm_model field and were all llama-3.1-8b-instant.
    model_counts = Counter(
        r.get("llm_model") or "llama-3.1-8b-instant" for r in ok_records
    )
    if len(model_counts) > 1:
        m_rows = [["Classifier model", "Claims"]]
        for name, count in model_counts.most_common():
            m_rows.append([name, str(count)])
        m_table = Table(m_rows, colWidths=[2.8 * inch, 1.4 * inch])
        m_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FFF8E1")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
        ]))
        story.append(Paragraph("LLM classifier used per claim", styles["Heading3"]))
        story.append(m_table)
        story.append(Paragraph(
            "Groq enforces a per-model daily token budget (500,000 for "
            "llama-3.1-8b-instant, ~5,200 tokens per claim). The run rotated "
            "across models as each budget was spent, so the corpus is <b>not</b> "
            "classified by a single model — compare claims within a model group "
            "rather than across groups.", note_style))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Run configuration caveats", styles["Heading3"]))
    story.append(Paragraph(
        "These results were produced under free-tier API constraints and are <b>not</b> "
        "identical to full production behaviour:<br/>"
        "• <b>Context expansion disabled</b> — fewer context-signal nodes than a normal run.<br/>"
        "• <b>Guardian ran base-query only</b> — its two adversarial queries "
        "(\"myth debunked\", \"false\") were dropped to fit a 500 calls/day cap, so refuting "
        "articles from Guardian are under-represented.<br/>"
        "• <b>NewsAPI exhausts after ~30 claims</b> (100 requests/day cap); later claims "
        "receive nothing from it.<br/>"
        "• <b>Reddit unavailable</b> — no API credentials configured; social evidence is "
        "Bluesky-only.<br/>"
        "• <b>GDELT contributed nothing</b> — its API permits one request every 5 seconds "
        "and rejects the pipeline's concurrent queries with HTTP 429. The retriever treats "
        "that as an empty result, so its absence is silent rather than an error.<br/>"
        "• LLM classification window is the top 14 ranked sources; further sources still "
        "appear as second-tier graph nodes.",
        note_style))
    story.append(PageBreak())

    # ── Per-claim pages ───────────────────────────────────────────────────
    for record in ok_records:
        claim_id = record["claim_id"]
        pair = derived.get(claim_id, {})
        verdict = record.get("verdict", "not_enough_info")
        vhex = VERDICT_HEX.get(verdict, "#757575")
        conf = record.get("confidence") or 0.0

        story.append(Paragraph(
            f"{record.get('index')}. {html.escape(record.get('claim',''))}", claim_style))
        story.append(Paragraph(
            f"<font color='{vhex}'><b>{verdict.replace('_',' ').upper()}</b></font>"
            f" &nbsp;·&nbsp; {conf*100:.0f}% confidence"
            f" &nbsp;·&nbsp; {html.escape(record.get('domain',''))}"
            f" &nbsp;·&nbsp; {claim_id}",
            styles["Normal"]))
        story.append(Spacer(1, 8))

        if pair.get("a"):
            story.append(Image(str(pair["a"]), width=IMG_W, height=IMG_H))
        story.append(Spacer(1, 8))

        stat_rows = [[
            "Sources", "Nodes", "Edges", "Support", "Refute", "Fact-check", "Context", "Time",
        ], [
            str(record.get("num_sources", "-")), str(record.get("total_nodes", "-")),
            str(record.get("total_edges", "-")), str(record.get("support_node_count", "-")),
            str(record.get("refute_node_count", "-")), str(record.get("factcheck_node_count", "-")),
            str(record.get("context_node_count", "-")), f"{record.get('elapsed_s','-')}s",
        ]]
        stat_table = Table(stat_rows, colWidths=[0.85 * inch] * 8)
        stat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECEFF1")),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(stat_table)
        story.append(Spacer(1, 6))
        src_counts = record.get("retrieval_source_counts") or {}
        if src_counts:
            story.append(Paragraph(
                "<b>Retrievers:</b> " + ", ".join(f"{k} {v}" for k, v in sorted(src_counts.items())),
                cell))
        story.append(PageBreak())

        # Page 2 — the side panel view
        story.append(Paragraph(
            f"{record.get('index')}. {html.escape(_truncate(record.get('claim',''), 90))} "
            f"— evidence panel", styles["Heading3"]))
        story.append(Spacer(1, 6))
        if pair.get("b"):
            story.append(Image(str(pair["b"]), width=IMG_W, height=IMG_H))
        story.append(Spacer(1, 8))
        notes = record.get("retrieval_notes")
        if notes:
            story.append(Paragraph(f"<b>Retrieval notes:</b> {html.escape(notes)}", cell))
        story.append(PageBreak())

    # ── Failure appendix ──────────────────────────────────────────────────
    failures = [r for r in records if r.get("status") != "ok"]
    story.append(Paragraph("Appendix — claims that did not complete", styles["Heading2"]))
    if not failures:
        story.append(Paragraph("None. All claims completed successfully.", styles["Normal"]))
    else:
        rows = [["#", "Claim", "Status", "Error"]]
        for r in failures:
            rows.append([
                str(r.get("index", "?")),
                Paragraph(html.escape(_truncate(r.get("claim", ""), 60)), cell),
                r.get("status", "?"),
                Paragraph(html.escape(_truncate(r.get("error_class") or "-", 30)), cell),
            ])
        ftable = Table(rows, colWidths=[0.4 * inch, 3.6 * inch, 1.0 * inch, 1.6 * inch])
        ftable.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FFEBEE")),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(ftable)

    SimpleDocTemplate(
        str(out_pdf), pagesize=LETTER,
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        title="FactGraph City — 200-Claim Evaluation Report",
    ).build(story)
    logger.info("PDF: %s (%.1f MB)", out_pdf, out_pdf.stat().st_size / 1e6)
    return out_pdf


def build_html(run_dir: Path, records: List[Dict], derived: Dict[str, Dict]) -> Path:
    out_html = run_dir / "index.html"
    ok_records = [r for r in records if r.get("status") == "ok"]
    verdicts = Counter(r.get("verdict") for r in ok_records)
    by_domain: Dict[str, int] = defaultdict(int)
    for r in ok_records:
        by_domain[r.get("domain", "?")] += 1

    rows = []
    for r in records:
        claim_id = r["claim_id"]
        pair = derived.get(claim_id, {})
        verdict = r.get("verdict") or r.get("status", "?")
        color = VERDICT_HEX.get(verdict, "#757575")
        conf = r.get("confidence")
        conf_txt = f"{conf*100:.0f}%" if isinstance(conf, (int, float)) else "—"
        imgs = ""
        for key in ("a", "b"):
            if pair.get(key):
                rel = Path(pair[key]).relative_to(run_dir)
                imgs += (f'<a href="{rel}" target="_blank">'
                         f'<img loading="lazy" src="{rel}" width="320"></a>')
        src = r.get("retrieval_source_counts") or {}
        src_txt = " · ".join(f"{k}&nbsp;{v}" for k, v in sorted(src.items())) or "—"
        rows.append(f"""
<tr data-verdict="{html.escape(str(verdict))}" data-status="{html.escape(r.get('status','?'))}">
  <td class="num">{r.get('index','?')}</td>
  <td class="claim"><div class="ctext">{html.escape(r.get('claim',''))}</div>
      <div class="meta">{html.escape(r.get('domain',''))} · {claim_id}</div>
      <div class="meta">{src_txt}</div></td>
  <td><span class="badge" style="background:{color}">{html.escape(str(verdict)).replace('_',' ')}</span>
      <div class="conf">{conf_txt}</div></td>
  <td class="shots">{imgs or '<span class="meta">no screenshots</span>'}</td>
</tr>""")

    out_html.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FactGraph City — 200-Claim Run</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ background:#0d1321; color:#e2e8f0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:0; padding:24px; }}
 h1 {{ margin:0 0 4px; font-size:22px; }}
 .sub {{ color:#94a3b8; margin-bottom:16px; }}
 .stats {{ display:flex; gap:18px; flex-wrap:wrap; margin-bottom:18px; }}
 .stat {{ background:#151b2b; border:1px solid #223; border-radius:8px; padding:10px 14px; }}
 .stat b {{ font-size:20px; display:block; }}
 .filters {{ margin-bottom:14px; display:flex; gap:8px; flex-wrap:wrap; }}
 button {{ background:#1b2337; color:#cbd5e1; border:1px solid #2c3852; border-radius:999px;
           padding:6px 14px; cursor:pointer; font:inherit; }}
 button.active {{ background:#3b4d78; color:#fff; }}
 table {{ border-collapse:collapse; width:100%; }}
 td {{ border-top:1px solid #1e2637; padding:12px 8px; vertical-align:top; }}
 .num {{ color:#64748b; width:44px; }}
 .claim {{ max-width:420px; }}
 .ctext {{ font-weight:600; }}
 .meta {{ color:#7c8ba1; font-size:11.5px; margin-top:3px; }}
 .badge {{ display:inline-block; padding:3px 10px; border-radius:999px; color:#fff;
           font-size:11px; font-weight:700; text-transform:uppercase; }}
 .conf {{ color:#94a3b8; font-size:12px; margin-top:4px; }}
 .shots img {{ border-radius:6px; border:1px solid #223; margin-right:8px; }}
</style></head><body>
<h1>FactGraph City — 200-Claim Evaluation</h1>
<div class="sub">{len(ok_records)} of {len(records)} claims completed · {len(by_domain)} domains</div>
<div class="stats">
  <div class="stat"><b>{verdicts.get('verified',0)}</b>Verified</div>
  <div class="stat"><b>{verdicts.get('rejected',0)}</b>Rejected</div>
  <div class="stat"><b>{verdicts.get('not_enough_info',0)}</b>Inconclusive</div>
  <div class="stat"><b>{len(records)-len(ok_records)}</b>Failed</div>
</div>
<div class="filters">
  <button class="active" data-f="all">All</button>
  <button data-f="verified">Verified</button>
  <button data-f="rejected">Rejected</button>
  <button data-f="not_enough_info">Inconclusive</button>
</div>
<table><tbody>{''.join(rows)}</tbody></table>
<script>
document.querySelectorAll('.filters button').forEach(b => b.onclick = () => {{
  document.querySelectorAll('.filters button').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  const f = b.dataset.f;
  document.querySelectorAll('tbody tr').forEach(tr => {{
    tr.style.display = (f === 'all' || tr.dataset.verdict === f) ? '' : 'none';
  }});
}});
</script>
</body></html>""", encoding="utf-8")
    logger.info("HTML: %s (%.0f KB)", out_html, out_html.stat().st_size / 1024)
    return out_html


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: build PDF + HTML report")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("No manifest at %s", manifest_path)
        return EXIT_ERROR

    latest, _ = Manifest.load_state(manifest_path)
    records = sorted(latest.values(), key=lambda r: r.get("index", 0))
    shots, _ = Manifest.load_state(run_dir / "shots.jsonl")

    logger.info("Building report for %d claims (%d with screenshots)",
                len(records), len(shots))
    derived = derive_images(run_dir, records, shots)
    build_pdf(run_dir, records, shots, derived)
    build_html(run_dir, records, derived)
    logger.info("Report complete.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

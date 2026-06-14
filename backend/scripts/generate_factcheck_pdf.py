"""Build a PDF report from the generated fact-check example graphs.

Reads data/artifacts/graph_samples/pdf_examples/summary.json and the
corresponding graph JSON + PNG snapshots, and produces a PDF with, for
each example: the claim, domain, overall verdict/confidence, the graph
snapshot image, and a table of retrieved evidence/sources.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)

EXAMPLES_DIR = Path("data/artifacts/graph_samples/pdf_examples")
OUT_PDF = EXAMPLES_DIR / "factcheck_examples_report.pdf"

VERDICT_COLORS = {
    "verified": colors.HexColor("#2E7D32"),
    "rejected": colors.HexColor("#C62828"),
    "not_enough_info": colors.HexColor("#9E9E9E"),
}


def _truncate(text: str, length: int = 70) -> str:
    text = text or ""
    return text if len(text) <= length else text[: length - 1] + "…"


def build_pdf() -> None:
    summary = json.loads((EXAMPLES_DIR / "summary.json").read_text())

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2_style = styles["Heading2"]
    body_style = styles["BodyText"]
    warn_style = ParagraphStyle(
        "Warning",
        parent=styles["Normal"],
        textColor=colors.HexColor("#C62828"),
        fontSize=12,
        leading=15,
        spaceAfter=12,
    )
    cell_style = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7, leading=9)

    story = []

    # --- Header / known-issue note ---
    story.append(Paragraph("FactGraph City — Fact-Checking Examples Report", title_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "NOTE: Ignore the arrows on the graph snapshots below — this is a "
            "known visualization issue we still have to fix.",
            warn_style,
        )
    )
    story.append(
        Paragraph(
            "Each example shows the claim that was checked, the system's overall "
            "verdict and confidence score, a snapshot of the evidence graph, and "
            "the sources retrieved for that claim.",
            body_style,
        )
    )
    story.append(PageBreak())

    for entry in summary:
        idx = entry["index"]
        domain = entry["domain"]
        claim = entry["claim"]

        story.append(Paragraph(f"Example {idx} — {domain}", h2_style))

        if "error" in entry:
            story.append(Paragraph(f"<b>Claim:</b> {claim}", body_style))
            story.append(Paragraph(f"<font color='red'>ERROR: {entry['error']}</font>", body_style))
            story.append(PageBreak())
            continue

        verdict = entry["verdict"]
        confidence = entry["confidence"]
        verdict_color = VERDICT_COLORS.get(verdict, colors.black)

        story.append(Paragraph(f"<b>Claim:</b> {claim}", body_style))
        story.append(
            Paragraph(
                f"<b>Verdict:</b> <font color='{verdict_color.hexval()}'>"
                f"{verdict.upper()}</font> &nbsp;&nbsp; "
                f"<b>Confidence:</b> {confidence:.2f} &nbsp;&nbsp; "
                f"<b>Sources retrieved:</b> {entry['num_sources']}",
                body_style,
            )
        )
        story.append(Spacer(1, 0.08 * inch))

        graph_path = Path(entry["graph_path"])
        png_path = graph_path.with_name(graph_path.stem + "_3d.png")
        if not png_path.exists():
            png_path = graph_path.with_suffix(".png")
        if png_path.exists():
            story.append(Image(str(png_path), width=5.5 * inch, height=3.44 * inch))

        story.append(Spacer(1, 0.08 * inch))

        # --- Sources / evidence table ---
        graph_data = json.loads(graph_path.read_text())
        rows = [["#", "Type", "Stance/Verdict", "Conf.", "Evidence snippet", "URL"]]
        for node in graph_data["nodes"]:
            if node.get("is_main_claim"):
                continue
            rows.append(
                [
                    node["node_id"].replace("node_", ""),
                    node["node_type"].replace("_", " "),
                    node["verdict"],
                    f"{node['confidence']:.2f}",
                    Paragraph(_truncate(node["text"], 110), cell_style),
                    Paragraph(_truncate(str(node.get("best_source_url") or ""), 45), cell_style),
                ]
            )

        table = Table(
            rows,
            colWidths=[0.4 * inch, 0.9 * inch, 0.85 * inch, 0.45 * inch, 2.6 * inch, 1.6 * inch],
            repeatRows=1,
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474F")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                ]
            )
        )
        story.append(table)
        story.append(PageBreak())

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=LETTER,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
    )
    doc.build(story)
    print(f"[DONE] Wrote {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()

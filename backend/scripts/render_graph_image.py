"""Render a FactGraph JSON snapshot as a static 2D PNG using networkx + matplotlib.

This is a simplified, static stand-in for the 3D force-graph frontend view —
useful for embedding graph snapshots in reports/PDFs.
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def _wrap(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(text or "", width=width)[:4])


def render_graph(graph_json_path: Path, out_png_path: Path) -> None:
    data = json.loads(graph_json_path.read_text())

    G = nx.DiGraph()

    node_colors = {}
    node_sizes = {}
    node_labels = {}

    for node in data["nodes"]:
        node_id = node["node_id"]
        G.add_node(node_id)
        node_colors[node_id] = node.get("color", "#999999")
        size = node.get("size", 10.0)
        node_sizes[node_id] = max(size * 60, 300)
        label = node["text"] if node.get("is_main_claim") else node.get("text", "")
        node_labels[node_id] = _wrap(label, width=20 if node.get("is_main_claim") else 18)

    edge_colors = []
    edge_styles = []
    edge_widths = []
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"])
        edge_colors.append(edge.get("color", "#999999"))
        edge_styles.append("dashed" if edge.get("dashed") else "solid")
        edge_widths.append(max(edge.get("width", 1.0), 0.5))

    main_id = next((n["node_id"] for n in data["nodes"] if n.get("is_main_claim")), None)
    pos = nx.spring_layout(G, seed=42, k=1.6)
    if main_id is not None:
        pos[main_id] = (0.0, 0.0)
        pos = nx.spring_layout(G, seed=42, k=1.6, pos=pos, fixed=[main_id])

    fig, ax = plt.subplots(figsize=(8, 6))

    for (u, v), color, style, width in zip(G.edges(), edge_colors, edge_styles, edge_widths):
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)],
            edge_color=color, style=style, width=width,
            arrows=True, arrowsize=14, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=[node_colors[n] for n in G.nodes()],
        node_size=[node_sizes[n] for n in G.nodes()],
        edgecolors="#333333", linewidths=0.8,
    )

    nx.draw_networkx_labels(
        G, pos, labels=node_labels, ax=ax,
        font_size=6.5, font_color="white", font_weight="bold",
    )

    meta = data["metadata"]
    ax.set_title(
        f"{meta['claim_text']}\nVerdict: {meta['overall_verdict'].upper()}  "
        f"(confidence={meta['overall_confidence']:.2f})",
        fontsize=9, wrap=True,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)


def main() -> None:
    examples_dir = Path("data/artifacts/graph_samples/pdf_examples")
    summary = json.loads((examples_dir / "summary.json").read_text())

    for entry in summary:
        if "error" in entry:
            continue
        graph_path = Path(entry["graph_path"])
        out_png = graph_path.with_suffix(".png")
        print(f"Rendering {graph_path.name} -> {out_png.name}")
        render_graph(graph_path, out_png)


if __name__ == "__main__":
    main()

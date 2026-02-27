import json
from pathlib import Path

from pydantic import ValidationError

from backend.app.schemas import GraphResponse


GRAPH_FILES = [
    Path("data/artifacts/graph_samples/sample_graph_1.json"),
    Path("data/artifacts/graph_samples/sample_graph_2.json"),
    Path("data/artifacts/graph_samples/sample_graph_3.json"),
]


def validate_graph_file(file_path: Path) -> None:
    print(f"\nValidating: {file_path}")

    if not file_path.exists():
        print(f"  ❌ File not found: {file_path}")
        return

    try:
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        graph = GraphResponse.model_validate(payload)

        print("  ✅ Validation passed")
        print(f"  Claim: {graph.metadata.claim_text}")
        print(f"  Verdict: {graph.metadata.overall_verdict}")
        print(f"  Confidence: {graph.metadata.overall_confidence}")
        print(f"  Nodes: {len(graph.nodes)}")
        print(f"  Edges: {len(graph.edges)}")

    except ValidationError as e:
        print("  ❌ Validation failed with schema errors:")
        print(e)

    except json.JSONDecodeError as e:
        print("  ❌ Invalid JSON:")
        print(e)

    except Exception as e:
        print("  ❌ Unexpected error:")
        print(e)


def main() -> None:
    print("=== Sample Graph Validation ===")
    for graph_file in GRAPH_FILES:
        validate_graph_file(graph_file)


if __name__ == "__main__":
    main()
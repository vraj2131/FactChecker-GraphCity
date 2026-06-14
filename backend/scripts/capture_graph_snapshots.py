"""Capture real 3D graph snapshots from the live frontend using Playwright.

For each example graph JSON in data/artifacts/graph_samples/pdf_examples/,
this script:
  1. Opens the frontend (must be running at http://localhost:5173)
  2. Intercepts the /api/v1/verify-claim request and returns the saved graph
     JSON instead of re-running the full pipeline
  3. Submits the claim from the landing page so the graph renders
  4. Waits for the 3D force-layout to settle, then clicks the same
     "Snapshot" button used on the website (renderer.toBlob -> PNG download)
  5. Saves the downloaded PNG next to the graph JSON
"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

EXAMPLES_DIR = Path("data/artifacts/graph_samples/pdf_examples")
FRONTEND_URL = "http://localhost:5173"


def main() -> None:
    summary = json.loads((EXAMPLES_DIR / "summary.json").read_text())

    with sync_playwright() as p:
        browser = p.chromium.launch()

        for entry in summary:
            if "error" in entry:
                continue

            graph_path = Path(entry["graph_path"])
            graph_data = json.loads(graph_path.read_text())
            out_png = graph_path.with_name(graph_path.stem + "_3d.png")

            page = browser.new_page(viewport={"width": 1280, "height": 800})

            def handle_route(route, request, _graph_data=graph_data):
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(_graph_data),
                )

            page.route("**/api/v1/verify-claim", handle_route)

            print(f"[{entry['index']:02d}] {entry['claim'][:60]!r}")
            page.goto(FRONTEND_URL)
            page.fill(".landing-input", entry["claim"])
            page.click(".landing-verify-btn")

            # Wait for the graph view (snapshot button) to appear, then let
            # the 3D force-layout settle and the camera ease into position.
            page.wait_for_selector(".snapshot-btn", timeout=30000)
            page.wait_for_timeout(6000)

            with page.expect_download() as download_info:
                page.click(".snapshot-btn")
            download = download_info.value
            download.save_as(str(out_png))
            print(f"  -> saved {out_png}")

            page.close()

        browser.close()


if __name__ == "__main__":
    main()

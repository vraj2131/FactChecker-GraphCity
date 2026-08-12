"""Phase 2 — capture two screenshots per verified claim.

Replays each saved graph JSON into the running frontend by intercepting the
verify request, so this phase costs zero API quota and can be re-run freely.

Per claim:
  shot A — the 3D graph on its own
  shot B — the same graph after the main claim node is selected, showing the
           side panel (verdict, gauge, reasoning, evidence)

Requires the Vite dev server on :5173. The `window.__factgraph` hook it uses
only exists under `vite dev`, which is also what makes node selection
deterministic instead of a canvas-pixel guess.

Usage:
    python -m backend.scripts.batch_harness.run_phase2_capture --run-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from backend.scripts.batch_harness.harness_lib import (
    EXIT_DEGRADED,
    EXIT_ERROR,
    EXIT_OK,
    Manifest,
    utc_now_iso,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2")

FRONTEND_URL = "http://localhost:5173"
VIEWPORT = {"width": 1600, "height": 900}
SETTLE_MS = 7000          # d3 force layout convergence
PANEL_SETTLE_MS = 1200    # 600ms camera fly + CSS panel slide
BROWSER_RECYCLE_EVERY = 25  # three.js leaks GPU memory across scene teardowns
MAX_ATTEMPTS = 2

# Headless Chromium needs an explicit software GL stack or the WebGL canvas
# renders as a uniform black rectangle.
BROWSER_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--disable-dev-shm-usage",
]


def image_is_blank(path: Path, stddev_threshold: float = 8.0) -> bool:
    """True when a screenshot is ~uniform — the signature of a dead WebGL context.

    Without this check a GL failure yields 400 identical black PNGs and isn't
    noticed until the report is built.
    """
    try:
        from PIL import Image, ImageStat
    except ImportError:
        logger.warning("Pillow not installed — skipping blank-image check")
        return False
    try:
        with Image.open(path) as img:
            small = img.convert("RGB").resize((32, 32))
            stat = ImageStat.Stat(small)
            return max(stat.stddev) < stddev_threshold
    except Exception as exc:
        logger.warning("blank check failed for %s: %s", path.name, exc)
        return False


def capture_one(browser, claim: str, graph: dict, shot_a: Path, shot_b: Path) -> Dict:
    """Capture both shots for a single claim. Returns a result dict."""
    # Fresh page per claim: App.jsx keeps a client-side graph cache keyed on
    # claim text, which would short-circuit the second render in a reused page.
    page = browser.new_page(viewport=VIEWPORT)
    info: Dict = {"panel_opened": False}
    try:
        page.route(
            "**/api/v1/verify-claim",
            lambda route, request: route.fulfill(
                status=200, content_type="application/json", body=json.dumps(graph)
            ),
        )
        page.goto(FRONTEND_URL, timeout=30000)
        page.fill(".landing-input", claim)
        page.click(".landing-verify-btn")
        page.wait_for_selector(".snapshot-btn", timeout=30000)
        page.wait_for_function(
            "() => window.__factgraph && window.__factgraph.ready === true",
            timeout=15000,
        )

        page.wait_for_timeout(SETTLE_MS)
        page.evaluate("window.__factgraph.freezeCamera()")
        page.wait_for_timeout(400)

        shot_a.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(shot_a))

        selected = page.evaluate("window.__factgraph.selectNode('node_main')")
        if not selected:
            raise RuntimeError("selectNode('node_main') returned false")
        page.wait_for_selector(".side-panel-slider--open", timeout=5000)
        page.wait_for_timeout(PANEL_SETTLE_MS)
        page.screenshot(path=str(shot_b))
        info["panel_opened"] = True

        info["blank_a"] = image_is_blank(shot_a)
        info["blank_b"] = image_is_blank(shot_b)
        return info
    finally:
        page.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: graph JSONs -> screenshots")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    shots_dir = run_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.jsonl"
    shots_path = run_dir / "shots.jsonl"

    if not manifest_path.exists():
        logger.error("No manifest at %s — run phase 1 first.", manifest_path)
        return EXIT_ERROR

    # Only claims that produced a graph are capturable.
    latest, _ = Manifest.load_state(manifest_path)
    targets = [
        r for r in latest.values()
        if r.get("status") == "ok" and r.get("graph_path") and Path(r["graph_path"]).exists()
    ]
    targets.sort(key=lambda r: r.get("index", 0))
    if args.limit:
        targets = targets[: args.limit]

    done, _ = Manifest.load_state(shots_path)
    todo = [
        r for r in targets
        if not (
            done.get(r["claim_id"], {}).get("status") == "ok"
            and (shots_dir / f"{r['claim_id']}_b_panel.png").exists()
        )
    ]

    logger.info(
        "Phase 2: %d capturable | %d already shot | %d to capture",
        len(targets), len(targets) - len(todo), len(todo),
    )
    if not todo:
        logger.info("Nothing to capture.")
        return EXIT_OK

    shots = Manifest(shots_path)
    ok_count = fail_count = 0
    preflight_done = False

    with sync_playwright() as p:
        browser = p.chromium.launch(args=BROWSER_ARGS)
        try:
            for n, record in enumerate(todo, start=1):
                claim_id = record["claim_id"]
                claim = record["claim"]
                graph = json.loads(Path(record["graph_path"]).read_text())
                shot_a = shots_dir / f"{claim_id}_a_graph.png"
                shot_b = shots_dir / f"{claim_id}_b_panel.png"

                logger.info("[%d/%d] %s %s", n, len(todo), claim_id, claim[:60])

                result: Optional[Dict] = None
                last_error: Optional[Exception] = None
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        result = capture_one(browser, claim, graph, shot_a, shot_b)
                        break
                    except (PlaywrightError, RuntimeError, Exception) as exc:
                        last_error = exc
                        logger.warning("    attempt %d failed: %s: %s",
                                       attempt, type(exc).__name__, str(exc)[:140])
                        time.sleep(2)

                if result is None:
                    fail_count += 1
                    shots.append({
                        "claim_id": claim_id, "status": "failed",
                        "finished_at": utc_now_iso(),
                        "error_class": type(last_error).__name__ if last_error else "Unknown",
                        "error_msg": str(last_error)[:300] if last_error else None,
                    })
                    continue

                # A dead GL context makes every shot blank — catch it on the
                # first claim rather than after 200 useless captures.
                if not preflight_done:
                    preflight_done = True
                    if result.get("blank_a") and result.get("blank_b"):
                        logger.error(
                            "PREFLIGHT FAILED: first screenshots are blank — WebGL is not "
                            "rendering in this browser. Aborting before wasting the run."
                        )
                        (run_dir / "RUN_DEGRADED").write_text(
                            f"phase2_blank_canvas at={utc_now_iso()}\n"
                        )
                        return EXIT_DEGRADED
                    logger.info("Preflight OK — WebGL is rendering.")

                ok_count += 1
                shots.append({
                    "claim_id": claim_id,
                    "status": "ok",
                    "finished_at": utc_now_iso(),
                    "shot_graph": str(shot_a),
                    "shot_panel": str(shot_b),
                    "panel_opened": result.get("panel_opened", False),
                    "blank_a": result.get("blank_a", False),
                    "blank_b": result.get("blank_b", False),
                })
                if result.get("blank_a") or result.get("blank_b"):
                    logger.warning("    -> captured but a shot looks blank")

                if n % BROWSER_RECYCLE_EVERY == 0 and n < len(todo):
                    logger.info("    recycling browser (memory hygiene)")
                    browser.close()
                    browser = p.chromium.launch(args=BROWSER_ARGS)
        finally:
            with_suppress = getattr(browser, "close", None)
            if with_suppress:
                try:
                    browser.close()
                except Exception:
                    pass
            shots.close()

    logger.info("Phase 2 complete: ok=%d failed=%d", ok_count, fail_count)
    return EXIT_OK if fail_count == 0 else EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

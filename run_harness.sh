#!/usr/bin/env bash
# Auto-restarting driver for the 200-claim batch harness.
#
# Phase 1 (verify) is restarted on crash because it resumes from
# manifest.jsonl and replays completed claims from cache in seconds.
# Exit code 3 means the run is degraded (quotas dead / WebGL broken) and a
# restart would only produce more junk, so we stop and leave the markers.
set -uo pipefail

PROJECT_DIR="/Users/vraj21/Desktop/Projects/Fact Checker"
cd "$PROJECT_DIR" || exit 1

# shellcheck disable=SC1091
source venv/bin/activate

export PYTHONUNBUFFERED=1
# Batch runs should wait out rate limits rather than fail fast the way an
# interactive /verify-claim request should.
export GROQ_RATE_LIMIT_MAX_ATTEMPTS=6
# Each model has its own tokens-per-day budget; rotate rather than idle.
export GROQ_MODEL_CHAIN="llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-120b"
export TOKENIZERS_PARALLELISM=false

# Reuse the newest existing run rather than keying off today's date — a run
# that crosses midnight would otherwise restart into an empty directory and
# redo every claim. Only start a fresh run when none exists.
if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR=$(ls -d data/artifacts/batch_run_* 2>/dev/null | sort | tail -1)
  RUN_DIR="${RUN_DIR:-data/artifacts/batch_run_$(date +%Y%m%d)}"
fi
mkdir -p "$RUN_DIR/logs"

echo "=============================================="
echo " FactGraph batch harness"
echo " run dir : $RUN_DIR"
echo " started : $(date -Iseconds)"
echo "=============================================="

# ---------- Phase 1: verify ----------
MAX_RESTARTS=40
restart=0
while (( restart < MAX_RESTARTS )); do
  ts=$(date +%H%M%S)
  echo ""
  echo "--- phase1 attempt $((restart+1)) at $(date -Iseconds) ---"
  python -m backend.scripts.batch_harness.run_phase1_verify \
      --run-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/logs/phase1_${ts}.log"
  code=${PIPESTATUS[0]}

  if (( code == 0 )); then
    echo "phase1 COMPLETE"
    break
  elif (( code == 3 )); then
    echo "phase1 HALTED (degraded) — not restarting. See $RUN_DIR/RUN_DEGRADED"
    exit 3
  else
    restart=$((restart+1))
    echo "phase1 exited $code — restarting in 120s (restart $restart/$MAX_RESTARTS)"
    sleep 120
  fi
done

# ---------- Phase 2: screenshots ----------
echo ""
echo "--- phase2 at $(date -Iseconds) ---"
for attempt in 1 2 3; do
  python -m backend.scripts.batch_harness.run_phase2_capture \
      --run-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/logs/phase2_${attempt}.log"
  code=${PIPESTATUS[0]}
  if (( code == 0 )); then
    echo "phase2 COMPLETE"
    break
  elif (( code == 3 )); then
    echo "phase2 HALTED (blank canvas) — see $RUN_DIR/RUN_DEGRADED"
    exit 3
  fi
  echo "phase2 attempt $attempt exited $code — retrying in 60s"
  sleep 60
done

# ---------- Phase 3: report ----------
echo ""
echo "--- phase3 report at $(date -Iseconds) ---"
python -m backend.scripts.batch_harness.build_report \
    --run-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/logs/report.log"

touch "$RUN_DIR/HARNESS_DONE"
echo ""
echo "=============================================="
echo " ALL DONE $(date -Iseconds)"
echo " PDF  : $RUN_DIR/report.pdf"
echo " HTML : $RUN_DIR/index.html"
echo "=============================================="

#!/usr/bin/env bash
# Cron watchdog: relaunch the harness tmux session if it disappears entirely
# (reboot, tmux server killed). Everything else — crashes, rate limits — is
# already handled by run_harness.sh's restart loop, so this is a no-op in the
# normal case and must stay cheap.
#
# Install:
#   */10 * * * * "/Users/vraj21/Desktop/Projects/Fact Checker/scripts/watchdog.sh" >> /tmp/factharness_watchdog.log 2>&1
set -uo pipefail

PROJECT_DIR="/Users/vraj21/Desktop/Projects/Fact Checker"
SESSION="factharness"
# Newest existing run, not today's date — a run spanning midnight must not be
# mistaken for "not started yet".
RUN_DIR=$(ls -d "$PROJECT_DIR"/data/artifacts/batch_run_* 2>/dev/null | sort | tail -1)
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/data/artifacts/batch_run_$(date +%Y%m%d)}"

# Finished or deliberately halted — nothing to do.
[[ -f "$RUN_DIR/HARNESS_DONE"  ]] && exit 0
[[ -f "$RUN_DIR/RUN_DEGRADED"  ]] && exit 0

if tmux has-session -t "$SESSION" 2>/dev/null; then
  exit 0
fi

echo "$(date -Iseconds) session '$SESSION' missing — relaunching"

tmux new-session -d -s "$SESSION" -n frontend -c "$PROJECT_DIR/frontend" \
  'npm run dev -- --port 5173 --strictPort 2>&1 | tee /tmp/vite_harness.log'

sleep 8   # let Vite bind before phase 2 could need it

tmux new-window -t "$SESSION" -n harness -c "$PROJECT_DIR" './run_harness.sh'

echo "$(date -Iseconds) relaunched"

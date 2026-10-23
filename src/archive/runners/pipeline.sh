#!/usr/bin/env bash
# Detached end-to-end pipeline (run as a systemd user service so it survives logout/disconnect):
#   1. wait for the running y26m_1536 training, then TTA inference + tuning -> submission.csv
#   2. experiment queue: consensus-label run, then 2048-resolution run (each -> submission_<run>.csv)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
echo "[$(date)] pipeline start"
src/run_final.sh > work/final.log 2>&1; echo "[$(date)] run_final exit $?"
src/queue.sh > work/queue.log 2>&1; echo "[$(date)] queue exit $?"

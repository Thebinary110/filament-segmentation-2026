#!/usr/bin/env bash
# Wait for the 3-channel training/inference service, then run the evaluation (CPU only).
set -euo pipefail
while systemctl --user is-active -q filament-c3; do sleep 60; done
grep -q C3_DONE "$(dirname "$0")/../../work/c3.log" || { echo "c3 run did not finish cleanly"; exit 1; }
cd "$(dirname "$0")"
nice -n 5 python3 c2_eval_c3.py
echo C2_DONE

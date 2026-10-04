#!/usr/bin/env bash
# Template "On-start script". deploy.py inlines this file into the Vast template.
# Fetches the worker scripts at the pinned commit (ORCH_REF) and starts boot.sh.
# A cold worker restarting with no network to GitHub reuses its cached copies.
mkdir -p /workspace/orch
cd /workspace/orch || exit 1
base="${ORCH_RAW_BASE:-https://raw.githubusercontent.com/klutzydrummer/vastai-llmapi-orchestrator}/${ORCH_REF:-main}/worker"
for f in boot.sh fetch_model.py smoke_test.py router.py pyworker_worker.py; do
    curl -fsSL --retry 5 --retry-delay 3 --max-time 60 "$base/$f" -o "$f.new" && mv -f "$f.new" "$f"
done
nohup bash /workspace/orch/boot.sh > /workspace/orch/boot.out 2>&1 &

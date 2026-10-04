#!/bin/sh
# orch shim            run the OpenAI-compatible shim
# orch <command> ...   run deploy.py <command> (check, apply, status, watch, ...)
set -e
case "${1:-shim}" in
    shim) exec python /app/shim/shim.py ;;
    *)    exec python /app/deploy/deploy.py "$@" ;;
esac

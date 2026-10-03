#!/usr/bin/env bash
# Boots one serverless worker: llama-server + Vast's PyWorker.
#
# The PyWorker decides when the worker can take traffic by watching MODEL_LOG
# for marker lines. This script owns those markers:
#   ORCH_READY  — written only after the model downloaded, verified, loaded and
#                 passed a real text + image request
#   ORCH_FATAL  — written on any failure (bad GPU/driver, download, checksum,
#                 load, smoke test, boot deadline, llama-server dying). The
#                 PyWorker reports the error and the autoscaler drops the worker
#                 instead of billing for a container that will never serve.
#
# Runs on every container start, so it must be idempotent: a cold worker
# restarting finds its verified weights on disk and skips the download.
set -uo pipefail

ORCH_DIR="${ORCH_DIR:-/workspace/orch}"
MODEL_LOG="${MODEL_LOG:-$ORCH_DIR/model.log}"
STATE_FILE="$ORCH_DIR/state"
READY_MARK="ORCH_READY"
FATAL_MARK="ORCH_FATAL"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/app/llama-server}"
LLAMA_HOST="127.0.0.1"
LLAMA_PORT="18000"   # pyworker's workers/openai/core.py hardcodes this port
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-model}"
LLAMA_CTX="${LLAMA_CTX:-32768}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-2}"
LLAMA_CACHE_TYPE="${LLAMA_CACHE_TYPE:-q8_0}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
MIN_VRAM_GB="${MIN_VRAM_GB:-20}"
BOOT_DEADLINE="${BOOT_DEADLINE:-2400}"      # seconds from container start to ORCH_READY
LOAD_TIMEOUT="${LOAD_TIMEOUT:-900}"         # seconds for llama-server to report healthy

PYWORKER_REPO="${PYWORKER_REPO:-https://github.com/vast-ai/pyworker}"
PYWORKER_REF="${PYWORKER_REF:-60cfeca889f979fd73cf9e00adcd7b0ebc016fdc}"
VAST_SDK_VERSION="${VAST_SDK_VERSION:-1.8.3}"
ORCH_SKIP_PYWORKER="${ORCH_SKIP_PYWORKER:-0}"   # 1 = manual test rental, no serverless

mkdir -p "$ORCH_DIR"
export LD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER_BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

log(){ echo "[$(date -u '+%H:%M:%S')] [boot] $*" | tee -a "$MODEL_LOG"; }

LLAMA_PID=""
fatal(){
    echo "fatal" > "$STATE_FILE"
    echo "[$(date -u '+%H:%M:%S')] [boot] $FATAL_MARK: $*" | tee -a "$MODEL_LOG" >&2
    [ -n "$LLAMA_PID" ] && kill "$LLAMA_PID" 2>/dev/null
    exit 1
}

# ── fresh log for this boot ───────────────────────────────────────────────────
# The PyWorker reads markers from MODEL_LOG; a stale ORCH_READY from the last
# boot must not mark this one ready before the model is loaded.
[ -f "$MODEL_LOG" ] && mv -f "$MODEL_LOG" "$MODEL_LOG.prev"
: > "$MODEL_LOG"
echo "booting" > "$STATE_FILE"
log "boot start (orch ref ${ORCH_REF:-unknown})"

# ── boot deadline watchdog ────────────────────────────────────────────────────
(
    sleep "$BOOT_DEADLINE"
    if [ "$(cat "$STATE_FILE" 2>/dev/null)" = "booting" ]; then
        echo "fatal" > "$STATE_FILE"
        echo "[$(date -u '+%H:%M:%S')] [boot] $FATAL_MARK: not ready within ${BOOT_DEADLINE}s" >> "$MODEL_LOG"
        pkill -f "$LLAMA_SERVER_BIN" 2>/dev/null
    fi
) &

# ── system packages ───────────────────────────────────────────────────────────
# python3/git/openssl/curl are required (fetch + smoke test, and the PyWorker's
# bootstrap clones a repo and signs a TLS cert). aria2c only speeds up downloads.
apt_install(){
    export DEBIAN_FRONTEND=noninteractive
    for i in 1 2 3; do
        apt-get update -qq >/dev/null 2>&1 \
            && apt-get install -y -qq --no-install-recommends ca-certificates "$@" >/dev/null 2>&1 \
            && return 0
        sleep $((i * 10))
    done
    return 1
}
need=()
for pair in python3:python3 git:git openssl:openssl curl:curl; do
    command -v "${pair%%:*}" >/dev/null 2>&1 || need+=("${pair##*:}")
done
if [ ${#need[@]} -gt 0 ]; then
    log "installing: ${need[*]}"
    apt_install "${need[@]}" || fatal "apt-get install failed for: ${need[*]}"
fi
if ! command -v aria2c >/dev/null 2>&1 && [ "${ORCH_SKIP_APT:-0}" != "1" ]; then
    apt_install aria2 || log "warn: aria2 unavailable, downloads will use curl"
fi
for b in python3 git openssl curl; do
    command -v "$b" >/dev/null 2>&1 || fatal "$b missing after install"
done

# ── start the PyWorker early, so the autoscaler sees this worker loading ──────
if [ "$ORCH_SKIP_PYWORKER" != "1" ]; then
    export BACKEND=llama
    export MODEL_LOG
    export MODEL_LOAD_LOG_MSG="$READY_MARK"
    export MODEL_ERROR_LOG_MSGS="$FATAL_MARK
error loading model
failed to load model"
    export MODEL_HEALTH_ENDPOINT="/health"
    export LLAMA_MODEL="$SERVED_MODEL_NAME"
    export PYWORKER_REPO PYWORKER_REF
    export SDK_VERSION="$VAST_SDK_VERSION"
    export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
    ss="$ORCH_DIR/start_server.sh"
    raw="https://raw.githubusercontent.com/${PYWORKER_REPO#https://github.com/}/$PYWORKER_REF/start_server.sh"
    if ! curl -fsSL --retry 5 --retry-delay 3 --max-time 60 "$raw" -o "$ss.new"; then
        [ -s "$ss" ] || fatal "could not download pyworker start_server.sh ($raw)"
        log "warn: using cached start_server.sh"
    else
        mv -f "$ss.new" "$ss"
    fi
    log "starting pyworker (ref ${PYWORKER_REF:0:12}, sdk $VAST_SDK_VERSION)"
    nohup bash "$ss" > "$ORCH_DIR/pyworker-boot.log" 2>&1 &
else
    log "ORCH_SKIP_PYWORKER=1: manual mode, no serverless worker"
fi

# ── GPU and driver sanity, before paying for a download ───────────────────────
[ -x "$LLAMA_SERVER_BIN" ] || fatal "llama-server not found at $LLAMA_SERVER_BIN (wrong image?)"
if command -v nvidia-smi >/dev/null 2>&1; then
    vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
              | awk '{s+=$1} END {print s+0}')
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | paste -sd ';')"
    [ "${vram_mb:-0}" -ge $((MIN_VRAM_GB * 1000)) ] \
        || fatal "only ${vram_mb:-0} MiB VRAM, need >= ${MIN_VRAM_GB} GB"
else
    fatal "nvidia-smi not available — no GPU visible in this container"
fi
devices=$("$LLAMA_SERVER_BIN" --list-devices 2>&1)
if ! grep -qi "cuda" <<<"$devices"; then
    log "llama-server --list-devices said: $(head -c 600 <<<"$devices")"
    fatal "llama-server cannot see a CUDA device (driver too old for this image?)"
fi
log "llama.cpp sees: $(grep -i cuda <<<"$devices" | head -3 | paste -sd ';')"

# ── weights ───────────────────────────────────────────────────────────────────
export PATHS_ENV="$ORCH_DIR/paths.env"
rm -f "$PATHS_ENV"
python3 "$ORCH_DIR/fetch_model.py" 2>&1 | tee -a "$MODEL_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || fatal "model fetch failed (exit $rc)"
# shellcheck disable=SC1090
. "$PATHS_ENV"
[ -s "${MODEL_PATH:-}" ] || fatal "fetch reported success but MODEL_PATH is missing"
export MODEL_PATH MMPROJ_PATH="${MMPROJ_PATH:-}"

# ── llama-server ──────────────────────────────────────────────────────────────
help=$("$LLAMA_SERVER_BIN" --help 2>&1)
has(){ grep -q -- "$1" <<<"$help"; }

args=(-m "$MODEL_PATH" --host "$LLAMA_HOST" --port "$LLAMA_PORT"
      -a "$SERVED_MODEL_NAME" -c "$LLAMA_CTX" -np "$LLAMA_PARALLEL" -ngl 999
      -ctk "$LLAMA_CACHE_TYPE" -ctv "$LLAMA_CACHE_TYPE")
[ -n "$MMPROJ_PATH" ] && args+=(--mmproj "$MMPROJ_PATH")
has "--kv-unified"       && args+=(--kv-unified)
has "--flash-attn"       && args+=(-fa on)
has "--jinja"            && args+=(--jinja)
has "--reasoning-budget" && args+=(--reasoning-budget "${LLAMA_REASONING_BUDGET:-0}")
has "--no-webui"         && args+=(--no-webui)
# LLAMA_EXTRA_ARGS: whitespace- or ';'-separated, since Vast env values can't
# always carry spaces.
if [ -n "$LLAMA_EXTRA_ARGS" ]; then
    read -r -a extra <<<"${LLAMA_EXTRA_ARGS//;/ }"
    args+=("${extra[@]}")
fi

log "launching llama-server: ${args[*]}"
"$LLAMA_SERVER_BIN" "${args[@]}" >> "$MODEL_LOG" 2>&1 &
LLAMA_PID=$!

t0=$SECONDS
until curl -sf "http://$LLAMA_HOST:$LLAMA_PORT/health" >/dev/null 2>&1; do
    kill -0 "$LLAMA_PID" 2>/dev/null || fatal "llama-server exited during load (see log above)"
    [ $((SECONDS - t0)) -lt "$LOAD_TIMEOUT" ] || fatal "llama-server not healthy after ${LOAD_TIMEOUT}s"
    [ "$(cat "$STATE_FILE" 2>/dev/null)" = "fatal" ] && fatal "boot deadline hit during load"
    sleep 3
done
log "llama-server healthy after $((SECONDS - t0))s"

# ── prove it works ────────────────────────────────────────────────────────────
LLAMA_URL="http://$LLAMA_HOST:$LLAMA_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    python3 "$ORCH_DIR/smoke_test.py" 2>&1 | tee -a "$MODEL_LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || fatal "smoke test failed"

[ "$(cat "$STATE_FILE" 2>/dev/null)" = "fatal" ] && fatal "boot deadline hit"
echo "ready" > "$STATE_FILE"
echo "[$(date -u '+%H:%M:%S')] [boot] $READY_MARK model=$SERVED_MODEL_NAME" >> "$MODEL_LOG"
log "worker ready"

# ── stay up with llama-server; report if it dies ──────────────────────────────
wait "$LLAMA_PID"
rc=$?
LLAMA_PID=""
fatal "llama-server exited with code $rc"

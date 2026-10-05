#!/usr/bin/env bash
# Boots one serverless worker: llama-server + Vast's PyWorker.
#
# The PyWorker decides when the worker can take traffic by watching MODEL_LOG
# for marker lines. This script owns those markers:
#   ORCH_READY  — written only after the model downloaded, verified, loaded and
#                 passed a real text + image request (and an embedding
#                 request, when an embedding model is configured)
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
# worker/router.py takes LLAMA_PORT, in front of the chat llama-server on
# CHAT_PORT and, with an embedding model (EMBED_FILE), the embedding one on
# EMBED_PORT. It also answers /orch/info for the shim's status page.
CHAT_PORT="18010"
EMBED_PORT="18011"
EMBED_SERVED_NAME="${EMBED_SERVED_NAME:-}"
EMBED_CTX="${EMBED_CTX:-4096}"
EMBED_POOLING="${EMBED_POOLING:-last}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-model}"
LLAMA_CTX="${LLAMA_CTX:-32768}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-2}"
LLAMA_CACHE_TYPE="${LLAMA_CACHE_TYPE:-q8_0}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
MIN_VRAM_GB="${MIN_VRAM_GB:-20}"
BOOT_DEADLINE="${BOOT_DEADLINE:-2400}"      # seconds to ORCH_READY, not counting the weight download
DOWNLOAD_MAX_S="${DOWNLOAD_MAX_S:-3600}"    # the download's own limit (fetch_model.py enforces it)
DOWNLOAD_METHOD="${DOWNLOAD_METHOD:-auto}"  # auto | hf | direct, see fetch_model.py
HF_HUB_VERSION="${HF_HUB_VERSION:-2.1.1}"   # huggingface_hub[hf_xet] for Xet-stored weights
HF_XET_VERSION="${HF_XET_VERSION:-1.6.0}"
DOWNLOAD_PHASE="downloading and verifying weights"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-900}"         # seconds for llama-server to report healthy
LOAD_HEARTBEAT_S="${LOAD_HEARTBEAT_S:-60}"  # a progress line this often while loading
LIST_DEVICES_TIMEOUT="${LIST_DEVICES_TIMEOUT:-300}"

PYWORKER_REPO="${PYWORKER_REPO:-https://github.com/vast-ai/pyworker}"
PYWORKER_REF="${PYWORKER_REF:-60cfeca889f979fd73cf9e00adcd7b0ebc016fdc}"
VAST_SDK_VERSION="${VAST_SDK_VERSION:-1.8.3}"
ORCH_SKIP_PYWORKER="${ORCH_SKIP_PYWORKER:-0}"   # 1 = manual test rental, no serverless

mkdir -p "$ORCH_DIR"
export LD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER_BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── where log lines go ────────────────────────────────────────────────────────
# MODEL_LOG (the PyWorker reads markers there), this script's stdout (boot.out,
# readable over SSH) and the container's own stdout, which is what
# `vastai logs <id>` shows: onstart runs this script in the background, so
# without that last copy a boot is invisible from outside. Writing there must
# never stop a boot, so SIGPIPE is ignored and errors are dropped.
trap '' PIPE
if [ -z "${ORCH_CONSOLE+set}" ]; then
    ORCH_CONSOLE=""
    if [ -w /proc/1/fd/1 ] && [ "$(readlink /proc/1/fd/1)" != "$(readlink /proc/$$/fd/1)" ]; then
        ORCH_CONSOLE=/proc/1/fd/1
    fi
fi
console(){ [ -n "$ORCH_CONSOLE" ] && printf '%s\n' "$*" >> "$ORCH_CONSOLE" 2>/dev/null; return 0; }
stamp(){ echo "[$(date -u '+%H:%M:%S')] [boot] $*"; }
log(){ local l; l=$(stamp "$*"); echo "$l" | tee -a "$MODEL_LOG"; console "$l"; }
# phase: what the boot is doing now, so a deadline failure can say where it was.
phase(){ echo "$*" > "$ORCH_DIR/phase"; log "$*"; }
# with_timeout SECS CMD...: steps that have hung silently on real hosts get a limit.
with_timeout(){
    local s="$1"; shift
    if command -v timeout >/dev/null 2>&1; then timeout --kill-after=10 "$s" "$@"; else "$@"; fi
}

# GPUs without prebuilt kernels in the image run PTX the driver compiles on
# first use. A cache big enough to hold all of it means that happens once per
# instance instead of on every load (the driver's default cache is smaller).
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"

# ── stop paying for a dead worker ─────────────────────────────────────────────
# After ORCH_FATAL a running PyWorker reports the error; Vast marks the worker
# Error (not billed) and the autoscaler deals with it, so we leave it alone.
# When nothing can report it (the PyWorker never started or died, or this is
# a manual rental), the instance stops or destroys itself ORCH_FATAL_GRACE
# seconds later. This is the in-container route Vast documents
# (`vastai stop|destroy instance $CONTAINER_ID`, authorised by the per-instance
# CONTAINER_API_KEY), done with curl against the same REST calls the CLI makes
# because the llama.cpp image ships no vastai CLI.
VAST_URL="${VAST_URL:-https://console.vast.ai}"
ORCH_FATAL_GRACE="${ORCH_FATAL_GRACE:-600}"
if [ "$ORCH_SKIP_PYWORKER" = "1" ]; then _default_action=stop; else _default_action=destroy; fi
ORCH_FATAL_ACTION="${ORCH_FATAL_ACTION:-$_default_action}"   # destroy | stop | none
ORCH_MANUAL_TTL="${ORCH_MANUAL_TTL:-14400}"   # manual rental stops itself after this many seconds; 0 = never
CLEANUP_MARK="$ORCH_DIR/.cleanup-scheduled"
PYWORKER_PIDFILE="$ORCH_DIR/pyworker.pid"

self_terminate(){   # $1 = destroy|stop, $2 = why
    local action="$1" why="$2" url code body i
    if [ -z "${CONTAINER_ID:-}" ] || [ -z "${CONTAINER_API_KEY:-}" ]; then
        log "warn: $why, but CONTAINER_ID/CONTAINER_API_KEY are unset; can't $action this instance"
        return 1
    fi
    url="$VAST_URL/api/v0/instances/$CONTAINER_ID/"
    log "$why: asking Vast to $action instance $CONTAINER_ID"
    for i in 1 2 3; do
        body="$ORCH_DIR/self_terminate.resp"
        if [ "$action" = destroy ]; then
            code=$(curl -s -o "$body" -w '%{http_code}' --max-time 30 -X DELETE \
                -H "Authorization: Bearer $CONTAINER_API_KEY" -H 'Content-Type: application/json' -d '{}' "$url")
        else
            code=$(curl -s -o "$body" -w '%{http_code}' --max-time 30 -X PUT \
                -H "Authorization: Bearer $CONTAINER_API_KEY" -H 'Content-Type: application/json' \
                -d '{"state":"stopped"}' "$url")
        fi
        # Vast answers {"success": true, ...}; a 200 with success false is a refusal.
        if [ "${code:0:1}" = 2 ] && ! grep -Eq '"success" *: *false' "$body" 2>/dev/null; then
            log "instance $CONTAINER_ID: $action accepted"
            return 0
        fi
        log "warn: $action returned HTTP $code: $(head -c 300 "$body" 2>/dev/null) (attempt $i)"
        sleep $((i * 10))
    done
    return 1
}

schedule_fatal_cleanup(){
    [ "$ORCH_FATAL_ACTION" = none ] && return 0
    mkdir "$CLEANUP_MARK" 2>/dev/null || return 0   # once per boot
    (
        sleep "$ORCH_FATAL_GRACE"
        # A new boot rewrites the state file; only act on this boot's failure.
        [ "$(cat "$STATE_FILE" 2>/dev/null)" = "fatal" ] || exit 0
        pid=$(cat "$PYWORKER_PIDFILE" 2>/dev/null)
        if [ "$ORCH_SKIP_PYWORKER" != "1" ] && [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log "PyWorker is running and reports the error; leaving this worker to the autoscaler"
            exit 0
        fi
        self_terminate "$ORCH_FATAL_ACTION" "still running ${ORCH_FATAL_GRACE}s after $FATAL_MARK"
    ) </dev/null >/dev/null 2>&1 &
}

LLAMA_PID=""; EMBED_PID=""; ROUTER_PID=""
fatal(){
    echo "fatal" > "$STATE_FILE"
    local l; l=$(stamp "$FATAL_MARK: $*")
    echo "$l" | tee -a "$MODEL_LOG" >&2
    console "$l"
    for p in $LLAMA_PID $EMBED_PID $ROUTER_PID; do kill "$p" 2>/dev/null; done
    schedule_fatal_cleanup
    exit 1
}

# ── fresh log for this boot ───────────────────────────────────────────────────
# The PyWorker reads markers from MODEL_LOG; a stale ORCH_READY from the last
# boot must not mark this one ready before the model is loaded.
[ -f "$MODEL_LOG" ] && mv -f "$MODEL_LOG" "$MODEL_LOG.prev"
: > "$MODEL_LOG"
echo "booting" > "$STATE_FILE"
rmdir "$CLEANUP_MARK" 2>/dev/null
rm -f "$PYWORKER_PIDFILE"
echo "starting" > "$ORCH_DIR/phase"
log "boot start (orch ref ${ORCH_REF:-unknown})"

# ── manual rentals don't run forever ──────────────────────────────────────────
if [ "$ORCH_SKIP_PYWORKER" = "1" ] && [ "$ORCH_MANUAL_TTL" -gt 0 ] 2>/dev/null; then
    log "manual mode: this instance stops itself after ${ORCH_MANUAL_TTL}s (ORCH_MANUAL_TTL, 0 = never)"
    ( sleep "$ORCH_MANUAL_TTL"
      self_terminate stop "manual rental reached ORCH_MANUAL_TTL=${ORCH_MANUAL_TTL}s"
    ) </dev/null >/dev/null 2>&1 &
fi

# ── boot deadline watchdog ────────────────────────────────────────────────────
# BOOT_DEADLINE counts everything except the weight download, which has its
# own limits in fetch_model.py (minimum speed, stall restart, DOWNLOAD_MAX_S):
# a slow but steady download is not a hung boot. As a backstop for a download
# step that hangs outright, the whole boot also gets BOOT_DEADLINE +
# DOWNLOAD_MAX_S + 300 seconds of wall-clock time.
(
    counted=0; t0=$SECONDS; hard=$((BOOT_DEADLINE + DOWNLOAD_MAX_S + 300))
    while [ "$(cat "$STATE_FILE" 2>/dev/null)" = "booting" ]; do
        sleep 2
        [ "$(cat "$ORCH_DIR/phase" 2>/dev/null)" = "$DOWNLOAD_PHASE" ] || counted=$((counted + 2))
        if [ "$counted" -ge "$BOOT_DEADLINE" ]; then
            why="not ready within ${BOOT_DEADLINE}s"
        elif [ $((SECONDS - t0)) -ge "$hard" ]; then
            why="not ready within ${hard}s including the download"
        else
            continue
        fi
        [ "$(cat "$STATE_FILE" 2>/dev/null)" = "booting" ] || break
        echo "fatal" > "$STATE_FILE"
        log "$FATAL_MARK: $why (still at: $(cat "$ORCH_DIR/phase" 2>/dev/null))"
        schedule_fatal_cleanup
        # pkill exits 1 when nothing matched; anything higher means it couldn't run.
        pkill -f "$LLAMA_SERVER_BIN" 2>>"$MODEL_LOG"; rc=$?
        [ "$rc" -le 1 ] || log "warn: pkill failed (exit $rc); llama-server may still be running"
        break
    done
) &

# ── system packages ───────────────────────────────────────────────────────────
# python3/git/openssl/curl are required (fetch + smoke test, and the PyWorker's
# bootstrap clones a repo and signs a TLS cert). aria2c only speeds up downloads.
apt_install(){
    export DEBIAN_FRONTEND=noninteractive
    for i in 1 2 3; do
        with_timeout 300 apt-get update -qq >/dev/null 2>&1 \
            && with_timeout 600 apt-get install -y -qq --no-install-recommends ca-certificates "$@" >/dev/null 2>&1 \
            && return 0
        log "warn: apt-get failed or timed out (attempt $i)"
        sleep $((i * 10))
    done
    return 1
}
need=()
for pair in python3:python3 git:git openssl:openssl curl:curl; do
    command -v "${pair%%:*}" >/dev/null 2>&1 || need+=("${pair##*:}")
done
if [ ${#need[@]} -gt 0 ]; then
    [ "${ORCH_SKIP_APT:-0}" != "1" ] || fatal "missing: ${need[*]} (ORCH_SKIP_APT=1, not installing)"
    phase "installing: ${need[*]}"
    apt_install "${need[@]}" || fatal "apt-get install failed for: ${need[*]}"
fi
if ! command -v aria2c >/dev/null 2>&1 && [ "${ORCH_SKIP_APT:-0}" != "1" ]; then
    phase "installing: aria2"
    apt_install aria2 || log "warn: aria2 unavailable, downloads will use curl"
fi
for b in python3 git openssl curl; do
    command -v "$b" >/dev/null 2>&1 || fatal "$b missing after install"
done

# ── huggingface_hub + hf_xet for the weights ──────────────────────────────────
# Large files on the Hub are stored with Xet; hf_xet fetches them from Xet
# storage directly and in parallel, much faster than HTTP range requests
# through the CDN bridge. Pinned versions, in a venv kept on disk so a cold
# restart reuses it. Any failure here falls back to aria2c/curl.
setup_hf(){
    local venv="$ORCH_DIR/hfvenv"
    if "$venv/bin/python" -c "import sys, huggingface_hub, hf_xet; sys.exit(huggingface_hub.__version__ != '$HF_HUB_VERSION')" 2>/dev/null; then
        HF_PYTHON="$venv/bin/python"; return 0
    fi
    rm -rf "$venv"
    if ! python3 -m venv "$venv" >/dev/null 2>&1; then
        # Debian/Ubuntu ship venv support (ensurepip) as a separate package.
        [ "${ORCH_SKIP_APT:-0}" != "1" ] || return 1
        apt_install python3-venv || return 1
        rm -rf "$venv"
        python3 -m venv "$venv" >/dev/null 2>&1 || return 1
    fi
    with_timeout 300 "$venv/bin/pip" install -q --disable-pip-version-check \
        "huggingface_hub[hf_xet]==$HF_HUB_VERSION" "hf_xet==$HF_XET_VERSION" >>"$MODEL_LOG" 2>&1 || return 1
    HF_PYTHON="$venv/bin/python"
}
if [ "$DOWNLOAD_METHOD" != "direct" ]; then
    if [ -n "${HF_PYTHON:-}" ] && [ -x "$HF_PYTHON" ]; then
        :   # provided by the environment
    else
        phase "installing: huggingface_hub $HF_HUB_VERSION + hf_xet $HF_XET_VERSION"
        HF_PYTHON=""
        setup_hf || { HF_PYTHON=""; log "warn: could not set up huggingface_hub; downloads will use aria2c/curl"; }
    fi
    export HF_PYTHON
fi

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
    # Our worker.py (stock llama routes + /v1/embeddings) goes at the root of
    # the pyworker checkout; start_server.sh at PYWORKER_REF runs it in place of
    # workers/llama/worker.py, and only clones when the checkout is missing.
    [ -n "${EMBED_FILE:-}" ] && export EMBED_SERVED_NAME
    pw_dir="$WORKSPACE_DIR/vast-pyworker"
    if [ ! -d "$pw_dir/.git" ]; then
        rm -rf "$pw_dir"
        log "cloning $PYWORKER_REPO"
        git clone -q "$PYWORKER_REPO" "$pw_dir" && git -C "$pw_dir" checkout -q "$PYWORKER_REF" \
            || fatal "could not check out $PYWORKER_REPO at $PYWORKER_REF"
    fi
    cp -f "$ORCH_DIR/pyworker_worker.py" "$pw_dir/worker.py" || fatal "could not install pyworker worker.py"
    ss="$ORCH_DIR/start_server.sh"
    raw_base="${PYWORKER_RAW_BASE:-https://raw.githubusercontent.com/${PYWORKER_REPO#https://github.com/}}"
    raw="$raw_base/$PYWORKER_REF/start_server.sh"
    if ! curl -fsSL --retry 5 --retry-delay 3 --max-time 60 "$raw" -o "$ss.new"; then
        [ -s "$ss" ] || fatal "could not download pyworker start_server.sh ($raw)"
        log "warn: using cached start_server.sh"
    else
        mv -f "$ss.new" "$ss"
    fi
    log "starting pyworker (ref ${PYWORKER_REF:0:12}, sdk $VAST_SDK_VERSION)"
    nohup bash "$ss" > "$ORCH_DIR/pyworker-boot.log" 2>&1 &
    echo $! > "$PYWORKER_PIDFILE"
else
    log "ORCH_SKIP_PYWORKER=1: manual mode, no serverless worker"
fi

# ── GPU and driver sanity, before paying for a download ───────────────────────
phase "checking GPU and driver"
[ -x "$LLAMA_SERVER_BIN" ] || fatal "llama-server not found at $LLAMA_SERVER_BIN (wrong image?)"
if command -v nvidia-smi >/dev/null 2>&1; then
    vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
              | awk '{s+=$1} END {print s+0}')
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | paste -sd ';')"
    [ "${vram_mb:-0}" -ge $((MIN_VRAM_GB * 1000)) ] \
        || fatal "only ${vram_mb:-0} MiB VRAM, need >= ${MIN_VRAM_GB} GB"
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if awk -v c="$cc" 'BEGIN { exit !(c ~ /^[0-9]+\.[0-9]+$/ && c + 0 < 8.6) }'; then
        log "note: compute capability $cc likely has no prebuilt kernels in this llama.cpp image;" \
            "the first load compiles them, which can take many minutes"
    fi
else
    fatal "nvidia-smi not available — no GPU visible in this container"
fi
devices=$(with_timeout "$LIST_DEVICES_TIMEOUT" "$LLAMA_SERVER_BIN" --list-devices 2>&1); rc=$?
[ "$rc" -ne 124 ] && [ "$rc" -ne 137 ] \
    || fatal "llama-server --list-devices did not finish in ${LIST_DEVICES_TIMEOUT}s (GPU or driver hung?)"
if ! grep -qi "cuda" <<<"$devices"; then
    log "llama-server --list-devices said: $(head -c 600 <<<"$devices")"
    fatal "llama-server cannot see a CUDA device (driver too old for this image?)"
fi
log "llama.cpp sees: $(grep -i cuda <<<"$devices" | head -3 | paste -sd ';')"

# ── will it fit? ──────────────────────────────────────────────────────────────
# worker/vram.py reads the model headers (from the Hub, a few MB each, before
# anything is downloaded) and works out whether chat model + projector +
# embedding model fit in the GPU memory that is free, giving up embedding
# context, then embedding cache precision, then chat context (down to
# LLAMA_CTX_MIN) if they don't. A set that can't fit fails here, before the
# download, saying by how much. If the Hub headers can't be read, the same
# check runs on the downloaded files instead.
PLAN_ENV="$ORCH_DIR/plan.env"
# vram_plan STAGE: runs the planner, its lines to the log; returns its exit status.
vram_plan(){
    rm -f "$PLAN_ENV"
    python3 "$ORCH_DIR/vram.py" plan --stage "$1" --env "$PLAN_ENV" --json "$ORCH_DIR/plan-$1.json" 2>&1 \
        | while IFS= read -r line; do echo "$line" | tee -a "$MODEL_LOG"; console "$line"; done
    return "${PIPESTATUS[0]}"
}
phase "checking that the models fit in GPU memory"
vram_plan pre; rc=$?
case "$rc" in
    0) ;;
    7) fatal "models don't fit in this GPU's free memory (see the [vram] lines above)" ;;
    *) log "warn: couldn't size GPU memory from the Hub headers (exit $rc); checking the downloaded files instead" ;;
esac

# ── weights ───────────────────────────────────────────────────────────────────
export PATHS_ENV="$ORCH_DIR/paths.env"
rm -f "$PATHS_ENV"
phase "$DOWNLOAD_PHASE"
python3 "$ORCH_DIR/fetch_model.py" 2>&1 | while IFS= read -r line; do echo "$line" | tee -a "$MODEL_LOG"; console "$line"; done
rc=${PIPESTATUS[0]}
[ "$rc" -ne 6 ] || fatal "weights download too slow on this host (see the [fetch] line above)"
[ "$rc" -eq 0 ] || fatal "model fetch failed (exit $rc)"
# shellcheck disable=SC1090
. "$PATHS_ENV"
[ -s "${MODEL_PATH:-}" ] || fatal "fetch reported success but MODEL_PATH is missing"
export MODEL_PATH MMPROJ_PATH="${MMPROJ_PATH:-}" EMBED_PATH="${EMBED_PATH:-}"

# ── llama-server ──────────────────────────────────────────────────────────────
help=$(with_timeout 120 "$LLAMA_SERVER_BIN" --help 2>&1)
has(){ grep -q -- "$1" <<<"$help"; }

# Sizes from the downloaded files (sha-verified, so an unreadable header is a
# real problem, not a network one).
phase "checking that the models fit in GPU memory"
vram_plan local; rc=$?
[ "$rc" -ne 7 ] || fatal "models don't fit in this GPU's free memory (see the [vram] lines above)"
[ "$rc" -eq 0 ] || fatal "couldn't size GPU memory from the model files (vram.py exit $rc, see above)"
# shellcheck disable=SC1090
. "$PLAN_ENV"

# serve_log NAME: one llama-server's output, line by line, to its own log
# ($ORCH_DIR/NAME.log), MODEL_LOG and the container console, prefixed with
# NAME, so its errors show in `vastai logs`. Once the worker is ready only
# warnings and errors go to the console (request logs would flood it).
QUIET_MARK="$ORCH_DIR/.serving"
rm -f "$QUIET_MARK"
serve_log(){
    local name="$1" line
    : > "$ORCH_DIR/$name.log"
    while IFS= read -r line; do
        printf '%s\n' "$line" >> "$ORCH_DIR/$name.log"
        printf '[%s] %s\n' "$name" "$line" >> "$MODEL_LOG"
        if [ ! -e "$QUIET_MARK" ] || [[ "$line" =~ [Ee]rror|ERROR|[Ww]arn|WARN|[Ff]ail|out\ of\ memory ]]; then
            console "[$name] $line"
        fi
    done
}
# load_error NAME: the line that best says why a server failed to load.
load_error(){
    local f="$ORCH_DIR/$1.log" l
    sleep 1   # let serve_log write the server's last lines
    l=$(grep -m1 -iE 'out of memory|cudaMalloc failed|failed to allocate' "$f" 2>/dev/null)
    [ -n "$l" ] || l=$(grep -iE 'error|failed' "$f" 2>/dev/null | tail -n 1)
    [ -n "$l" ] || l=$(tail -n 1 "$f" 2>/dev/null)
    printf '%s' "${l:0:300}"
}
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
# wait_healthy URL WHAT PID: until URL answers 200, with a heartbeat; fatal if
# PID exits, the load takes too long or the boot deadline passes.
wait_healthy(){
    local url="$1" what="$2" pid="$3" name="$4" t0=$SECONDS next_beat=$LOAD_HEARTBEAT_S
    until curl -sf "$url" >/dev/null 2>&1; do
        kill -0 "$pid" 2>/dev/null || fatal "$what exited during load: $(load_error "$name")"
        [ -z "$ROUTER_PID" ] || kill -0 "$ROUTER_PID" 2>/dev/null || fatal "router exited during load (see log above)"
        [ $((SECONDS - t0)) -lt "$LOAD_TIMEOUT" ] || fatal "$what not healthy after ${LOAD_TIMEOUT}s"
        [ "$(cat "$STATE_FILE" 2>/dev/null)" = "fatal" ] && fatal "boot deadline hit during load"
        if [ $((SECONDS - t0)) -ge "$next_beat" ]; then
            log "still loading $what after $((SECONDS - t0))s; it says: $(tail -n 1 "$ORCH_DIR/$name.log" 2>/dev/null | cut -c1-200)"
            next_beat=$((next_beat + LOAD_HEARTBEAT_S))
        fi
        sleep 1
    done
    log "$what healthy after $((SECONDS - t0))s"
}

phase "loading the model"
# The embedding server starts first and alone: the chat server is then sized
# from the memory that is actually left, not from an estimate of what the
# embedding server will take.
if [ -n "${EMBED_PATH:-}" ]; then
    # Each input must fit in one batch, so batch = context; one slot gets all
    # of it (inputs in a request are processed one after another).
    ectx="${PLAN_EMBED_CTX:-0}"
    eargs=(-m "$EMBED_PATH" --host "$LLAMA_HOST" --port "$EMBED_PORT"
           -a "${EMBED_SERVED_NAME:-embedding}" --embedding --pooling "$EMBED_POOLING" -np 1)
    case "$(tr '[:upper:]' '[:lower:]' <<<"${EMBED_GPU:-1}")" in
        0|false|no|off)
            ectx="$EMBED_CTX"
            eargs+=(-c "$ectx" -b "$ectx" -ub "$ectx" -ngl 0)
            log "embedding model on the CPU ([embedding] gpu = false)" ;;
        *)
            [ "$ectx" -gt 0 ] 2>/dev/null || fatal "the GPU memory plan has no embedding context (see the [vram] lines above)"
            eargs+=(-c "$ectx" -b "$ectx" -ub "$ectx" -ngl 999)
            [ -n "${PLAN_EMBED_CACHE:-}" ] && eargs+=(-ctk "$PLAN_EMBED_CACHE" -ctv "$PLAN_EMBED_CACHE") ;;
    esac
    # Flash attention keeps the attention buffer small at a large batch size.
    has "--flash-attn" && eargs+=(-fa on)
    has "--no-webui"   && eargs+=(--no-webui)
    log "launching embedding llama-server: ${eargs[*]}"
    used0=$(gpu_used)
    "$LLAMA_SERVER_BIN" "${eargs[@]}" > >(serve_log embed) 2>&1 &
    EMBED_PID=$!
    wait_healthy "http://$LLAMA_HOST:$EMBED_PORT/health" "embedding llama-server" "$EMBED_PID" embed
    log "VRAM after embedding: $(gpu_used) MiB used (was $used0 MiB before it started; estimated ${PLAN_EMBED_EST_MIB:-?} MiB)"
    vram_plan chat; rc=$?
    [ "$rc" -ne 7 ] || fatal "chat model doesn't fit in the GPU memory left after the embedding server (see the [vram] lines above)"
    [ "$rc" -eq 0 ] || fatal "couldn't size GPU memory for the chat model (vram.py exit $rc, see above)"
    # shellcheck disable=SC1090
    . "$PLAN_ENV"
fi

chat_port="$CHAT_PORT"
# -ngl 999 keeps every layer on the GPU (and turns llama.cpp's own --fit off):
# the plan above already chose settings that fit.
args=(-m "$MODEL_PATH" --host "$LLAMA_HOST" --port "$chat_port"
      -a "$SERVED_MODEL_NAME" -c "${PLAN_CTX:-$LLAMA_CTX}" -np "$LLAMA_PARALLEL" -ngl 999
      -b "${PLAN_BATCH:-2048}" -ub "${PLAN_UBATCH:-512}"
      -ctk "$LLAMA_CACHE_TYPE" -ctv "$LLAMA_CACHE_TYPE")
[ -n "$MMPROJ_PATH" ] && args+=(--mmproj "$MMPROJ_PATH")
has "--kv-unified"       && args+=(--kv-unified)
has "--flash-attn"       && args+=(-fa on)
has "--jinja"            && args+=(--jinja)
has "--reasoning-budget" && args+=(--reasoning-budget "${LLAMA_REASONING_BUDGET:-0}")
# A budget of 0 alone leaves thinking on in templates that open every prompt
# with a think turn (WaifuGemma4: the reply is often just '<|thought|>',
# rental 54351312), so turn it off in the template too.
[ "${LLAMA_REASONING_BUDGET:-0}" = 0 ] && has "--chat-template-kwargs" \
    && args+=(--chat-template-kwargs '{"enable_thinking":false}')
has "--no-webui"         && args+=(--no-webui)
# LLAMA_EXTRA_ARGS: whitespace- or ';'-separated, since Vast env values can't
# always carry spaces.
if [ -n "$LLAMA_EXTRA_ARGS" ]; then
    read -r -a extra <<<"${LLAMA_EXTRA_ARGS//;/ }"
    args+=("${extra[@]}")
fi

log "launching llama-server: ${args[*]}"
used0=$(gpu_used)
"$LLAMA_SERVER_BIN" "${args[@]}" > >(serve_log chat) 2>&1 &
LLAMA_PID=$!
wait_healthy "http://$LLAMA_HOST:$chat_port/health" "llama-server" "$LLAMA_PID" chat

embed_url=""
[ -n "${EMBED_PATH:-}" ] && embed_url="http://$LLAMA_HOST:$EMBED_PORT"
ROUTER_PORT="$LLAMA_PORT" CHAT_URL="http://$LLAMA_HOST:$CHAT_PORT" EMBED_URL="$embed_url" ORCH_DIR="$ORCH_DIR" \
    python3 "$ORCH_DIR/router.py" > >(serve_log router) 2>&1 &
ROUTER_PID=$!
wait_healthy "http://$LLAMA_HOST:$LLAMA_PORT/health" "router" "$ROUTER_PID" router

# What llama.cpp actually allocated, next to the plan, so every boot shows
# how close the estimate was.
for n in embed chat; do
    [ -s "$ORCH_DIR/$n.log" ] || continue
    grep -E 'KV buffer size|compute buffer size|model buffer size|CLIP.*buffer size' "$ORCH_DIR/$n.log" 2>/dev/null \
        | sed -E 's/^[[:space:]]+//' | head -n 12 | while IFS= read -r line; do log "[$n] $line"; done
done
log "VRAM after chat: $(gpu_used) MiB used (was $used0 MiB before it started; estimated ${PLAN_CHAT_EST_MIB:-?} MiB, plus a ${PLAN_MARGIN_MIB:-?} MiB margin kept free)"

# ── prove it works ────────────────────────────────────────────────────────────
phase "smoke test"
LLAMA_URL="http://$LLAMA_HOST:$LLAMA_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" LLAMA_REASONING_BUDGET="${LLAMA_REASONING_BUDGET:-0}" \
    EMBED_SERVED_NAME="$([ -n "${EMBED_PATH:-}" ] && echo "${EMBED_SERVED_NAME:-embedding}")" \
    python3 "$ORCH_DIR/smoke_test.py" 2>&1 | while IFS= read -r line; do echo "$line" | tee -a "$MODEL_LOG"; console "$line"; done
[ "${PIPESTATUS[0]}" -eq 0 ] || fatal "smoke test failed"

[ "$(cat "$STATE_FILE" 2>/dev/null)" = "fatal" ] && fatal "boot deadline hit"
echo "ready" > "$STATE_FILE"
echo "ready" > "$ORCH_DIR/phase"
touch "$QUIET_MARK"
log "$READY_MARK model=$SERVED_MODEL_NAME"

# ── stay up with the servers; report the first one that dies ─────────────────
wait -n $LLAMA_PID $EMBED_PID $ROUTER_PID
rc=$?
for p in $LLAMA_PID $EMBED_PID $ROUTER_PID; do
    kill -0 "$p" 2>/dev/null && continue
    case "$p" in
        "$LLAMA_PID") what="llama-server" ;;
        "$EMBED_PID") what="embedding llama-server" ;;
        *) what="router" ;;
    esac
    break
done
fatal "${what:-llama-server} exited with code $rc"

#!/usr/bin/env bash
# End-to-end test of worker/boot.sh with a fake Hub, fake GPU and fake
# llama-server. Runs each failure mode and checks the marker the PyWorker
# would act on. Needs: bash, python3, curl. Uses port 18000 and 18999.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
WORK="$(mktemp -d)"
HF_PORT=18999
VAST_PORT=18998
PASS=0; FAIL=0
HF_PID=""; VAST_PID=""
trap '[ -n "$HF_PID" ] && kill "$HF_PID" 2>/dev/null; [ -n "$VAST_PID" ] && kill "$VAST_PID" 2>/dev/null; pkill -f fake_llama_server.py 2>/dev/null; pkill -f "$WORK/orch/router.py" 2>/dev/null; rm -rf "$WORK"' EXIT

# fake weights: GGUF magic + filler
mkdir -p "$WORK/hub" "$WORK/bin"
python3 - "$WORK/hub" <<'EOF'
import os, sys
d = sys.argv[1]
for name, n in (("model.gguf", 3_000_000), ("mmproj.gguf", 400_000), ("embed.gguf", 200_000)):
    with open(os.path.join(d, name), "wb") as f:
        f.write(b"GGUF" + os.urandom(n))
EOF
cp "$HERE/fakes/fake_llama_server.py" "$WORK/bin/llama-server"
chmod +x "$WORK/bin/llama-server"
cat > "$WORK/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *memory.total\ --format=csv,noheader,nounits*) echo "${FAKE_VRAM_MB:-24576}";;
  *) echo "FAKE GPU, ${FAKE_VRAM_MB:-24576} MiB, 999.99";;
esac
EOF
chmod +x "$WORK/bin/nvidia-smi"

start_hub(){
    [ -n "${HF_PID:-}" ] && kill "$HF_PID" 2>/dev/null && sleep 0.3
    python3 "$HERE/fakes/fake_hf.py" "$HF_PORT" "$WORK/hub" "$@" & HF_PID=$!
    for _ in $(seq 50); do curl -s -o /dev/null "http://127.0.0.1:$HF_PORT/" && break; sleep 0.1; done
}

# run_case NAME EXPECT(ready|fatal|ready_then_fatal) TIMEOUT [ENV=VAL...]
# HOLD=N keeps the boot running N more seconds after the marker shows up.
run_case(){
    local name="$1" expect="$2" timeout="$3"; shift 3
    local orch="$WORK/orch"
    mkdir -p "$orch"
    cp "$REPO/worker/"{boot.sh,fetch_model.py,hf_download.py,smoke_test.py,router.py,pyworker_worker.py} "$orch/"
    rm -f "$orch/model.log" "$orch/model.log.prev" "$orch/console.log"
    pkill -f fake_llama_server.py 2>/dev/null; pkill -f "$WORK/bin/llama-server" 2>/dev/null; sleep 0.3
    env PATH="$WORK/bin:$PATH" ORCH_DIR="$orch" MODEL_LOG="$orch/model.log" \
        ORCH_SKIP_PYWORKER=1 ORCH_SKIP_APT=1 LLAMA_SERVER_BIN="$WORK/bin/llama-server" \
        ORCH_CONSOLE="$orch/console.log" HF_ENDPOINT="http://127.0.0.1:$HF_PORT" MODELS_DIR="$WORK/models" WORKSPACE_DIR="$WORK/workspace" \
        MODEL_REPO=test/repo MODEL_FILE=model.gguf MMPROJ_FILE=mmproj.gguf \
        SERVED_MODEL_NAME=testmodel BOOT_DEADLINE=60 DOWNLOAD_METHOD=direct "$@" \
        setsid bash "$orch/boot.sh" > "$orch/boot.out" 2>&1 &
    local pid=$! got=""
    local deadline=$((SECONDS + timeout))
    while (( SECONDS < deadline )); do
        if [ "$expect" = ready_then_fatal ]; then
            grep -q "ORCH_READY" "$orch/model.log" 2>/dev/null && grep -q "ORCH_FATAL" "$orch/model.log" && { got=ready_then_fatal; break; }
        else
            grep -q "ORCH_FATAL" "$orch/model.log" 2>/dev/null && { got=fatal; break; }
            grep -q "ORCH_READY" "$orch/model.log" 2>/dev/null && { got=ready; break; }
        fi
        sleep 0.5
    done
    [ -n "$got" ] && sleep "${HOLD:-0}"
    kill -- -"$pid" 2>/dev/null; pkill -f "$WORK/bin/llama-server" 2>/dev/null; wait "$pid" 2>/dev/null
    if [ "$got" = "$expect" ]; then
        echo "PASS  $name ($got)"; PASS=$((PASS + 1))
    else
        echo "FAIL  $name: expected $expect, got '${got:-timeout}'"; FAIL=$((FAIL + 1))
        sed 's/^/      /' "$orch/model.log" | tail -25
    fi
    LAST_LOG="$orch/model.log"
    LAST_CONSOLE="$orch/console.log"
}
# check NAME CMD...: one PASS/FAIL line for a condition after a case
check(){
    local name="$1"; shift
    if "$@"; then echo "PASS  $name"; PASS=$((PASS+1)); else echo "FAIL  $name"; FAIL=$((FAIL+1)); fi
}

start_hub
run_case "fresh boot downloads, verifies, gates on smoke test" ready 60
grep -q "ORCH_READY model=testmodel" "$LAST_LOG" || { echo "FAIL  ready marker format"; FAIL=$((FAIL+1)); }
check "boot and fetch lines and ORCH_READY reach the container console" \
    grep -q "ORCH_READY model=testmodel" "$LAST_CONSOLE"
check "fetch output reaches the container console" grep -q "\[fetch\] all files present and verified" "$LAST_CONSOLE"
grep -q -- "--mmproj" "$LAST_LOG" && grep -q -- "--kv-unified" "$LAST_LOG" \
    && echo "PASS  llama-server got --mmproj and --kv-unified" && PASS=$((PASS+1)) \
    || { echo "FAIL  expected flags missing"; FAIL=$((FAIL+1)); }

run_case "cold restart reuses verified weights" ready 60
grep -q "already verified, skipping" "$LAST_LOG" \
    && echo "PASS  no re-download on restart" && PASS=$((PASS+1)) \
    || { echo "FAIL  restart re-downloaded"; FAIL=$((FAIL+1)); }

run_case "too little VRAM fails before downloading" fatal 30 FAKE_VRAM_MB=8000
grep -q "\[fetch\]" "$LAST_LOG" && { echo "FAIL  fetched despite VRAM check"; FAIL=$((FAIL+1)); }

# path_without DIR CMD...: a PATH dir with every command except CMD...
path_without(){
    local dir="$1"; shift; mkdir -p "$dir"
    for d in $WORK/bin ${PATH//:/ }; do
        for f in "$d"/*; do [ -x "$f" ] && [ ! -e "$dir/${f##*/}" ] && ln -s "$f" "$dir/${f##*/}"; done
    done 2>/dev/null
    for c in "$@"; do rm -f "$dir/$c"; done
}
# No openssl, and an apt-get that only records that it was called.
NOSSL="$WORK/nossl"; path_without "$NOSSL" openssl apt-get
printf '#!/bin/sh\necho called >> "%s"\n' "$WORK/apt_calls" > "$NOSSL/apt-get"; chmod +x "$NOSSL/apt-get"
run_case "missing required tool with ORCH_SKIP_APT is fatal, no apt" fatal 20 PATH="$NOSSL"
grep -q "missing: openssl" "$LAST_LOG" && [ ! -e "$WORK/apt_calls" ] \
    && { echo "PASS  ORCH_SKIP_APT respected for required packages"; PASS=$((PASS+1)); } \
    || { echo "FAIL  expected 'missing: openssl' and no apt-get call"; FAIL=$((FAIL+1)); }

run_case "missing file on the Hub" fatal 30 MODEL_FILE=nope.gguf
run_case "llama-server crashes during load" fatal 30 FAKE_LLAMA_MODE=crash
run_case "vision not loaded (bad mmproj)" fatal 30 FAKE_LLAMA_MODE=novision
run_case "text request fails" fatal 30 FAKE_LLAMA_MODE=textfail
run_case "boot deadline" fatal 30 FAKE_LLAMA_LOAD_SECS=30 BOOT_DEADLINE=6
check "deadline failure names the step it was stuck at" grep -q "not ready within 6s (still at: loading the model)" "$LAST_LOG"
check "deadline ORCH_FATAL reaches the container console" grep -q "ORCH_FATAL: not ready within 6s" "$LAST_CONSOLE"
run_case "slow load prints progress" ready 40 FAKE_LLAMA_LOAD_SECS=5 LOAD_HEARTBEAT_S=1
check "loading heartbeat reaches the console" grep -q "still loading after" "$LAST_CONSOLE"
run_case "hung device listing times out" fatal 30 FAKE_LLAMA_MODE=hang_devices LIST_DEVICES_TIMEOUT=2
check "says which step hung" grep -q "list-devices did not finish in 2s" "$LAST_LOG"
start_hub --slow 4
rm -rf "$WORK/models"
run_case "slow download prints progress" ready 60 DOWNLOAD_PROGRESS_S=1
check "download progress reaches the console" grep -Eq "\[fetch\] model.gguf: [0-9.]+/[0-9.]+ GiB" "$LAST_CONSOLE"
rm -rf "$WORK/models"
run_case "slow download doesn't count against the boot deadline" ready 60 BOOT_DEADLINE=8 DOWNLOAD_MIN_MBPS=0
rm -rf "$WORK/models"
start_hub --slow 30
run_case "host too slow for the weights fails early" fatal 30 DOWNLOAD_MIN_MBPS=10 DOWNLOAD_PROBE_S=2
check "says the download is too slow" grep -q "ORCH_FATAL: weights download too slow on this host" "$LAST_CONSOLE"
check "gives the measured speed" grep -q "is below DOWNLOAD_MIN_MBPS=10" "$LAST_LOG"
rm -rf "$WORK/models"
start_hub --stall-once
run_case "stalled download is restarted" ready 60 DOWNLOAD_STALL_S=2
check "says it restarted a stall" grep -q "no data for 2s, restarting" "$LAST_LOG"
rm -rf "$WORK/models"
start_hub

# huggingface_hub path, with a stand-in for the venv python: it is called as
# HF_PYTHON hf_download.py REPO PATH REVISION LOCAL_DIR and, like
# huggingface_hub, writes LOCAL_DIR/.cache/huggingface/download/*.incomplete,
# then renames it to LOCAL_DIR/PATH.
cat > "$WORK/bin/fake-hf-python" <<'EOF2'
#!/usr/bin/env bash
[ -n "${FAKE_HF_FAIL:-}" ] && { echo "fake hf: failing" >&2; exit 1; }
repo="$2" path="$3" rev="$4" dir="$5"
tmp="$dir/.cache/huggingface/download/$path.x.incomplete"
mkdir -p "$(dirname "$tmp")" "$(dirname "$dir/$path")"
curl -fsS "$HF_ENDPOINT/$repo/resolve/$rev/$path" -o "$tmp" && mv -f "$tmp" "$dir/$path"
EOF2
chmod +x "$WORK/bin/fake-hf-python"
run_case "weights through huggingface_hub" ready 60 DOWNLOAD_METHOD=auto HF_PYTHON="$WORK/bin/fake-hf-python"
check "uses huggingface_hub when it is available" grep -q "model.gguf: downloading with huggingface_hub/hf_xet" "$LAST_LOG"
rm -rf "$WORK/models"
run_case "huggingface_hub failing falls back to direct download" ready 90 DOWNLOAD_METHOD=auto \
    HF_PYTHON="$WORK/bin/fake-hf-python" FAKE_HF_FAIL=1
check "says it switched" grep -q "huggingface_hub failed twice, switching to direct download" "$LAST_LOG"
rm -rf "$WORK/models"
NOPKILL="$WORK/nopkill"; path_without "$NOPKILL" pkill
HOLD=1 run_case "boot deadline without pkill" fatal 30 FAKE_LLAMA_LOAD_SECS=30 BOOT_DEADLINE=6 PATH="$NOPKILL"
grep -q "warn: pkill failed" "$LAST_LOG" \
    && { echo "PASS  a pkill that can't run is logged"; PASS=$((PASS+1)); } \
    || { echo "FAIL  pkill failure not logged"; FAIL=$((FAIL+1)); }
run_case "llama-server dies after ready" ready_then_fatal 40 FAKE_LLAMA_MODE=die_after_ready

# ── chat + embedding model behind the local router ───────────────────────────
EMB=(EMBED_REPO=test/repo EMBED_FILE=embed.gguf EMBED_SERVED_NAME=testembed)
run_case "chat + embedding servers behind the router" ready 60 "${EMB[@]}"
grep -q -- "--embedding --pooling last" "$LAST_LOG" && grep -q "embedding ok" "$LAST_LOG" \
    && grep -q "text ok" "$LAST_LOG" && grep -q "image ok" "$LAST_LOG" \
    && echo "PASS  chat, image and embedding requests all went through the router" && PASS=$((PASS+1)) \
    || { echo "FAIL  router path incomplete"; FAIL=$((FAIL+1)); sed 's/^/      /' "$LAST_LOG" | tail -15; }
run_case "embedding server returns zero vectors" fatal 30 "${EMB[@]}" FAKE_EMBED_MODE=zero
run_case "embedding server crashes during load" fatal 30 "${EMB[@]}" FAKE_EMBED_MODE=crash
grep -q "embedding llama-server exited during load" "$LAST_LOG" \
    && { echo "PASS  embedding crash named in the fatal line"; PASS=$((PASS+1)); } \
    || { echo "FAIL  embedding crash not named"; FAIL=$((FAIL+1)); }
run_case "chat server dies after ready, embedding keeps running" ready_then_fatal 40 "${EMB[@]}" FAKE_LLAMA_MODE=die_after_ready
grep -q "ORCH_FATAL: llama-server exited" "$LAST_LOG" \
    && { echo "PASS  the dead chat server is named"; PASS=$((PASS+1)); } \
    || { echo "FAIL  dead server not named"; FAIL=$((FAIL+1)); }

rm -rf "$WORK/models"
start_hub --corrupt model.gguf
run_case "checksum mismatch rejected" fatal 40
[ -e "$WORK/models/test__repo/model.gguf" ] \
    && { echo "FAIL  corrupt file left on disk"; FAIL=$((FAIL+1)); } \
    || { echo "PASS  corrupt file removed"; PASS=$((PASS+1)); }

# ── instance stops/destroys itself instead of billing ────────────────────────
VAST_LOG="$WORK/vast_calls.log"
python3 "$HERE/fakes/fake_vast_api.py" "$VAST_PORT" "$VAST_LOG" & VAST_PID=$!
for _ in $(seq 50); do curl -s -o /dev/null "http://127.0.0.1:$VAST_PORT/" && break; sleep 0.1; done
start_hub
SELF=(CONTAINER_ID=4242 CONTAINER_API_KEY=inst-key VAST_URL="http://127.0.0.1:$VAST_PORT" ORCH_FATAL_GRACE=1)
expect_call(){   # NAME PATTERN
    if grep -q -- "$2" "$VAST_LOG" 2>/dev/null; then echo "PASS  $1"; PASS=$((PASS+1))
    else echo "FAIL  $1: no call matching '$2'"; sed 's/^/      /' "$VAST_LOG" 2>/dev/null; FAIL=$((FAIL+1)); fi
}

: > "$VAST_LOG"
HOLD=4 run_case "manual rental fails" fatal 30 "${SELF[@]}" MODEL_FILE=nope.gguf
expect_call "failed manual rental stops itself" 'PUT /api/v0/instances/4242/ Bearer inst-key {"state":"stopped"}'

: > "$VAST_LOG"
HOLD=4 run_case "serverless-style failure" fatal 30 "${SELF[@]}" ORCH_FATAL_ACTION=destroy MODEL_FILE=nope.gguf
expect_call "failed worker destroys itself" 'DELETE /api/v0/instances/4242/ Bearer inst-key'

: > "$VAST_LOG"
HOLD=4 run_case "boot deadline with self-cleanup" fatal 30 "${SELF[@]}" ORCH_FATAL_ACTION=destroy \
    FAKE_LLAMA_LOAD_SECS=30 BOOT_DEADLINE=6
expect_call "deadline failure destroys the instance" 'DELETE /api/v0/instances/4242/'
[ "$(grep -c DELETE "$VAST_LOG")" -eq 1 ] && { echo "PASS  cleanup requested once"; PASS=$((PASS+1)); } \
    || { echo "FAIL  expected one DELETE, got $(grep -c DELETE "$VAST_LOG")"; FAIL=$((FAIL+1)); }

: > "$VAST_LOG"
HOLD=4 run_case "healthy manual rental with a short TTL" ready 60 "${SELF[@]}" ORCH_MANUAL_TTL=2
expect_call "manual rental stops itself at its TTL" 'PUT /api/v0/instances/4242/ Bearer inst-key {"state":"stopped"}'

: > "$VAST_LOG"
HOLD=3 run_case "healthy manual rental, default TTL" ready 60 "${SELF[@]}"
[ -s "$VAST_LOG" ] && { echo "FAIL  healthy worker called the Vast API"; FAIL=$((FAIL+1)); } \
    || { echo "PASS  healthy worker leaves itself running"; PASS=$((PASS+1)); }

: > "$VAST_LOG"
HOLD=4 run_case "fatal with ORCH_FATAL_ACTION=none" fatal 30 "${SELF[@]}" ORCH_FATAL_ACTION=none MODEL_FILE=nope.gguf
[ -s "$VAST_LOG" ] && { echo "FAIL  ORCH_FATAL_ACTION=none still called the API"; FAIL=$((FAIL+1)); } \
    || { echo "PASS  ORCH_FATAL_ACTION=none leaves the instance alone"; PASS=$((PASS+1)); }

# Serverless mode: a live PyWorker reports the error, so the worker is left to
# the autoscaler; with no PyWorker running, the instance destroys itself.
# A local stand-in for the vast-ai/pyworker repo, so nothing is cloned from GitHub.
FAKE_PYW="$WORK/fake-pyworker"
git init -q "$FAKE_PYW" && echo stub > "$FAKE_PYW/README" \
    && git -C "$FAKE_PYW" add README \
    && git -C "$FAKE_PYW" -c user.name=t -c user.email=t@t commit -q -m stub
FAKE_PYW_REF=$(git -C "$FAKE_PYW" rev-parse HEAD)
PYW=(ORCH_SKIP_PYWORKER=0 PYWORKER_RAW_BASE=http://127.0.0.1:1
     PYWORKER_REPO="$FAKE_PYW" PYWORKER_REF="$FAKE_PYW_REF")
mkdir -p "$WORK/orch"
printf 'exec sleep 300\n' > "$WORK/orch/start_server.sh"
: > "$VAST_LOG"
HOLD=4 run_case "serverless failure with a live PyWorker" fatal 30 "${SELF[@]}" "${PYW[@]}" MODEL_FILE=nope.gguf
[ -s "$VAST_LOG" ] && { echo "FAIL  destroyed a worker the PyWorker is reporting"; FAIL=$((FAIL+1)); } \
    || { echo "PASS  live PyWorker: worker left to the autoscaler"; PASS=$((PASS+1)); }
grep -q "leaving this worker to the autoscaler" "$LAST_LOG" \
    && { echo "PASS  says why it left the worker alone"; PASS=$((PASS+1)); } \
    || { echo "FAIL  no log line for leaving the worker"; FAIL=$((FAIL+1)); }
cmp -s "$REPO/worker/pyworker_worker.py" "$WORK/workspace/vast-pyworker/worker.py" \
    && [ "$(git -C "$WORK/workspace/vast-pyworker" rev-parse HEAD)" = "$FAKE_PYW_REF" ] \
    && { echo "PASS  our worker.py installed in the pyworker checkout at the pinned ref"; PASS=$((PASS+1)); } \
    || { echo "FAIL  pyworker checkout or worker.py missing"; FAIL=$((FAIL+1)); }

printf 'exit 1\n' > "$WORK/orch/start_server.sh"
: > "$VAST_LOG"
HOLD=4 run_case "serverless failure with a dead PyWorker" fatal 30 "${SELF[@]}" "${PYW[@]}" MODEL_FILE=nope.gguf
expect_call "dead PyWorker: worker destroys itself (default action)" 'DELETE /api/v0/instances/4242/ Bearer inst-key'
rm -f "$WORK/orch/start_server.sh"

echo "---- $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

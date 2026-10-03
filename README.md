# vastai-llmapi-orchestrator

An OpenAI-compatible endpoint that only costs money while it's being used.

A small **shim** runs at home and looks like any OpenAI-style API to
SillyTavern. Behind it, **Vast.ai Serverless** rents the cheapest GPU that
matches your constraints when a request arrives, and releases it when things go
quiet. Each GPU runs upstream llama.cpp (`ghcr.io/ggml-org/llama.cpp`, pinned
build) serving one GGUF model plus its vision projector.

```
SillyTavern ──► shim (home, :8787) ──► Vast autoscaler ──► worker (rented GPU)
                 │ /v1/models answered      picks cheapest      PyWorker :3000
                 │ locally                  offer matching      llama-server :18000
                 │ keepalives during        search_params       (GGUF + mmproj)
                 └ cold start
```

## What keeps you from paying for a broken container

A worker is only marked ready after it has proven it can serve. Anything else
ends in an `ORCH_FATAL` line, which the Vast PyWorker reports as an error so
the autoscaler drops that worker instead of billing for it.

| Stage | Check | On failure |
| --- | --- | --- |
| before download | `nvidia-smi` shows ≥ `min_vram_gb`; `llama-server --list-devices` sees CUDA (catches driver/image mismatch) | fatal, nothing downloaded |
| download | file exists at the **pinned commit** on the Hub; enough disk | fatal |
| verify | size + **sha256 match the Hub's LFS hash**; GGUF magic bytes | file deleted, fatal |
| load | `llama-server` stays alive and `/health` goes 200 within `load_timeout_s` | fatal |
| smoke test | `/props` reports vision on; a text request and a **real image request** both return text | fatal |
| deadline | whole boot finishes within `deadline_s` | fatal |
| serving | `llama-server` exits later | fatal |
| after fatal | no PyWorker left to report it, `ORCH_FATAL_GRACE` (600 s) later | instance destroys itself (manual rental: stops) |

Everything remote is pinned when you deploy: model revisions, this repo's
commit, the llama.cpp image build, the PyWorker commit and the Vast SDK
version. A cold worker that restarts finds its verified weights on disk and
skips the download (size + marker check, no re-hash).

After a fatal error a running PyWorker reports it; Vast marks the worker
Error, which isn't billed, and the autoscaler handles it, so the worker is
left alone. When nothing can report the error (the PyWorker never started or
died), the instance destroys itself `ORCH_FATAL_GRACE` seconds later. This is
Vast's documented in-container route, `vastai destroy instance $CONTAINER_ID`
authorised by the per-instance `CONTAINER_API_KEY`, made as the same REST
call with curl because the llama.cpp image has no `vastai` CLI. Manual
rentals stop instead, so the logs stay on disk (stopped instances still pay
for storage). Set `ORCH_FATAL_ACTION=none` to turn this off.

## Spend limits

Spend limits live in `deploy/config.toml`: `max_workers = 1` caps GPUs billed
at once, `dph_total<=` in `search_params` caps the hourly price,
`test_workers = 1` limits benchmark instances, and `min_load = 0` lets it
scale to zero running workers. Vast's own defaults are much higher
(`max_workers` 20, `cold_workers` 5, `test_workers` 3), so `check` refuses a
config that leaves any of them out, has no `dph_total<=` ceiling, or whose
worst case `(max_workers + test_workers) x ceiling` exceeds
`[limits] max_hourly_usd`.

## Duplicates and orphans

- `apply` treats any error or odd answer from the Vast API as a stop, never as
  "nothing exists". It refuses when two endpoints share the name, when the
  endpoint has more than one workergroup, or when the endpoint recorded in
  `deploy/state.json` was renamed. It records each endpoint and workergroup in
  `state.json` before creating it, so a re-run after a crash refuses instead of
  creating a second one. A lock file stops two runs racing.
- Every instance carries `ORCH_DEPLOYMENT=<endpoint name>`. `deploy.py status`
  lists them with their hourly cost and flags orphans: marked instances older
  than `--min-age` (15 min) that the autoscaler doesn't count as workers.
  `deploy.py sweep` destroys orphans after asking, and waits until Vast no
  longer lists them. It destroys nothing if the autoscaler can't list its
  workers. Instances without the marker are listed but never touched.
- `destroy` deletes the endpoint, which per Vast's API also deletes its
  workergroups and destroys their workers. It then destroys anything left
  (workers Vast reports as failed, manual rentals) and waits until Vast no
  longer lists them. It exits non-zero if any survive.
- `apply` creates the workergroup with every limit set explicitly
  (`test_workers`, `cold_workers`, `max_workers`, `min_load`): the REST API
  accepts them per workergroup with defaults of 3 / 3 / 20 / 1, but the SDK's
  `create_workergroup` doesn't pass them, so that one call goes to the API
  directly.

## Layout

```
worker/onstart.sh       template on-start: fetches the worker scripts at the pinned commit
worker/boot.sh          the boot sequence above
worker/fetch_model.py   Hub lookup, resumable download (aria2c, curl fallback), sha256 verify
worker/smoke_test.py    /props + text + image checks
shim/shim.py            OpenAI-compatible proxy (vastai SDK client)
deploy/deploy.py        preflight, then template / endpoint / workergroup
tests/                  fakes for the Hub, GPU, llama-server, autoscaler and worker
```

## Setup

You need Python 3.11+ and a Vast API key.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r shim/requirements.txt
export VAST_API_KEY=...
cp deploy/config.example.toml deploy/config.toml
```

### 1. Prove it on one manual rental first (recommended)

Rent a single instance by hand with the image from `config.toml`, the docker
options that `deploy.py apply --dry-run` prints, `-e ORCH_SKIP_PYWORKER=1`
added, and `worker/onstart.sh` as the on-start script. Then:

```bash
tail -f /workspace/orch/model.log      # wait for ORCH_READY (or ORCH_FATAL with the reason)
curl -s localhost:18000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```

Destroy it when you're happy. That run is the reference for everything after.
A manual rental stops itself after `ORCH_MANUAL_TTL` seconds (default 4 hours;
`-e ORCH_MANUAL_TTL=0` to disable), and `deploy.py sweep` finds one you forgot.

### 2. Deploy

```bash
python3 deploy/deploy.py check            # changes nothing; prints matching offers and prices
python3 deploy/deploy.py apply --dry-run  # shows the pinned template settings
python3 deploy/deploy.py apply
python3 deploy/deploy.py status           # first worker: download, load, smoke test, Vast benchmark
```

`check` must pass before `apply` changes anything. `deploy/state.json` records
what was created; keep it. Other commands: `logs`, `pause` (workers go inactive,
storage cost only), `resume`, `sweep` (destroy orphaned instances), `destroy`.

The repo must be public, and the commit you deploy must be pushed: workers
fetch their scripts from `raw.githubusercontent.com` at that commit. For gated
Hugging Face repos, set `HF_TOKEN` in your Vast account's environment
variables, not in the template.

### 3. Run the shim at home

```bash
cp shim/config.example.env shim/.env      # VAST_API_KEY, ENDPOINT_NAME, ...
set -a; . shim/.env; set +a
python3 shim/shim.py
```

On NixOS, the Dockerfile in `shim/` works with `virtualisation.oci-containers`:

```nix
virtualisation.oci-containers.containers.llmapi-shim = {
  image = "llmapi-shim:latest";          # docker build -t llmapi-shim shim/
  ports = [ "8787:8787" ];
  environmentFiles = [ /etc/llmapi-shim.env ];
};
```

`shim/llmapi-shim.service` is a plain systemd unit for other hosts.

### 4. SillyTavern

Chat Completion → Custom (OpenAI-compatible): base URL `http://<homelab>:8787/v1`,
API key = `SHIM_API_KEY` (anything if unset), model = `served_name`.

The first message after an idle period waits for a worker. Expect a few
minutes when a cold worker resumes, longer when a fresh machine has to download
~18 GB. Streaming shows nothing until the worker is up, then flows normally.
`POST /wake` starts a worker ahead of time, and `GET /status` shows worker
states.

## Tuning

- **Context:** `llama.ctx` is shared across `parallel` slots (`--kv-unified`),
  so one long chat can use all of it. The model was trained at 8K; the base
  supports far more.
- **Price vs speed:** loosen or tighten `search_params`. `inet_down` matters
  because a slow host bills you for every minute spent downloading.
- **Quant:** change `model.file`; `check` confirms it exists and that disk fits.

## Tests

```bash
tests/run_all.sh
```

These run the real `boot.sh` against a fake Hub, fake GPU, fake llama-server
and fake Vast API through every failure mode, run the real shim with the real
vastai SDK client against a fake autoscaler and worker, and run `deploy.py`'s
apply, sweep, destroy and spend checks against an in-memory fake of the Vast
API.

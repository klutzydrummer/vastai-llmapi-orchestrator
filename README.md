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
left alone (and `deploy.py watch`, below, destroys any worker Vast still
reports as booting long past the deadline). When nothing can report the error
(the PyWorker never started or died), the instance destroys itself `ORCH_FATAL_GRACE` seconds later. This is
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

## No guessing with money

Every decision that creates or destroys something paid rests on what Vast
itself reports, never on an inference:

- **Ownership comes from Vast's answers.** Serverless workers are the ids the
  autoscaler lists for the endpoint (`get_endpoint_workers`). Test rentals are
  made by `deploy.py rent-test`, which records the instance id Vast returns.
  Both go into `state.json` (in Docker, the `orch-state` volume). Anything else on the account is
  "unrecorded": `status` shows it with its cost, and only
  `sweep --destroy ID` removes it.
- **Every write is read back.** After creating or updating the endpoint or
  workergroup, and after `pause`/`resume`, `deploy.py` re-reads it and fails
  if Vast reports a different `max_workers`, `cold_workers`, `min_load`,
  state or template than it asked for. A field Vast doesn't report is named
  as unconfirmed, not assumed. `pause`/`resume` send every configured limit
  with the new state, so nothing depends on how Vast treats omitted fields.
- **Any API error or odd answer stops the run**, never reads as "nothing
  exists". `apply` refuses duplicate endpoints, more than one workergroup, or
  an endpoint renamed since `state.json` recorded it; it records each creation
  as pending first, so a re-run after a crash refuses instead of creating a
  second one; a lock file stops two runs racing.
- **Orphans** are instances `state.json` recorded as ours that Vast still
  lists but that are no longer workers (for `--min-age`, 15 min) or are test
  rentals past `limits.manual_ttl_s`. `sweep` destroys them after asking and
  waits until Vast no longer lists them. If the autoscaler can't list its
  workers, it destroys nothing.
- **`destroy`** deletes the endpoint, which per Vast's API also deletes its
  workergroups and destroys their workers (it reports any it couldn't). It
  then destroys every recorded instance Vast still lists and waits until
  they're gone, exiting non-zero if any survive.
- **`deploy.py watch`** runs at home next to the shim (see
  `deploy/orch-watch.service`). Every minute it destroys workers Vast still
  reports as booting (`CREATING`/`LOADING`/`STARTING`…, all billed) more than
  `boot.deadline_s + limits.stuck_grace_s` after it first saw them, destroys
  recorded orphans, and pauses the endpoint if the account's running $/hr, as
  Vast reports it, exceeds `limits.max_hourly_usd`. Worker statuses it doesn't
  recognise and unrecorded instances are reported, never acted on.
- `apply` creates the workergroup with every limit set explicitly
  (`test_workers`, `cold_workers`, `max_workers`, `min_load`): the REST API
  accepts them per workergroup with defaults of 3 / 3 / 20 / 1, but the SDK's
  `create_workergroup` doesn't pass them, so that one call goes to the API
  directly.

Not documented by Vast, so not relied on: what `endpoint_state=stopped` does
beyond being the CLI's documented pause value, and whether every field is
echoed back by `show_workergroups` (unconfirmed ones are printed).

## Layout

```
worker/onstart.sh       template on-start: fetches the worker scripts at the pinned commit
worker/boot.sh          the boot sequence above
worker/fetch_model.py   Hub lookup, resumable download (aria2c, curl fallback), sha256 verify
worker/smoke_test.py    /props + text + image checks
shim/shim.py            OpenAI-compatible proxy (vastai SDK client)
deploy/deploy.py        preflight, template / endpoint / workergroup, rent-test, sweep, watch, destroy
deploy/orch-watch.service  systemd unit for `deploy.py watch`
Dockerfile, compose.yaml   one image: shim, watchdog and deploy CLI
tests/                  fakes for the Hub, GPU, llama-server, autoscaler and worker
```

## Setup

You need a Vast API key. The simplest way to run everything is Docker: one
image holds the shim, the watchdog and the deploy commands.

```bash
cp deploy/config.example.toml deploy/config.toml   # do this before compose, or Docker
cp shim/config.example.env shim/.env               # mounts an empty directory instead
# fill in both: VAST_API_KEY, SHIM_API_KEY, endpoint/model names, [limits]
docker compose build
alias orch='docker compose run --rm deploy'        # orch check, orch apply, ...
```

`deploy.py` keeps its record of what it created (`state.json`) in the
`orch-state` volume, not in the checkout. Don't delete that volume while
anything is deployed. To back it up:

```bash
docker compose run --rm --entrypoint cat deploy /data/state.json > state-backup.json
```

Without Docker (Python 3.11+):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r shim/requirements.txt
cp deploy/config.example.toml deploy/config.toml
set -a; . shim/.env; set +a
alias orch='python3 deploy/deploy.py'
```

### 1. Prove it on one test rental first (recommended)

```bash
orch rent-test        # cheapest matching offer, no serverless; records the id
```

Then, on the instance (`vastai ssh-url <id>`):

```bash
tail -f /workspace/orch/model.log      # wait for ORCH_READY (or ORCH_FATAL with the reason)
curl -s localhost:18000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```

That run is the reference for everything after. Remove it with
`deploy.py destroy` or `sweep --destroy <id>`. It stops itself after
`limits.manual_ttl_s` (4 hours), after which `sweep` and `watch` destroy it.

### 2. Deploy

```bash
orch check            # changes nothing; prints matching offers and prices
orch apply --dry-run  # shows the pinned template settings
orch apply
orch status           # first worker: download, load, smoke test, Vast benchmark
```

`check` must pass before `apply` changes anything. The state file records
what was created; keep it. Other commands: `logs`, `pause` (workers go inactive,
storage cost only), `resume`, `sweep` (destroy orphaned instances), `destroy`,
`watch` (the watchdog; run it as a service next to the shim).

The repo must be public, and the commit you deploy must be pushed: workers
fetch their scripts from `raw.githubusercontent.com` at that commit. For gated
Hugging Face repos, set `HF_TOKEN` in your Vast account's environment
variables, not in the template.

### 3. Run the shim and watchdog at home

```bash
docker compose up -d                       # shim on :8787 + watchdog, restarted on boot
docker compose logs -f watch               # "[watch]" lines show every action it takes
```

Both containers stay up with `restart: unless-stopped`. Stopping the watchdog
container mid-check is safe: its state lock is an flock the kernel drops.

On NixOS, the same image works with `virtualisation.oci-containers`. The image
is only built locally, never pulled, so build it with the same engine as
`virtualisation.oci-containers.backend` (`docker build -t
vastai-llmapi-orchestrator .`, or `podman build` for the podman default) before
the first `nixos-rebuild switch`, and again after each `git pull`:

```nix
virtualisation.oci-containers.containers = let
  orch = cmd: {
    image = "vastai-llmapi-orchestrator:latest";
    cmd = cmd;
    # Quoted strings, not Nix paths: an unquoted /etc/... path is copied into
    # the world-readable /nix/store (or fails in pure flake evaluation), and
    # this file holds VAST_API_KEY.
    environmentFiles = [ "/etc/llmapi-shim.env" ];
    volumes = [ "orch-state:/data" "/etc/orch/config.toml:/config/config.toml:ro" ];
  };
in {
  llmapi-shim = orch [ "shim" ] // { ports = [ "8787:8787" ]; };
  orch-watch  = orch [ "watch" "--interval" "60" ];
};
```

Without Docker, `shim/llmapi-shim.service` and `deploy/orch-watch.service` are
plain systemd units (both read `shim/.env`).

### 4. SillyTavern

Chat Completion → Custom (OpenAI-compatible): base URL `http://<homelab>:8787/v1`,
API key = `SHIM_API_KEY` (anything if unset), model = `served_name`.

The first message after an idle period waits for a worker. The example config
keeps no stopped workers (`cold_workers = 0`), so every start downloads ~18 GB
on a fresh machine; set `cold_workers = 1` to keep one stopped worker with the
weights on disk (storage cost only), which resumes in a few minutes. Streaming shows nothing until the worker is up, then flows normally.
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
docker compose run --rm test   # in the test image, with everything it needs
tests/run_all.sh               # or directly on the host
```

On the host the suite needs bash, python3 with `shim/requirements.txt`
installed, and curl, git, openssl, pkill (procps), setsid (util-linux) and
timeout (coreutils). `run_all.sh` checks for them and stops with a list of
what's missing, and gives each suite a time limit so it fails instead of
hanging. Nothing touches Vast.

These run the real `boot.sh` against a fake Hub, fake GPU, fake llama-server
and fake Vast API through every failure mode, run the real shim with the real
vastai SDK client against a fake autoscaler and worker, and run `deploy.py`'s
apply, read-back verification, rent-test, sweep, watch, destroy and spend
checks against an in-memory fake of the Vast API.

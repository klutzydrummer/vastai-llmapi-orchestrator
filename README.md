# vastai-llmapi-orchestrator

An OpenAI-compatible endpoint that only costs money while it's being used.

A small **shim** runs at home and looks like any OpenAI-style API to
SillyTavern. Behind it, **Vast.ai Serverless** rents the cheapest GPU that
matches your constraints when a request arrives, and releases it when things go
quiet. Each GPU runs upstream llama.cpp (`ghcr.io/ggml-org/llama.cpp`, pinned
build) serving one chat GGUF model plus its vision projector and, optionally,
an embedding model for `/v1/embeddings`.

```
SillyTavern ──► shim (home, :8787) ──► Vast autoscaler ──► worker (rented GPU)
                 │ /v1/models answered      picks cheapest      PyWorker :3000
                 │ locally                  offer matching      router :18000
                 │ keepalives during        search_params       ├ chat llama-server :18010
                 └ cold start                                   │   (GGUF + mmproj)
                                                                └ embedding llama-server :18011
```

Without an `[embedding]` section there is no router: the chat llama-server
listens on :18000 itself.

## What keeps you from paying for a broken container

A worker is only marked ready after it has proven it can serve. Anything else
ends in an `ORCH_FATAL` line, which the Vast PyWorker reports as an error so
the autoscaler drops that worker instead of billing for it.

| Stage | Check | On failure |
| --- | --- | --- |
| before download | `nvidia-smi` shows ≥ `min_vram_gb`; `llama-server --list-devices` sees CUDA (catches driver/image mismatch) | fatal, nothing downloaded |
| before download | the models fit in the GPU's **free** memory, from their GGUF headers (see **GPU memory**) | fatal, nothing downloaded, says by how much |
| download | file exists at the **pinned commit** on the Hub; enough disk; after 90 s, the projected finish at the last minute's speed is within `download_max_s` | fatal (a download with no data for 2 minutes is restarted first) |
| verify | size + **sha256 match the Hub's LFS hash**; GGUF magic bytes | file deleted, fatal |
| load | the embedding server, then the chat server (sized from the memory left), stay alive and `/health` goes 200 within `load_timeout_s` | fatal, with llama-server's own error line |
| smoke test | `/props` reports vision on; a text request and a **real image request** both return a clean answer (no chat-template tokens such as `<\|thought\|>`, no leading role name such as `user\n`; the raw reply and the prompt the template built are logged when one is refused); with an embedding model, `/v1/embeddings` returns finite, non-zero vectors | fatal |
| deadline | the boot, not counting the download, finishes within `deadline_s` | fatal, naming the step it was stuck at |
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

That worst case covers the running price only. Vast bills three things: the
GPU and the disk per second while running (together `dph_total`; the disk
also while stopped) and bandwidth per byte, in any state. Bandwidth is not in
`dph_total`, and every cold start downloads the weights again (18.7 GB for
WaifuGemma4): about $0.46 on a host charging $25/TB, more than two hours of a
cheap 3090. So the example configs also cap the download price with
`inet_down_cost<=0.01` ($/GB, at most $0.19 per WaifuGemma4 cold start), and
`check` warns when `search_params` has no `inet_down_cost<=` ceiling. `check`
names the most a cold start's download can cost under that ceiling, and its
offer table shows, per offer, the download cost (`dl`) and one cold start's
cost (`cold`: `dph_total` for the boot, estimated from a 12-minute measured
boot scaled to the weights, plus the download). `rent-test` prints the
download cost of the offer it picks. Both still rank offers by `dph_total`,
as Vast's autoscaler does with the same `search_params`; the
`inet_down_cost` ceiling is what keeps an expensive download out of either.

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
  `boot.deadline_s + boot.download_max_s + limits.stuck_grace_s` after it first saw them,
  destroys `rent-test` instances Vast still reports as booting (no container
  yet) `limits.stuck_grace_s` after they were rented, destroys
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
worker/smoke_test.py    /props + text + image (+ embedding) checks
worker/vram.py          GGUF header reader and GPU memory planner (also used by `deploy.py check`)
worker/router.py        sends /v1/embeddings to the embedding llama-server, the rest to chat; /orch/info
worker/pyworker_worker.py  Vast's llama PyWorker routes plus /v1/embeddings and /orch/info
shim/shim.py            OpenAI-compatible proxy (vastai SDK client)
shim/status.html        status page: API URL, model, slots, context per slot, copy buttons
deploy/deploy.py        preflight, template / endpoint / workergroup, rent-test, sweep, watch, destroy
deploy/orch-watch.service  systemd unit for `deploy.py watch`
Dockerfile, compose.yaml   one image: shim, watchdog and deploy CLI
tests/                  fakes for the Hub, GPU, llama-server, autoscaler and worker
```

## Setup

You need a Vast API key. The simplest way to run everything is Docker: one
image holds the shim, the watchdog and the deploy commands.

Two example configs, pick one to copy to `deploy/config.toml`. Both use the
Gemma 4 26B-A4B vision projector (`unsloth/gemma-4-26B-A4B-it-GGUF`
`mmproj-BF16.gguf`) and serve embeddings:

| File | Models | GPU | Price ceiling / budget |
| --- | --- | --- | --- |
| `config.example.toml` | Pantheon-Reasoning 26B-A4B Q4_K_M + vision, Qwen3-Embedding-0.6B | 24 GB+ | $0.40/hr per GPU, $0.80/hr |
| `config.waifugemma4.example.toml` | WaifuGemma4 26B-A4B Q4_K_M + vision, Qwen3-Embedding-0.6B | 24 GB | $0.40/hr per GPU, $0.80/hr |

Keep `ENDPOINT_NAME`, `SERVED_MODEL_NAME` and `EMBED_MODEL_NAME` in
`shim/.env` matching the one you chose (see `shim/config.example.env`).

```bash
cp deploy/config.example.toml deploy/config.toml   # do this before compose, or Docker
cp shim/config.example.env shim/.env               # mounts an empty directory instead
# fill in both: VAST_API_KEY, SHIM_API_KEY, endpoint/model names, [limits]
docker compose pull                                # the published image, latest main
alias orch='docker compose run --rm deploy'        # orch check, orch apply, ...
```

Every push to `main` publishes the image to
`ghcr.io/klutzydrummer/vastai-llmapi-orchestrator`, tagged `latest` and
`sha-<commit>`, after the offline tests pass in the same build
(`.github/workflows/image.yml`). To update, `docker compose pull && docker
compose up -d`. To run your own checkout instead, `docker compose build` (or
`up -d --build`); it builds under the same name, so the next `pull` replaces
it with `main` again. To stay on one version, set `image:` in `compose.yaml`
to a `sha-` tag.

Commands that rent or delete something (`rent-test`, `destroy`, `sweep
--destroy`) ask first. `docker compose run` attaches a terminal, so you can
answer; where nothing can answer (scripts, cron, `-T`), they stop with "no
terminal to confirm on" and change nothing. Pass `--yes` to go ahead without
asking.

If `docker compose build` warns "Docker Compose is configured to build using
Bake, but buildx isn't installed", it's harmless: compose falls back to the
classic builder. Install the buildx plugin or set `COMPOSE_BAKE=false` to
silence it.

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
orch rent-test        # cheapest matching offer, no serverless; records the id, follows the boot
orch logs <id>        # follows an instance's boot until ORCH_READY (exit 0) or a failure (exit 1)
```

Following a boot (`rent-test`, or `logs <id>`) does two things at once:

- **The container log** is fetched in the background from the start
  (`vastai logs <id>` shows the same lines): each boot step, download progress
  every 30 s, a line a minute while the model loads, and the final marker.
- **Vast's own report** on the instance (`actual_status` and `status_msg`) is
  read every 20 s and decides whether the boot is still valid. It is printed
  when it changes, and repeated about once a minute while nothing new arrives.
  The boot has failed when Vast lists the instance as gone, when Vast reports
  an error (for example "Secrets fetch failed") for a minute while the
  container hasn't started, or when the instance is still booting at Vast
  (image pull, container start) `limits.stuck_grace_s` (10 min) after it was
  rented. Our own boot (downloads, model load) runs after Vast reports the
  container running and has its own limits.

When a boot fails, the log so far is fetched one last time and saved with
Vast's last report under `boots/<id>.log` next to `state.json`. If it failed
because of its host (stuck or erroring at Vast, or "weights download too slow
on this host"), `rent-test` then destroys it and rents the next cheapest other
host, up to `limits.rent_attempts` (3) rentals in all. Any other failure, such
as a smoke test, is reported and the instance left for a look.

`rent-test` searches offers again right before each rental, since the
preflight can take minutes. If Vast refuses the rental, its answer is printed
and the account is checked for an instance that appeared anyway (if one did,
it stops and names it). If the offer is no longer listed, someone else took
it: the next offer is tried, counting toward `rent_attempts`. If Vast refuses
an offer that is still listed, it stops; nothing was rented.

`rent-test --no-follow` only rents. `logs <id>` never destroys anything; it
exits 1 and says how. If the boot runs out of time, the ORCH_FATAL line names
the step it was stuck at. `logs <id> --once` prints what is there now without
waiting.

For the full log, including llama-server's own output, on the instance
(`vastai ssh-url <id>`, with an SSH key added to your Vast account; Vast puts
the account's keys on every rental):

```bash
tail -f /workspace/orch/model.log      # wait for ORCH_READY (or ORCH_FATAL with the reason)
curl -s localhost:18000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```

To start `llama-server` by hand there, run it from `/app` or set
`LD_LIBRARY_PATH=/app`; otherwise it stops with
`error while loading shared libraries: libllama-server-impl.so`.

A failed test rental stops itself 10 minutes (`ORCH_FATAL_GRACE`) after the
failure. For a longer look, rent with `rent-test --hold-on-fatal MINUTES`:
that rental alone waits that long instead, at its full $/hr (printed before
renting, with the total). It can't exceed `limits.manual_ttl_s`, which still
stops the rental, and `sweep` and `watch` still destroy it after that. A host
failure that `rent-test` replaces (stuck at Vast, Vast error, download too
slow) is still destroyed at once.

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
what was created; keep it. Other commands: `logs` (the autoscaler's log;
`logs <id>` follows one instance's boot), `pause` (workers go inactive,
storage cost only), `resume`, `sweep` (destroy orphaned instances), `destroy`,
`watch` (the watchdog; run it as a service next to the shim).

The repo must be public, and the commit you deploy must be pushed: workers
fetch their scripts from `raw.githubusercontent.com` at that commit. Set
`HF_TOKEN` (a read token) in your Vast account's environment variables, not
in the template: gated repos need it, and Hugging Face gives unauthenticated
downloads lower rate limits. The boot log says whether it is set.

### 3. Run the shim and watchdog at home

```bash
docker compose up -d                       # shim on :8787 + watchdog, restarted on boot
docker compose logs -f watch               # "[watch]" lines show every action it takes
```

Both containers stay up with `restart: unless-stopped`. Stopping the watchdog
container mid-check is safe: its state lock is an flock the kernel drops.

On NixOS, the same image works with `virtualisation.oci-containers`, which
pulls it on the first `nixos-rebuild switch`. It does not pull again on its
own, so to update, pull `latest` with the backend's engine (`podman pull
ghcr.io/klutzydrummer/vastai-llmapi-orchestrator:latest`, or `docker pull`)
and restart the two services, or pin `image` to a `sha-` tag and change it:

```nix
virtualisation.oci-containers.containers = let
  orch = cmd: {
    image = "ghcr.io/klutzydrummer/vastai-llmapi-orchestrator:latest";
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
For vector storage / RAG, point the embedding source at the same base URL
(OpenAI-compatible); the model is `embedding.served_name` (`EMBED_MODEL_NAME`
in `shim/.env`). An embedding request wakes a worker like any other.

The first message after an idle period waits for a worker. The example
configs keep no stopped workers (`cold_hours` is off, see **Schedules**),
so a start downloads the weights (about 29 GB for the default config, 17 GB
for WaifuGemma4) on a fresh machine. See **Downloads and cold starts** below for the trade-off. Streaming
shows nothing until the worker is up, then flows normally.
`POST /wake` starts a worker ahead of time, and `GET /status` shows worker
states.

**Status page.** Open `http://<homelab>:8787/` for everything SillyTavern
asks for, each with a Copy button: the API URL, model ids, whether a key is
needed, context per slot (SillyTavern's Context Size), slots, total and
trained context, image support, the embedding model's input limit and
dimensions, and the worker's GPU and memory plan. The page itself never
sends anything to a worker, so opening it never starts a GPU and leaving it
open never keeps one billing (see **When a worker is released**). The values
come from the worker right after a chat, embedding or wake request succeeds
(at most every 30 s, `INFO_MIN_INTERVAL`), and the page shows how old they
are (kept in `shim/.info-cache.json` across restarts, `INFO_CACHE` to move
it). With `SHIM_API_KEY` set the page asks for the key once and keeps it in
the browser. If a reverse proxy or tunnel only passes `/v1/*` to the shim,
open `/v1/info` instead: the same page, whose calls then go to
`/v1/info.json` and `/v1/wake` (`/v1/status` works too).

**When a worker is released.** Vast's autoscaler releases a running worker
once the endpoint has seen no activity for `endpoint.inactivity_timeout`
(1800 s, 30 minutes, in the examples) and `min_load` allows zero workers. The rule this
repo holds every path to:

- *Keeps a worker:* any client work, while it runs and for
  `inactivity_timeout` after it ends. Client work is a chat or completion
  request, an embedding request, or `POST /wake`. A streamed reply counts
  until its last chunk; a request the client abandons ends when it is
  cancelled.
- *Never keeps a worker:* anything else: `/v1/models`, `/status`, the status
  page, `/health`, the watchdog's and `deploy.py`'s API polls, and the
  PyWorker's own health checks.
- *Deliberate exception:* `warm_hours`, which holds a worker with no traffic
  by raising `min_load` (see **Schedules** below), not by sending requests.

How the code matches it: every client request goes through Vast's router and
the worker's PyWorker, the only activity the autoscaler can see. Embeddings
are a PyWorker route like chat, not a side channel. Each request is reported
with a load above zero, from one rule that the shim (`request_cost`, the
routing cost) and the worker (`request_workload`, the load it reports) share,
and a test keeps the two equal: embeddings count about 4 input characters
per token, generations their `max_tokens` / `max_completion_tokens` /
`n_predict`, else 512. The stock llama worker counted a chat request without
`max_tokens` as zero load; this one doesn't. The PyWorker keeps a request in
its working set until the response is fully sent, so a long stream reports
load the whole time. The shim's only other worker request, `/orch/info` for
the status page, is sent only right after client work succeeded and only
while a worker is listed as running, so it never extends idle time by more
than that one request. What Vast's autoscaler counts as "activity" internally
isn't documented; the code relies only on requests routed to workers.

**Schedules.** Two optional endpoint settings change what is kept by time of
day and day of week, in `endpoint.schedule_tz` (an IANA zone such as
`"America/Chicago"`; required when either is set, since the container's clock
is UTC). Both take comma-separated ranges, each optionally prefixed with days:
`"Mon-Fri 14:00-24:00, Sat-Sun 08:00-24:00"`. Days are `Mon`..`Sun` or a
range (`Fri-Mon` wraps); a range without days applies every day, and one that
crosses midnight (`"Fri 22:00-02:00"`) belongs to the day it starts. `watch`
applies them about once a minute, so it has to be running (`docker compose up
-d` starts it); `apply`, `pause` and `resume` send the values for the current
time, so none of them undoes a schedule. `deploy.py status` prints both.

- **`cold_hours`** (off in the examples; e.g. `"Mon-Fri 14:00-24:00, Sat-Sun
  08:00-24:00"`, see *What a live test showed* below). Outside them nothing is kept: `cold_workers` is 0, so a
  worker a request starts is destroyed once idle, and `watch` destroys any
  stopped worker it finds. Inside them nothing is started ahead of a request
  either; once a request has started a worker, `watch` raises the endpoint's
  `cold_workers` to `cold_hours_workers` (1), so when that worker idles out
  Vast stops it with its weights on disk instead of destroying it, and the
  next request resumes it without downloading. A stopped worker pays only
  Vast's storage rate. At the end of the window `cold_workers` goes back to 0:
  a worker still serving requests keeps running and idles out as usual, and a
  stopped one is destroyed. `cold_workers` is raised only once a worker
  exists because Vast describes `cold_workers` as a floor on total workers,
  which could otherwise make it rent one with no request. `endpoint.cold_workers`
  must be 0 when `cold_hours` is set.
- **`warm_hours`** (off in the examples) keeps one worker *running*. At the
  start of the window `watch` raises the endpoint's `min_load` to
  `warm_min_load` (1), so the autoscaler holds a worker up with no traffic
  instead of releasing it after `inactivity_timeout`; at the end it puts
  `min_load` back, and the worker idles out as usual. No requests are sent, so
  llama.cpp's prompt cache is left alone. Start the window about 25 minutes
  before you need it, since the first worker pays the cold start. While warm,
  a worker bills its full rate (a 3090 at about $0.17/hr is about $1.40 for 8
  hours).

Each change is read back, and a value Vast reports differently is put back.

*What a live test showed* (endpoint 39473, 2026-10-05, no requests sent):
Vast does go by the endpoint's `cold_workers`, does list a stopped worker as
the endpoint's, and did stop an idle worker with its weights instead of
destroying it. But two things make the saving unreliable, which is why the
examples leave `cold_hours` off:

- Right after `cold_workers` went to 1 (with one idle worker running, and a
  template change rolling out at the same time), Vast rented and dropped six
  extra workers over 15 minutes (A100, RTX 5090, RTX 5000 Ada), each for a
  minute or two, up to two billing at once, despite `max_workers = 1`. It
  stopped once the running worker was recycled and stopped. Vast documents
  `cold_workers` as a floor on total workers that the autoscaler fills, so
  it can rent to meet it; which of the two changes set this off isn't known.
- A stopped worker holds no GPU. 14 minutes after it stopped, its host rented
  the GPU to someone else and Vast reported it `unavail`: it can't resume
  until that rental ends, so the next request would have downloaded anyway.
  `watch` now sets `cold_workers` back to 0 as soon as the kept worker is
  `unavail` (so Vast has no floor to refill) and destroys it.

`deploy.py status` shows a stopped instance at its storage rate
(`storage_cost` x `disk_space`, about $0.01/hr in that test), not its running
`dph_total`.

## GPU memory

Every worker works out what fits before it downloads anything. `worker/vram.py`
reads the GGUF headers of the chat model, projector and embedding model from
the Hub (a few MB each, by HTTP range request) and adds up, per server: the
weights, the context cache (per layer, so Gemma's sliding-window layers and
per-layer KV heads count correctly; older headers use the plain formula), the
compute buffer, the projector and its vision encoder, a CUDA context per
process and a margin (256 MiB + 3% of the card). The compute, projector and
CUDA figures are fitted to a 3090 rental: the embedding server's compute buffer
grows by about 0.6 MiB per token of context (its batch size follows its
context), so `embedding.ctx = 8192` costs about 2.8 GiB more than the default
4096 and doesn't fit next to the chat model on 24 GB. If that's more than the GPU's free memory it
gives up, in order: embedding context (halved down to 2048), embedding cache
precision (f16 to q8_0), then chat context (down to `llama.ctx_min`). If even
that doesn't fit, the boot fails before downloading and says by how much.
`llama.ctx` is a ceiling and is never raised; it is also capped at the trained
context times `parallel`.

The servers start one at a time: the embedding server first, then, once its
`/health` answers, the chat server is sized again from the memory actually
left (`[boot] VRAM after embedding: X MiB used`). Each server's output goes to
`vastai logs` (and so `orch logs` and `deploy/boots/<id>.log`) prefixed
`[chat]` / `[embed]`, all of it while loading and warnings and errors after
that, and to `/workspace/orch/chat.log` and `embed.log`. After loading, the
boot logs llama.cpp's own buffer sizes and what each server added to
`nvidia-smi`'s used memory next to the estimate.

With a projector the batch size (`-ub`) is raised to `llama.image_tokens`
(1120 for Gemma 4), since a smaller batch cuts images down. The embedding
model runs on the GPU; `embedding.gpu = false` moves it to the CPU (`-ngl 0`,
nothing reserved for it).

`deploy.py check` runs the same plan for the smallest GPU your
`search_params` (`gpu_ram>=`) and `min_vram_gb` allow, prints the breakdown,
and fails if it doesn't fit, listing the options: a larger GPU, a lower
`llama.ctx_min`, the projector in system RAM (`--no-mmproj-offload` in
`llama.extra_args`) or the embedding model on the CPU. It doesn't pick one
for you.

## Tuning

- **Context:** `llama.ctx` is shared across `parallel` slots (`--kv-unified`),
  so one long chat can use all of it. A larger context costs VRAM for the KV
  cache; `check` shows whether it fits, and the worker lowers it toward
  `llama.ctx_min` when a card has less free memory than planned.
- **Price vs speed:** loosen or tighten `search_params`. `inet_down` matters
  because a slow host bills you for every minute spent downloading, and
  `inet_down_cost` because the bytes themselves are billed too (see
  **Spend limits**).
- **Quant:** change `model.file`; `check` confirms it exists and that disk fits.

### Downloads and cold starts

Workers download weights with `huggingface_hub` + `hf_xet` (pinned versions,
installed into a venv on the worker), which is Hugging Face's recommended way
to fetch Xet-stored files. If that can't be set up, they fall back to aria2c,
then curl. Every file is still checked against the pinned revision's size and
sha256. The log shows the method, progress every 30 s and the final speed.
hf_xet holds data in memory and writes the file late and out of order, so
progress and the speed checks below count the bytes huggingface_hub reports
as received, not the file's size on disk.

A host's advertised `inet_down` is not what you get from Hugging Face: a test
on a 1336 Mbps host managed 5–11 MB/s over plain HTTP. So the download has its
own limits instead of eating the boot deadline:

- `download_max_s` is the time budget for all files together: 5400 s in the
  default config (29 GB), 3600 s in the WaifuGemma4 one (17 GB) and if unset.
  After 90 s, the worker projects the finish: the bytes still to fetch (this
  file and the ones after it) at the speed of the last minute. Only if that
  misses the budget does the boot fail, right away, with "weights download too
  slow on this host", instead of billing until the budget runs out. The
  autoscaler (or `rent-test`, by itself) can pick another host. The log line
  at the start says what average speed the budget needs (about 4.7 MB/s for
  17 GB in an hour).
- The last minute's speed, not the average from the start, because downloads
  often ramp up (8.5 → 19.4 → 20.6 MB/s in one rental) and Hub speeds vary
  from host to host (8–20 MB/s seen so far).
- `download_min_mbps` adds an optional fixed floor on that speed. It is unset
  in the examples: fixed floors near typical Hub speeds killed downloads that
  would have finished in time.
- A download that receives nothing for 2 minutes is restarted.

Every cold start downloads everything again. To skip that, set
`endpoint.cold_hours` (see **Schedules**, including why the examples don't) or, around the clock,
`endpoint.cold_workers = 1`: one stopped worker keeps its verified weights on
disk and resumes without downloading. A stopped worker still pays Vast's
storage rate for its disk (`template.disk_gb`), and the worker it keeps is on
one host, which may be unavailable when you need it. With `cold_workers = 0`
nothing is billed while idle, and each start pays for the download time
and the downloaded bytes (`inet_down_cost`) instead: at the 8–20 MB/s seen so far, 17 GB takes 15–35 minutes and the
default config's 29 GB 25–60 minutes.

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
checks against an in-memory fake of the Vast API. `tests/test_fetch.py` checks
the download budget against the speeds real rentals saw.

#!/usr/bin/env python3
"""Create / update / inspect the Vast Serverless setup from config.toml.

  deploy.py check     preflight only: every remote reference resolves, files
                      exist, disk fits, the image tag exists, offers match
  deploy.py apply     check, then create/update template, endpoint, workergroup
  deploy.py status    endpoint, workergroups and workers
  deploy.py logs      recent endpoint logs from the autoscaler
  deploy.py logs ID   follow one instance's boot through `vastai logs` until
                      it prints ORCH_READY (exit 0) or ORCH_FATAL (exit 1);
                      --once prints what is there now and returns
  deploy.py pause     stop the endpoint (workers go inactive: storage cost only)
  deploy.py resume    reactivate it
  deploy.py sweep     destroy orphans: instances recorded as ours (former
                      workers, test rentals past their TTL) that are no longer
                      workers; --destroy ID also removes named ones (asks first)
  deploy.py destroy   delete the endpoint (Vast destroys its workers), destroy
                      every recorded instance and wait until gone (asks first)
  deploy.py rent-test rent one instance outside serverless to prove a config,
                      and record its id
  deploy.py watch     watchdog loop: destroy workers Vast reports stuck booting
                      and recorded orphans, pause the endpoint on overspend

Only facts Vast reports drive these: the autoscaler's worker list and the
instance ids Vast returned, recorded in deploy/state.json. Every create or
update is read back and compared with what was asked.

Options: --config PATH (default deploy/config.toml), --yes (no prompts),
--dry-run (apply/sweep/rent-test: show what would happen, change nothing).
Needs VAST_API_KEY in the environment.
"""

import argparse
import fcntl
import json
import os
import re
import sys
import time
import tomllib
import urllib.parse

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# In the container this points at a mounted volume (see compose.yaml).
STATE_PATH = os.environ.get("ORCH_STATE_PATH") or os.path.join(HERE, "state.json")
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
GH_API = "https://api.github.com"
GH_RAW = "https://raw.githubusercontent.com"
UA = {"User-Agent": "vastai-llmapi-orchestrator-deploy"}


class CheckFailed(Exception):
    pass


class ApiError(CheckFailed):
    """A Vast API call failed or answered with something we can't trust.

    Every decision to create or destroy something paid depends on these
    answers, so an error is never read as "nothing there"."""


def say(msg=""):
    print(msg, flush=True)


# ── config ───────────────────────────────────────────────────────────────────
def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for section in ("model", "llama", "endpoint", "workergroup", "template", "boot"):
        if section not in cfg:
            raise SystemExit(f"{path}: missing [{section}]")
    return cfg


def _no_space(name, value):
    if any(c.isspace() for c in str(value)) or '"' in str(value) or "'" in str(value):
        raise CheckFailed(f"{name}={value!r}: values passed to the container can't contain spaces or quotes")
    return str(value)


def download_max_s(cfg):
    """The weight download's own time limit; the boot deadline excludes it."""
    return int(cfg["boot"].get("download_max_s", 3600))


def boot_limit_s(cfg):
    """Longest a healthy worker can take from boot.sh start to ready: the boot
    deadline plus the download's own limit."""
    return int(cfg["boot"]["deadline_s"]) + download_max_s(cfg)


def docker_options(cfg, pins):
    """The template's 'Docker options' string: port + env, all pinned."""
    m, l, b = cfg["model"], cfg["llama"], cfg["boot"]
    env = {
        "ORCH_RAW_BASE": f"{GH_RAW}/{b['orch_repo']}",
        "ORCH_REF": pins["orch_ref"],
        "SERVED_MODEL_NAME": m["served_name"],
        "MODEL_REPO": m["repo"],
        "MODEL_FILE": m["file"],
        "MODEL_REVISION": pins["model_revision"],
        "MMPROJ_REPO": m.get("mmproj_repo", ""),
        "MMPROJ_FILE": m.get("mmproj_file", ""),
        "MMPROJ_REVISION": pins.get("mmproj_revision", ""),
        "LLAMA_CTX": l["ctx"],
        "LLAMA_PARALLEL": l["parallel"],
        "LLAMA_CACHE_TYPE": l["cache_type"],
        "LLAMA_REASONING_BUDGET": l.get("reasoning_budget", 0),
        "LLAMA_EXTRA_ARGS": str(l.get("extra_args", "")).strip().replace(" ", ";"),
        "MIN_VRAM_GB": l["min_vram_gb"],
        "BOOT_DEADLINE": b["deadline_s"],
        "DOWNLOAD_MIN_MBPS": b.get("download_min_mbps"),   # optional floor; unset = none
        "DOWNLOAD_MAX_S": download_max_s(cfg),
        "LOAD_TIMEOUT": b["load_timeout_s"],
        "PYWORKER_REF": b["pyworker_ref"],
        "VAST_SDK_VERSION": b["vast_sdk_version"],
    }
    e = cfg.get("embedding")
    if e:
        env.update({
            "EMBED_SERVED_NAME": e["served_name"],
            "EMBED_REPO": e["repo"],
            "EMBED_FILE": e["file"],
            "EMBED_REVISION": pins.get("embed_revision", ""),
            "EMBED_CTX": e.get("ctx", 8192),
            "EMBED_POOLING": e.get("pooling", "last"),
        })
    parts = ["-p 3000:3000"]
    for k, v in env.items():
        if v == "" or v is None:
            continue
        parts.append(f"-e {k}={_no_space(k, v)}")
    return " ".join(parts)


def price_ceiling(search_params):
    """The dph_total upper bound in a Vast search string, or None."""
    caps = [float(m.group(2)) for m in re.finditer(r"dph_total\s*(<=|<)\s*([0-9]*\.?[0-9]+)", search_params)]
    return min(caps) if caps else None


def check_limits(cfg):
    """Refuse configs that leave spend to Vast's defaults.

    Vast fills in max_workers=20, cold_workers=5 and test_workers=3 when they
    are omitted, and a search without a dph_total ceiling rents at any price.
    Returns the worst-case hourly GPU cost this config allows."""
    e, w, lim = cfg["endpoint"], cfg["workergroup"], cfg.get("limits")
    if not lim or "max_hourly_usd" not in lim:
        raise CheckFailed("[limits] max_hourly_usd is not set; add it (see config.example.toml)")
    for section, key in (("endpoint", "max_workers"), ("endpoint", "cold_workers"),
                         ("workergroup", "test_workers")):
        val = cfg[section].get(key)
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise CheckFailed(f"{section}.{key} must be set to a whole number (Vast's default is much higher)")
    if lim.get("on_breach", "pause") not in ("pause", "alert"):
        raise CheckFailed("limits.on_breach must be \"pause\" or \"alert\"")
    if e["max_workers"] < 1:
        raise CheckFailed("endpoint.max_workers must be at least 1")
    cap = lim.get("max_workers_cap", 2)
    if e["max_workers"] > cap:
        raise CheckFailed(f"endpoint.max_workers={e['max_workers']} is above limits.max_workers_cap={cap}")
    dph = price_ceiling(w["search_params"])
    if dph is None:
        raise CheckFailed("workergroup.search_params has no dph_total<= price ceiling")
    # Benchmark (test) workers can run alongside regular ones; count both.
    worst = (e["max_workers"] + w["test_workers"]) * dph
    if worst > float(lim["max_hourly_usd"]) + 1e-9:
        raise CheckFailed(f"worst case ({e['max_workers']} workers + {w['test_workers']} test) x "
                          f"${dph:.2f}/hr = ${worst:.2f}/hr is above limits.max_hourly_usd="
                          f"{float(lim['max_hourly_usd']):.2f}")
    return worst


def onstart_script():
    with open(os.path.join(REPO_ROOT, "worker", "onstart.sh")) as f:
        return f.read()


# ── preflight ────────────────────────────────────────────────────────────────
def hf_resolve(repo, revision):
    r = requests.get(f"{HF}/api/models/{urllib.parse.quote(repo, safe='/')}/revision/"
                     f"{urllib.parse.quote(revision, safe='')}", headers=_hf_headers(), timeout=30)
    if r.status_code != 200:
        raise CheckFailed(f"{repo}@{revision}: Hub returned HTTP {r.status_code}")
    return r.json()["sha"]


def hf_file_size(repo, sha, path):
    r = requests.post(f"{HF}/api/models/{urllib.parse.quote(repo, safe='/')}/paths-info/{sha}",
                      data={"paths": path}, headers=_hf_headers(), timeout=30)
    if r.status_code != 200:
        raise CheckFailed(f"{repo}: paths-info HTTP {r.status_code}")
    for it in r.json():
        if it.get("path") == path:
            lfs = it.get("lfs") or {}
            if not lfs.get("oid"):
                raise CheckFailed(f"{repo}:{path} has no LFS sha256; the worker can't verify it")
            return int(lfs.get("size", it.get("size")))
    raise CheckFailed(f"{path} not found in {repo}@{sha[:12]} (check the exact filename)")


def _hf_headers():
    h = dict(UA)
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return h


def gh_resolve(repo, ref):
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref):
        return ref
    import subprocess
    try:
        out = subprocess.run(["git", "ls-remote", f"https://github.com/{repo}", ref, f"refs/heads/{ref}",
                              f"refs/tags/{ref}"], capture_output=True, text=True, timeout=60).stdout
        for line in out.splitlines():
            sha, name = line.split("\t", 1)
            if name in (f"refs/heads/{ref}", f"refs/tags/{ref}", ref, f"refs/tags/{ref}^{{}}"):
                return sha
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    r = requests.get(f"{GH_API}/repos/{repo}/commits/{ref}", headers=UA, timeout=30)
    if r.status_code != 200:
        raise CheckFailed(f"github {repo}@{ref}: HTTP {r.status_code} (is the repo public and pushed?)")
    return r.json()["sha"]


def gh_raw_exists(repo, sha, path):
    r = requests.head(f"{GH_RAW}/{repo}/{sha}/{path}", headers=UA, timeout=30, allow_redirects=True)
    return r.status_code == 200


MANIFEST_TYPES = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json"])


def _ghcr_token(name):
    return requests.get(f"https://ghcr.io/token?scope=repository:{name}:pull", timeout=20).json()["token"]


def _ghcr_has(name, tag, tok):
    r = requests.head(f"https://ghcr.io/v2/{name}/manifests/{tag}", timeout=20,
                      headers={"Authorization": f"Bearer {tok}", "Accept": MANIFEST_TYPES})
    return r.status_code == 200


def image_exists(image):
    """True/False for ghcr.io images; None when the registry can't be checked."""
    if not image.startswith("ghcr.io/"):
        return None
    name, _, tag = image[len("ghcr.io/"):].rpartition(":")
    try:
        return _ghcr_has(name, tag, _ghcr_token(name))
    except Exception:
        return None


def nearest_older_image(image, tries=40):
    """For a missing ...-b<N> tag, the newest existing ...-b<M> with M < N.

    llama.cpp publishes container images for only some release builds, so a
    release number that exists on GitHub can still have no image."""
    m = re.fullmatch(r"ghcr\.io/(.+):(.*-b)(\d+)", image)
    if not m:
        return None
    name, prefix, n = m.group(1), m.group(2), int(m.group(3))
    try:
        tok = _ghcr_token(name)
        for k in range(n - 1, max(n - 1 - tries, 0), -1):
            if _ghcr_has(name, f"{prefix}{k}", tok):
                return f"ghcr.io/{name}:{prefix}{k}"
    except Exception:
        pass
    return None


def vast():
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        raise SystemExit("VAST_API_KEY is not set")
    from vastai import VastAI
    return VastAI(api_key=key)


def check(cfg, v=None):
    """Run every preflight check; return the pins to deploy with."""
    m, l, b = cfg["model"], cfg["llama"], cfg["boot"]
    problems, pins = [], {}

    def step(label, fn):
        try:
            out = fn()
            say(f"  ok    {label}" + (f": {out}" if isinstance(out, str) else ""))
            return out
        except CheckFailed as e:
            say(f"  FAIL  {label}: {e}")
            problems.append(str(e))
        except requests.RequestException as e:
            say(f"  FAIL  {label}: network error {e}")
            problems.append(str(e))

    say("spend limits")
    step("max workers, test workers, price ceiling",
         lambda: f"worst case ${check_limits(cfg):.2f}/hr")

    say("model files")
    pins["model_revision"] = step(f"{m['repo']}@{m['revision']}", lambda: hf_resolve(m["repo"], m["revision"]))
    sizes = []
    if pins["model_revision"]:
        sz = step(m["file"], lambda: hf_file_size(m["repo"], pins["model_revision"], m["file"]))
        if sz:
            sizes.append(sz)
    if m.get("mmproj_file"):
        mm_repo = m.get("mmproj_repo") or m["repo"]
        pins["mmproj_revision"] = step(f"{mm_repo}@{m.get('mmproj_revision', 'main')}",
                                       lambda: hf_resolve(mm_repo, m.get("mmproj_revision", "main")))
        if pins["mmproj_revision"]:
            sz = step(m["mmproj_file"], lambda: hf_file_size(mm_repo, pins["mmproj_revision"], m["mmproj_file"]))
            if sz:
                sizes.append(sz)
    else:
        say("  warn  no mmproj configured: image input will not work")
    e = cfg.get("embedding")
    if e:
        pins["embed_revision"] = step(f"{e['repo']}@{e.get('revision', 'main')}",
                                      lambda: hf_resolve(e["repo"], e.get("revision", "main")))
        if pins["embed_revision"]:
            sz = step(e["file"], lambda: hf_file_size(e["repo"], pins["embed_revision"], e["file"]))
            if sz:
                sizes.append(sz)

    weights_gb = sum(sizes) / 1024**3
    if sizes:
        need_disk = weights_gb + 12   # image layers, pyworker venv, logs, headroom
        say(f"  info  weights {weights_gb:.1f} GiB; disk {cfg['template']['disk_gb']} GB (need ~{need_disk:.0f})")
        if cfg["template"]["disk_gb"] < need_disk:
            problems.append(f"template.disk_gb={cfg['template']['disk_gb']} is too small; use >= {need_disk:.0f}")
            say("  FAIL  disk too small")
        if weights_gb + 2.5 > l["min_vram_gb"]:
            say(f"  warn  weights (all models) {weights_gb:.1f} GiB leave little room for KV cache under "
                f"min_vram_gb={l['min_vram_gb']}")

    say("worker code and image")
    pins["orch_ref"] = step(f"github {b['orch_repo']}@{b['orch_ref']}", lambda: gh_resolve(b["orch_repo"], b["orch_ref"]))
    if pins["orch_ref"]:
        for f in ("worker/onstart.sh", "worker/boot.sh", "worker/fetch_model.py", "worker/hf_download.py",
                  "worker/smoke_test.py",
                  "worker/router.py", "worker/pyworker_worker.py"):
            def _has(f=f):
                if not gh_raw_exists(b["orch_repo"], pins["orch_ref"], f):
                    raise CheckFailed("not found at that commit (push it first)")
            step(f"  {f}", _has)
    def _pyworker():
        # worker/pyworker_worker.py imports workers/openai/core.py and is run by
        # that ref's start_server.sh.
        for f in ("start_server.sh", "workers/openai/core.py"):
            if not gh_raw_exists("vast-ai/pyworker", b["pyworker_ref"], f):
                raise CheckFailed(f"{f} missing at that ref")
    step(f"vast-ai/pyworker@{b['pyworker_ref'][:12]}", _pyworker)
    ok = image_exists(l["image"])
    if ok is False:
        alt = nearest_older_image(l["image"])
        hint = f"; newest older build with an image: {alt}" if alt else ""
        say(f"  FAIL  image {l['image']} not found{hint}")
        problems.append(f"image {l['image']} not found{hint}")
    elif ok is None:
        say(f"  warn  could not verify image {l['image']}")
    else:
        say(f"  ok    image {l['image']}")

    say("container settings")
    def _opts():
        docker_options(cfg, {**pins, "orch_ref": pins.get("orch_ref") or "x",
                             "model_revision": pins.get("model_revision") or "x"})
    step("docker options", _opts)

    say("offers matching workergroup.search_params")
    if not re.search(r"\bcompute_cap\s*>=?\s*\d+", cfg["workergroup"]["search_params"]):
        say("  warn  no compute_cap>= filter: old cards (Pascal P40, Volta V100, Turing) can be rented. "
            "Add compute_cap>=800 (Ampere and newer)")
    try:
        v = v or vast()
        offers = v.search_offers(query=cfg["workergroup"]["search_params"] + " rented=False",
                                 order="dph_total", limit=8, storage=cfg["template"]["disk_gb"])
        if not isinstance(offers, list) or not offers:
            problems.append("no offers match search_params right now")
            say("  FAIL  no offers match right now (loosen dph_total or inet_down?)")
        else:
            for o in offers[:8]:
                say(f"  {o.get('gpu_name', '?'):<18} {o.get('gpu_ram', 0) / 1000:>5.0f} GB  "
                    f"${o.get('dph_total', 0):.3f}/hr  down {o.get('inet_down', 0):>6.0f} Mbps  "
                    f"cuda {o.get('cuda_max_good', '?')}  cc {o.get('compute_cap', '?')}  rel {(o.get('reliability') or o.get('reliability2') or 0):.3f}")
    except SystemExit:
        raise
    except Exception as e:
        say(f"  warn  offer search failed: {e}")

    if problems:
        raise CheckFailed(f"{len(problems)} problem(s); nothing was changed")
    return pins


# ── state ────────────────────────────────────────────────────────────────────
POLL_S = 5               # seconds between checks while waiting on Vast
CONFIRM_WAIT_S = 60      # how long a just-created endpoint/workergroup may take to show up
DESTROY_WAIT_S = 300     # how long destroyed instances may take to disappear


def load_state():
    """state.json, or {} when there is none. A corrupt file is an error: it is
    how a re-run knows what an earlier, interrupted run already created."""
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise CheckFailed(f"{STATE_PATH} is unreadable ({e}); fix or remove it after checking the dashboard")


def save_state(st):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_PATH)


LOCK_WAIT_S = 60        # how long a command waits for another run (e.g. a watch tick) to finish


class StateLock:
    """Stops two apply/destroy/sweep/watch runs from racing each other into
    duplicates. An flock: the kernel releases it if the holder dies, so a
    crashed or killed run (or container) never leaves a stale lock behind."""

    def __enter__(self):
        self.path = STATE_PATH + ".lock"
        self.f = open(self.path, "a+")
        deadline = time.time() + LOCK_WAIT_S
        while True:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.time() >= deadline:
                    self.f.close()
                    raise CheckFailed(f"another apply/destroy/sweep/watch run holds {self.path}; "
                                      "retry when it finishes")
                time.sleep(0.5)

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _rows(result, what):
    if not isinstance(result, list):
        raise ApiError(f"{what} returned {str(result)[:300]!r} instead of a list; refusing to guess")
    return result


def find_endpoint(v, name, st=None, allow_state_id=False):
    """The one endpoint named `name`, or None.

    Raises on API errors and on duplicate names. If state.json's endpoint now
    goes by another name, it is returned when allow_state_id (destroy, sweep,
    status) and refused otherwise, since creating a fresh one would leave it
    running."""
    rows = _rows(v.show_endpoints(), "show_endpoints")
    matches = [r for r in rows if r.get("endpoint_name") == name]
    if len(matches) > 1:
        ids = ", ".join(str(r.get("id")) for r in matches)
        raise ApiError(f"{len(matches)} endpoints are named {name!r} (ids {ids}) and each can bill its own "
                       "workers. Delete the extras in the Vast dashboard, then retry")
    if matches:
        return matches[0]
    known = (st or {}).get("endpoint_id")
    for r in rows:
        if known is not None and r.get("id") == known:
            if allow_state_id:
                say(f"note: endpoint {known} from state.json is now named {r.get('endpoint_name')!r}")
                return r
            raise ApiError(f"state.json says this tool created endpoint {known}, now named "
                           f"{r.get('endpoint_name')!r} instead of {name!r}. Rename it back or destroy it "
                           "first; creating another would leave it running")
    return None


def find_workergroups(v, endpoint_id):
    rows = _rows(v.show_workergroups(), "show_workergroups")
    return [w for w in rows if w.get("endpoint_id") == endpoint_id]


def _created_id(res):
    if isinstance(res, dict):
        for k in ("result", "id", "endpoint_id", "autojob_id"):
            if isinstance(res.get(k), int) and not isinstance(res.get(k), bool):
                return res[k]
    return None


def _wait_for(fn, what):
    deadline = time.time() + CONFIRM_WAIT_S
    while True:
        got = fn()
        if got:
            return got
        if time.time() >= deadline:
            raise ApiError(f"{what} was created but hasn't shown up after {CONFIRM_WAIT_S}s. Don't re-run "
                           "apply until `deploy.py status` or the dashboard shows it; state.json marks it "
                           "pending so a re-run refuses instead of creating a second one")
        time.sleep(POLL_S)


def confirm(prompt, yes):
    if yes:
        return True
    # `docker compose run` without -it, cron, CI: no one can answer, so stop
    # before doing anything rather than crash on the prompt.
    no_tty = SystemExit(f"{prompt}\nno terminal to confirm on; pass --yes to go ahead without asking")
    if not sys.stdin.isatty():
        raise no_tty
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        raise no_tty from None


# ── verify after write ───────────────────────────────────────────────────────
def _same(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def _verify(row, want, what):
    """Compare what Vast now reports with what we asked for. A mismatch is an
    error; a field Vast doesn't report is named, never assumed applied."""
    if row is None:
        raise ApiError(f"{what} is not listed after the change")
    bad = [f"{k}: asked {val!r}, Vast reports {row[k]!r}" for k, val in want.items()
           if k in row and not _same(row[k], val)]
    if bad:
        raise ApiError(f"{what} did not take the settings: " + "; ".join(bad))
    missing = [k for k in want if k not in row]
    if missing:
        say(f"  warn  Vast doesn't report {', '.join(missing)} for {what}; can't confirm them")
    else:
        say(f"  ok    {what} confirmed: " + ", ".join(f"{k}={row[k]}" for k in want))
    return missing


def verify_endpoint(v, ep_id, want):
    rows = _rows(v.show_endpoints(), "show_endpoints")
    return _verify(next((r for r in rows if r.get("id") == ep_id), None), want, f"endpoint {ep_id}")


def verify_workergroup(v, wg_id, want):
    rows = _rows(v.show_workergroups(), "show_workergroups")
    return _verify(next((r for r in rows if r.get("id") == wg_id), None), want, f"workergroup {wg_id}")


def endpoint_limits(cfg):
    e = cfg["endpoint"]
    return {k: e[k] for k in ("max_workers", "cold_workers", "min_load", "target_util",
                              "cold_mult", "inactivity_timeout") if k in e}


# ── instances ────────────────────────────────────────────────────────────────
# Attribution uses only what Vast reports: the autoscaler's worker list for
# the endpoint, and instance ids Vast returned when this tool created them.
# Every id ever seen as a worker or created as a test rental is recorded in
# state.json. Nothing is inferred from env vars, images or templates; anything
# Vast lists that isn't recorded is "unattributed": shown with its cost, and
# destroyed only when named explicitly (sweep --destroy ID).
BOOTING = {"creating", "created", "pending", "loading", "model_loading", "starting"}


def _dph(inst):
    try:
        return float(inst.get("dph_total") or 0)
    except (TypeError, ValueError):
        return 0.0


def _status(inst):
    return str(inst.get("actual_status") or inst.get("cur_state") or "?")


def _describe(inst, note=""):
    return (f"instance {inst.get('id')}: {_status(inst)}, {inst.get('gpu_name', '?')}, "
            f"${_dph(inst):.3f}/hr" + (f", label {inst['label']!r}" if inst.get("label") else "")
            + (f" ({note})" if note else ""))


def endpoint_workers(v, endpoint_id):
    """The autoscaler's worker rows for this endpoint. Raises when it can't
    say, so nothing is destroyed on a guess."""
    out = v.get_endpoint_workers(endpoint_id)
    if isinstance(out, dict) and out.get("error_msg"):
        raise ApiError(f"get_endpoint_workers: {out['error_msg']}")
    return [r for r in _rows(out, "get_endpoint_workers") if isinstance(r, dict)]


def survey(v, cfg, st, ep, min_age, now=None):
    """Sort every instance on the account using recorded facts only.

    Updates st in place: records new worker ids, drops records of instances
    Vast no longer lists. The caller saves it."""
    now = now or time.time()
    instances = _rows(v.show_instances(), "show_instances")
    workers = endpoint_workers(v, ep["id"]) if ep else []
    live = {w.get("id") for w in workers}
    seen = st.setdefault("seen_workers", {})
    manual = st.setdefault("manual_instances", {})
    for wid in live:
        seen.setdefault(str(wid), now)
    listed = {str(i.get("id")) for i in instances}
    for k in [k for k in seen if k not in listed and int(k) not in live]:
        del seen[k]
    # A rental Vast just created may not be listed yet; forget it only once
    # it has had time to appear.
    for k in [k for k in manual if k not in listed and now - manual[k].get("created_at", 0) > 600]:
        del manual[k]
    ttl = cfg.get("limits", {}).get("manual_ttl_s", 14400)
    out = {"workers": [], "manual": [], "orphans": [], "young": [], "unattributed": [], "worker_rows": workers}
    for i in instances:
        key = str(i.get("id"))
        if i.get("id") in live:
            out["workers"].append(i)
        elif key in manual:
            age = now - manual[key].get("created_at", now)
            (out["orphans"] if ttl and age >= ttl else out["manual"]).append(i)
        elif key in seen:
            # Was a worker of ours, isn't one now.
            (out["orphans"] if now - seen[key] >= min_age else out["young"]).append(i)
        else:
            out["unattributed"].append(i)
    return out


def destroy_and_wait(v, ids):
    """Destroy instances and poll until Vast no longer lists them."""
    ids = set(ids)
    for i in sorted(ids):
        try:
            v.destroy_instance(i)
            say(f"  instance {i}: destroy requested")
        except Exception as e:
            say(f"  instance {i}: destroy failed ({e}); will retry")
    deadline, last_retry = time.time() + DESTROY_WAIT_S, time.time()
    while True:
        try:
            left = ids & {i.get("id") for i in _rows(v.show_instances(), "show_instances")}
        except Exception as e:
            say(f"  could not list instances ({e}); retrying")
            left = ids
        if not left:
            say(f"  confirmed gone: {', '.join(str(i) for i in sorted(ids))}")
            return
        if time.time() >= deadline:
            raise CheckFailed(f"instances {sorted(left)} still exist after {DESTROY_WAIT_S}s and may still be "
                              "billing. Destroy them in the Vast dashboard, then run `deploy.py sweep`")
        if time.time() - last_retry >= 30:
            last_retry = time.time()
            for i in sorted(left):
                try:
                    v.destroy_instance(i)
                except Exception:
                    pass
        time.sleep(POLL_S)


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_check(cfg, args):
    say("preflight")
    check(cfg)
    say("\nall checks passed")


def cmd_apply(cfg, args):
    v = vast()
    say("preflight")
    pins = check(cfg, v)
    opts = docker_options(cfg, pins)
    say("\npinned:")
    for k, val in pins.items():
        say(f"  {k} = {val}")
    if args.dry_run:
        say("\ntemplate docker options:\n  " + opts)
        say("\ndry run: nothing changed")
        return
    if not confirm("\ncreate/update the template, endpoint and workergroup?", args.yes):
        return
    with StateLock():
        _apply(v, cfg, pins, opts)


def _apply(v, cfg, pins, opts):
    e, w, t = cfg["endpoint"], cfg["workergroup"], cfg["template"]
    st = load_state()

    # Look everything up before creating anything, so an API error or an
    # ambiguous answer stops the run while nothing new exists yet.
    ep = find_endpoint(v, e["name"], st)
    if ep is None and st.get("endpoint_pending") == e["name"]:
        raise ApiError(f"an earlier apply created endpoint {e['name']!r} but never saw it appear. Check the "
                       "dashboard; if it really doesn't exist, delete endpoint_pending from state.json")
    groups = find_workergroups(v, ep["id"]) if ep else []
    if len(groups) > 1:
        raise ApiError(f"endpoint {ep['id']} has {len(groups)} workergroups "
                       f"({', '.join(str(g.get('id')) for g in groups)}); this tool manages exactly one. "
                       "Delete the extras, then retry")
    if ep:
        st.pop("endpoint_pending", None)
    if groups:
        st.pop("workergroup_pending", None)
    if ep and not groups and st.get("workergroup_pending") == ep["id"]:
        raise ApiError(f"an earlier apply created a workergroup for endpoint {ep['id']} but never saw it "
                       "appear. Check the dashboard; if it really doesn't exist, delete workergroup_pending "
                       "from state.json")

    tpl = v.create_template(
        name=t["name"], image=cfg["llama"]["image"], env=opts, onstart_cmd=onstart_script(),
        ssh=True, direct=True, disk_space=float(t["disk_gb"]),
        search_params=w["search_params"], desc="vastai-llmapi-orchestrator worker", public=False)
    tpl_obj = tpl.get("template") if isinstance(tpl, dict) else None
    tpl_hash = (tpl_obj or {}).get("hash_id") or (tpl_obj or {}).get("hash") or tpl.get("hash_id")
    if not tpl_hash:
        raise SystemExit(f"template creation returned no hash: {tpl}")
    st["template_hash"] = tpl_hash
    st["template_hashes"] = (st.get("template_hashes", []) + [tpl_hash])[-20:]
    st["pins"] = pins
    save_state(st)
    say(f"template {tpl_hash} ({t['name']})")

    ep_fields = endpoint_limits(cfg)
    if ep:
        v.update_endpoint(ep["id"], endpoint_name=e["name"], **ep_fields)
        say(f"endpoint {ep['id']} updated")
        ep_id = ep["id"]
    else:
        st["endpoint_pending"] = e["name"]
        save_state(st)
        res = v.create_endpoint(endpoint_name=e["name"], **ep_fields)
        if _created_id(res) is not None:
            st["endpoint_id"] = _created_id(res)
            save_state(st)
        ep = _wait_for(lambda: find_endpoint(v, e["name"], st), f"endpoint {e['name']!r}")
        ep_id = ep["id"]
        st.pop("endpoint_pending", None)
        say(f"endpoint {ep_id} created")
    st.update({"endpoint_id": ep_id, "endpoint_name": e["name"]})
    save_state(st)
    verify_endpoint(v, ep_id, ep_fields)

    if groups:
        g = groups[0]
        # The SDK appends its default filters (verified, rentable, not rented)
        # itself, so pass the configured search unchanged.
        v.update_workergroup(g["id"], template_hash=tpl_hash, search_params=w["search_params"],
                             gpu_ram=w.get("gpu_ram"), endpoint_id=ep_id)
        say(f"workergroup {g['id']} now uses template {tpl_hash}")
        say(f"existing workers keep running the old template until replaced; roll them with "
            f"`vastai update workers {g['id']}` (or destroy + apply)")
        st["workergroup_id"] = g["id"]
        save_state(st)
        verify_workergroup(v, g["id"], {"template_hash": tpl_hash})
    else:
        st["workergroup_pending"] = ep_id
        save_state(st)
        # Raw POST /autojobs/ instead of v.create_workergroup(): the REST API
        # (docs.vast.ai/api-reference/serverless/create-workergroup) takes
        # test_workers, cold_workers, max_workers and min_load per workergroup,
        # with defaults of 3 / 3 / 20 / 1, but the SDK's create_workergroup
        # drops them. Send them all so no Vast default applies. The search
        # gets the same default filters create_workergroup would add.
        blob = {"client_id": "me", "template_hash": tpl_hash,
                "search_params": (w["search_params"] + " verified=True rentable=True rented=False").strip(),
                "gpu_ram": w.get("gpu_ram"), "endpoint_id": ep_id, "endpoint_name": e["name"],
                "test_workers": w["test_workers"], "cold_workers": e["cold_workers"],
                "max_workers": e["max_workers"],
                **{k: e[k] for k in ("min_load", "target_util", "cold_mult") if k in e},
                "autoscaler_instance": "prod"}
        r = v.client.post("/autojobs/", json_data=blob)
        r.raise_for_status()
        res = r.json()
        if isinstance(res, dict) and res.get("success") is False:
            raise ApiError(f"workergroup creation refused: {str(res)[:300]}")
        made = _wait_for(lambda: find_workergroups(v, ep_id), f"workergroup for endpoint {ep_id}")
        st.pop("workergroup_pending", None)
        st["workergroup_id"] = _created_id(res) or made[0].get("id")
        say(f"workergroup {st['workergroup_id']} created")
        save_state(st)
        unconfirmed = verify_workergroup(v, st["workergroup_id"], {
            k: blob[k] for k in ("template_hash", "test_workers", "cold_workers", "max_workers", "min_load")
            if k in blob})
        if unconfirmed:
            say("  the endpoint's own max_workers (confirmed above) still caps workers billed at once")
    save_state(st)
    say("\ndone. watch it with: deploy.py status   (first worker: download + load + benchmark)")


def _report(r):
    for label, key in (("serverless workers (per the autoscaler)", "workers"),
                       ("test rentals made by rent-test", "manual"),
                       ("ORPHANS: recorded as ours, no longer a worker, or past their TTL", "orphans"),
                       ("former workers, too recent to call orphans", "young")):
        if r[key]:
            say(f"{label}:")
            for i in r[key]:
                say(f"  {_describe(i)}")
    mine = r["workers"] + r["manual"] + r["orphans"] + r["young"]
    say(f"this deployment: {len(mine)} instance(s), ${sum(_dph(i) for i in mine):.3f}/hr")
    if r["unattributed"]:
        say("instances this tool has no record of (left alone; `sweep --destroy ID` to remove one):")
        for i in r["unattributed"]:
            say(f"  {_describe(i)}")
    total = sum(_dph(i) for i in mine + r["unattributed"])
    say(f"whole account: ${total:.3f}/hr")
    if r["orphans"]:
        say("run `deploy.py sweep` to destroy the orphans")


def cmd_status(cfg, args):
    v = vast()
    st = load_state()
    ep = find_endpoint(v, cfg["endpoint"]["name"], st, allow_state_id=True)
    if not ep:
        say("endpoint not found")
    else:
        say(json.dumps({k: ep.get(k) for k in ("id", "endpoint_name", "endpoint_state", "max_workers",
                                               "cold_workers", "min_load", "inactivity_timeout")}, indent=2))
        for g in find_workergroups(v, ep["id"]):
            say(json.dumps({k: g.get(k) for k in ("id", "template_hash", "search_params", "gpu_ram",
                                                  "test_workers", "cold_workers")}, indent=2))
    try:
        with StateLock():
            st = load_state()
            r = survey(v, cfg, st, ep, args.min_age)
            save_state(st)
    except CheckFailed as e:
        say(f"(could not check instances: {e})")
        return
    for w in r["worker_rows"]:
        say(f"worker {w.get('id')}: {w.get('status')}")
    _report(r)


def cmd_sweep(cfg, args):
    v = vast()
    with StateLock():
        st = load_state()
        ep = find_endpoint(v, cfg["endpoint"]["name"], st, allow_state_id=True)
        r = survey(v, cfg, st, ep, args.min_age)
        save_state(st)
        _report(r)
        targets = list(r["orphans"])
        if args.destroy:
            by_id = {i.get("id"): i for i in r["unattributed"] + r["manual"] + r["young"] + r["orphans"]}
            live = {i.get("id") for i in r["workers"]}
            for i in args.destroy:
                if i in live:
                    raise CheckFailed(f"instance {i} is a live worker of the endpoint; use destroy or pause")
                if i not in by_id:
                    raise CheckFailed(f"instance {i} is not on this account")
                if by_id[i] not in targets:
                    targets.append(by_id[i])
        if not targets:
            say("nothing to destroy")
            return
        if args.dry_run:
            say("dry run: nothing destroyed")
            return
        for i in targets:
            say(f"  will destroy {_describe(i)}")
        cost = sum(_dph(i) for i in targets)
        if not confirm(f"destroy {len(targets)} instance(s) (${cost:.3f}/hr)?", args.yes):
            return
        destroy_and_wait(v, [i["id"] for i in targets])
        for i in targets:
            st["seen_workers"].pop(str(i["id"]), None)
            st["manual_instances"].pop(str(i["id"]), None)
        save_state(st)


LOGS_POLL_S = 20
GONE = {"exited", "stopped", "offline", "destroyed"}


def _log_text(out):
    """vastai logs gives the log text, or a dict when Vast has no log to hand back."""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        return str(out.get("msg") or out.get("error") or json.dumps(out))
    return str(out)


def follow_instance_logs(v, iid, timeout, once=False, poll=LOGS_POLL_S):
    """Print an instance's container log as it grows until a boot marker shows
    up. Returns "ready", "fatal", "gone", "timeout" or "once"."""
    seen, t0, note = set(), time.time(), None
    while True:
        try:
            text = _log_text(v.logs(iid, tail="1000"))
        except Exception as e:   # a log request that isn't ready yet is normal early on
            text, msg = "", f"no log from Vast yet ({type(e).__name__}: {str(e)[:200]})"
            if msg != note:
                say(msg)
                note = msg
        marker = None
        for line in text.splitlines():
            if line in seen:
                continue
            seen.add(line)
            say(line)
            if "ORCH_FATAL:" in line:
                marker = marker or "fatal"
            elif "ORCH_READY model=" in line:
                marker = marker or "ready"
        if marker or once:
            return marker or "once"
        inst = v.show_instance(iid)
        status = str((inst or {}).get("actual_status") or "").lower()
        if not inst or status in GONE:
            say(f"instance {iid} is {status or 'gone'} and printed no ORCH_READY or ORCH_FATAL")
            return "gone"
        if time.time() - t0 >= timeout:
            say(f"no ORCH_READY or ORCH_FATAL from instance {iid} within {timeout:.0f}s "
                f"(status {status or 'unknown'}); the full boot log is /workspace/orch/model.log on it")
            return "timeout"
        time.sleep(poll)


def cmd_logs(cfg, args):
    v = vast()
    if getattr(args, "instance_id", None) is not None:
        # The worker gives up at its own boot and download limits; allow for
        # the image pull before boot.sh starts on top of that.
        timeout = args.timeout or boot_limit_s(cfg) + 900
        r = follow_instance_logs(v, args.instance_id, timeout, once=args.once)
        if r in ("fatal", "gone", "timeout"):
            sys.exit(1)
        return
    ep = find_endpoint(v, cfg["endpoint"]["name"], load_state(), allow_state_id=True)
    if not ep:
        raise SystemExit("endpoint not found")
    out = v.get_endpt_logs(ep["id"], level=1, tail=200)
    say(json.dumps(out, indent=2) if not isinstance(out, str) else out)


def set_endpoint_state(v, cfg, ep, state):
    # Send every configured limit with the new state, so nothing depends on
    # how Vast treats fields an update leaves out, then read it back.
    v.update_endpoint(ep["id"], endpoint_name=cfg["endpoint"]["name"], endpoint_state=state,
                      **endpoint_limits(cfg))
    verify_endpoint(v, ep["id"], {"endpoint_state": state, **endpoint_limits(cfg)})
    say(f"endpoint {ep['id']} -> {state}")


def _set_state(cfg, state):
    v = vast()
    ep = find_endpoint(v, cfg["endpoint"]["name"], load_state(), allow_state_id=True)
    if not ep:
        raise SystemExit("endpoint not found")
    set_endpoint_state(v, cfg, ep, state)


def cmd_destroy(cfg, args):
    v = vast()
    with StateLock():
        st = load_state()
        ep = find_endpoint(v, cfg["endpoint"]["name"], st, allow_state_id=True)
        groups = find_workergroups(v, ep["id"]) if ep else []
        try:
            r = survey(v, cfg, st, ep, 0)
        except ApiError as e:
            # Deleting the endpoint destroys its workers anyway; go on with
            # what state.json recorded.
            say(f"warn: {e}; continuing with recorded instances only")
            r = survey(v, cfg, st, None, 0)
        save_state(st)
        mine = r["workers"] + r["manual"] + r["orphans"] + r["young"]
        if not ep and not groups and not mine:
            say("no endpoint and no recorded instances from this deployment; nothing to do")
            _report(r)
            return
        for i in mine:
            say(f"  {_describe(i)}")
        what = f"endpoint {ep['id']} and {len(groups)} workergroup(s)" if ep else "no endpoint (already gone)"
        if not confirm(f"delete {what}, and destroy the {len(mine)} instance(s) listed above "
                       "(test rentals included)? Cached weights go away.", args.yes):
            return
        failed = []
        if ep:
            # Deleting the endpoint deletes its workergroups and destroys their
            # workers, reporting deleted_workers / failed_workers
            # (docs.vast.ai/api-reference/serverless/delete-endpoint). Deleting
            # a workergroup on its own does not destroy its instances, so the
            # endpoint goes first and alone.
            try:
                res = v.delete_endpoint(ep["id"])
                if isinstance(res, dict) and res.get("success") is False:
                    raise ApiError(str(res)[:300])
                res = res if isinstance(res, dict) else {}
                say(f"endpoint {ep['id']} deleted; workers destroyed: {res.get('deleted_workers', '?')}"
                    + (f", FAILED: {res['failed_workers']}" if res.get("failed_workers") else ""))
                for wid in res.get("failed_workers") or []:
                    st["seen_workers"].setdefault(str(wid), time.time())
            except Exception as e:   # keep going: the instances are what bill
                say(f"endpoint {ep['id']}: delete failed ({e})")
                failed.append(f"endpoint {ep['id']}")
        # Destroy and confirm every recorded instance Vast still lists:
        # failed_workers, test rentals, workers seen earlier. A second pass
        # catches workers launched while the endpoint was being deleted.
        done = set()
        for _ in range(2):
            listed = {i.get("id") for i in _rows(v.show_instances(), "show_instances")}
            recorded = {int(k) for k in st["seen_workers"]} | {int(k) for k in st["manual_instances"]}
            if ep and not failed:
                try:
                    recorded |= {w.get("id") for w in endpoint_workers(v, ep["id"])}
                except Exception:
                    pass   # endpoint gone, as intended
            ids = (recorded & listed) - done
            if not ids:
                break
            destroy_and_wait(v, ids)
            done |= ids
        if failed:
            raise CheckFailed(f"recorded instances are gone, but deleting {', '.join(failed)} failed; the "
                              "autoscaler may start new workers. Re-run destroy or delete it in the dashboard")
        for k in ("endpoint_id", "endpoint_name", "workergroup_id", "endpoint_pending", "workergroup_pending"):
            st.pop(k, None)
        st["seen_workers"], st["manual_instances"] = {}, {}
        save_state(st)
        say("destroyed; no recorded instances from this deployment remain")
        if r["unattributed"]:
            say(f"{len(r['unattributed'])} instance(s) on the account were never recorded by this tool and "
                "were left alone; see `deploy.py status`")


def cmd_rent_test(cfg, args):
    """Rent one instance outside serverless (no PyWorker) to prove a config,
    recording the id Vast returns so status, sweep, watch and destroy know it."""
    v = vast()
    say("preflight")
    pins = check(cfg, v)
    ttl = int(cfg["limits"].get("manual_ttl_s", 14400))
    opts = docker_options(cfg, pins) + f" -e ORCH_SKIP_PYWORKER=1 -e ORCH_MANUAL_TTL={ttl}"
    t, w = cfg["template"], cfg["workergroup"]
    offers = _rows(v.search_offers(query=w["search_params"] + " rented=False", order="dph_total",
                                   limit=1, storage=t["disk_gb"]), "search_offers")
    if not offers:
        raise CheckFailed("no offer matches workergroup.search_params right now")
    o = offers[0]
    say(f"cheapest match: offer {o.get('id')} {o.get('gpu_name', '?')} ${o.get('dph_total', 0):.3f}/hr")
    if args.dry_run:
        say("docker options:\n  " + opts + "\ndry run: nothing rented")
        return
    if not confirm(f"rent it? It stops itself after {ttl}s; `deploy.py destroy` or `sweep` removes it.", args.yes):
        return
    label = f"orch-test:{cfg['endpoint']['name']}"
    with StateLock():
        st = load_state()
        res = v.create_instance(o["id"], image=cfg["llama"]["image"], disk=float(t["disk_gb"]), env=opts,
                                onstart_cmd=onstart_script(), label=label, ssh=True, direct=True,
                                cancel_unavail=True)
        iid = res.get("new_contract") if isinstance(res, dict) else None
        if not isinstance(iid, int):
            raise ApiError(f"create_instance returned no instance id: {str(res)[:300]}. Check the dashboard "
                           f"for an instance labelled {label!r} and destroy it if present")
        st.setdefault("manual_instances", {})[str(iid)] = {"created_at": time.time(), "label": label}
        save_state(st)
        say(f"instance {iid} rented and recorded")
        inst = _wait_for(lambda: next((i for i in _rows(v.show_instances(), "show_instances")
                                       if i.get("id") == iid), None), f"instance {iid}")
        say(f"  {_describe(inst)}")
    say(f"follow the boot with: deploy.py logs {iid}")


# ── watch ────────────────────────────────────────────────────────────────────
def watch_tick(v, cfg, st, now=None):
    """One pass of the watchdog. Acts only on what Vast reports; returns the
    actions taken (for logs and tests)."""
    now = now or time.time()
    actions = []
    ep = find_endpoint(v, cfg["endpoint"]["name"], st, allow_state_id=True)
    r = survey(v, cfg, st, ep, cfg.get("limits", {}).get("orphan_min_age_s", 900), now)

    # 1. Workers Vast still reports as booting long past our own boot
    #    deadline: they bill (Creating/Loading/Starting are billed states) and
    #    will not become ready. Status strings we don't know are left alone.
    limit = boot_limit_s(cfg) + cfg.get("limits", {}).get("stuck_grace_s", 600)
    stuck = [w for w in r["worker_rows"] if str(w.get("status", "")).lower() in BOOTING
             and now - st["seen_workers"].get(str(w.get("id")), now) >= limit]
    # 2. Instances recorded as ours that are no longer workers, or test
    #    rentals past their TTL.
    targets = {w.get("id") for w in stuck} | {i.get("id") for i in r["orphans"]}
    for w in stuck:
        actions.append(f"destroy stuck worker {w.get('id')} ({w.get('status')} for >{limit}s)")
    for i in r["orphans"]:
        actions.append(f"destroy orphan {_describe(i)}")
    if targets:
        destroy_and_wait(v, targets)
        for i in targets:
            st["seen_workers"].pop(str(i), None)
            st["manual_instances"].pop(str(i), None)

    # 3. Whole-account spend, from Vast's own dph_total, against the budget.
    budget = float(cfg["limits"]["max_hourly_usd"])
    running = [i for i in r["workers"] + r["manual"] + r["young"] + r["unattributed"]
               if i.get("id") not in targets and _status(i) == "running"]
    burn = sum(_dph(i) for i in running)
    if burn > budget + 1e-9:
        msg = f"account burn ${burn:.3f}/hr is over limits.max_hourly_usd={budget:.2f}"
        if ep and cfg["limits"].get("on_breach", "pause") == "pause" and ep.get("endpoint_state") != "stopped":
            set_endpoint_state(v, cfg, ep, "stopped")
            actions.append(msg + "; endpoint paused (resume with `deploy.py resume`)")
        else:
            actions.append(msg + "; alert only")
    for a in actions:
        say(f"[watch] {a}")
    return actions


def cmd_watch(cfg, args):
    check_limits(cfg)
    v = vast()
    say(f"[watch] every {args.interval:.0f}s; acting only on what Vast reports")
    while True:
        try:
            with StateLock():
                st = load_state()
                try:
                    watch_tick(v, cfg, st)
                finally:
                    save_state(st)
        except Exception as e:   # keep watching; say plainly what we couldn't see
            say(f"[watch] could not check Vast this round: {e}")
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.interval)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "apply", "status", "logs", "pause", "resume", "sweep",
                                       "destroy", "rent-test", "watch"])
    p.add_argument("instance_id", nargs="?", type=int,
                   help="logs: an instance id to follow until its boot prints ORCH_READY or ORCH_FATAL")
    p.add_argument("--config", default=os.environ.get("ORCH_CONFIG") or os.path.join(HERE, "config.toml"))
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--min-age", type=float, default=900,
                   help="sweep/status: a former worker must be gone from the worker list this long "
                        "(seconds) before it counts as an orphan")
    p.add_argument("--destroy", type=int, nargs="+", default=[], metavar="ID",
                   help="sweep: also destroy these instance ids (e.g. ones listed as unrecorded)")
    p.add_argument("--interval", type=float, default=60, help="watch: seconds between checks")
    p.add_argument("--once", action="store_true",
                   help="watch: one check, then exit; logs ID: print the log once, don't wait")
    p.add_argument("--timeout", type=float, default=0,
                   help="logs ID: seconds to wait for a marker (default: boot.deadline_s + download_max_s + 900)")
    args = p.parse_args()
    if args.instance_id is not None and args.command != "logs":
        p.error(f"{args.command} takes no instance id")
    cfg = load_config(args.config)
    try:
        {"check": cmd_check, "apply": cmd_apply, "status": cmd_status, "logs": cmd_logs,
         "pause": lambda c, a: _set_state(c, "stopped"),
         "resume": lambda c, a: _set_state(c, "active"),
         "sweep": cmd_sweep, "destroy": cmd_destroy, "rent-test": cmd_rent_test,
         "watch": cmd_watch}[args.command](cfg, args)
    except CheckFailed as e:
        say(f"\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

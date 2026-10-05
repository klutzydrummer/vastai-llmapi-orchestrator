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
import threading
import time
import tomllib
import urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# In the container this points at a mounted volume (see compose.yaml).
STATE_PATH = os.environ.get("ORCH_STATE_PATH") or os.path.join(HERE, "state.json")
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
GH_API = "https://api.github.com"
GH_RAW = "https://raw.githubusercontent.com"
UA = {"User-Agent": "vastai-llmapi-orchestrator-deploy"}

# The worker's GPU memory planner; `check` runs the same one before anything is rented.
sys.path.insert(0, os.path.join(REPO_ROOT, "worker"))
import vram  # noqa: E402


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
        "LLAMA_CTX_MIN": l.get("ctx_min"),
        "LLAMA_UBATCH": l.get("ubatch"),
        "LLAMA_IMAGE_TOKENS": l.get("image_tokens"),
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
            "EMBED_CTX": e.get("ctx", 4096),
            "EMBED_POOLING": e.get("pooling", "last"),
            "EMBED_GPU": "1" if e.get("gpu", True) else "0",
            "EMBED_CACHE_TYPE": e.get("cache_type", "f16"),
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
    parse_warm_hours(e.get("warm_hours", ""))
    if e.get("warm_tz"):
        try:
            ZoneInfo(e["warm_tz"])
        except Exception:
            raise CheckFailed(f"endpoint.warm_tz: unknown time zone {e['warm_tz']!r} "
                              "(use an IANA name like \"America/Chicago\")")
    wml = e.get("warm_min_load", 1)
    if isinstance(wml, bool) or not isinstance(wml, (int, float)) or wml <= 0:
        raise CheckFailed("endpoint.warm_min_load must be a number above 0")
    if lim.get("on_breach", "pause") not in ("pause", "alert"):
        raise CheckFailed("limits.on_breach must be \"pause\" or \"alert\"")
    ra = lim.get("rent_attempts", 3)
    if not isinstance(ra, int) or isinstance(ra, bool) or not 1 <= ra <= 10:
        raise CheckFailed("limits.rent_attempts must be a whole number from 1 to 10")
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


# ── GPU memory preflight ─────────────────────────────────────────────────────
CARD_GIB = (8, 10, 11, 12, 16, 20, 24, 32, 40, 45, 48, 80, 94, 96, 141)
# What the driver keeps with nothing running: rental 54231885's RTX 3090
# reported 23859 of 24576 MiB free.
DRIVER_RESERVE_MIB = 720


def smallest_gpu_mib(cfg):
    """Memory of the smallest card the search can rent: the next card size at
    or above both search_params' gpu_ram floor and llama.min_vram_gb."""
    m = re.search(r"\bgpu_ram\s*(>=|>)\s*([0-9]*\.?[0-9]+)", cfg["workergroup"]["search_params"])
    floor = max(float(m.group(2)) if m else 0.0, float(cfg["llama"]["min_vram_gb"]))
    return next((g for g in CARD_GIB if g >= floor), floor) * 1024


def plan_settings(cfg):
    """The planner's settings from the config, as the worker reads them from its env."""
    l, e = cfg["llama"], cfg.get("embedding") or {}
    cache = l["cache_type"]
    ecache = e.get("cache_type", "f16")
    for name, val in (("llama.cache_type", cache), ("embedding.cache_type", ecache)):
        if val not in vram.CACHE_BYTES:
            raise CheckFailed(f"{name}={val!r} is not one of {', '.join(vram.CACHE_BYTES)}")
    if not isinstance(e.get("gpu", True), bool):
        raise CheckFailed("embedding.gpu must be true or false")
    ctx_min = l.get("ctx_min", l["ctx"])
    if not 256 * l["parallel"] <= ctx_min <= l["ctx"]:
        raise CheckFailed(f"llama.ctx_min={ctx_min} must be between {256 * l['parallel']} (256 per slot) "
                          f"and llama.ctx={l['ctx']}")
    return {"ctx": l["ctx"], "ctx_min": ctx_min, "parallel": l["parallel"], "cache_type": cache,
            "ubatch": l.get("ubatch", 512), "image_tokens": l.get("image_tokens", 1120),
            "embed_ctx": e.get("ctx", 4096), "embed_cache": ecache, "embed_gpu": e.get("gpu", True),
            "mmproj_offload": "--no-mmproj-offload" not in str(l.get("extra_args", "")).replace(";", " ").split()}


def _hub_header(repo, sha, path):
    return vram.read_header(vram.HttpSource(vram.hub_url(repo, sha, path), _hf_headers()))


def gpu_memory_plan(cfg, files, read_header=_hub_header):
    """Whether the models fit the smallest GPU the search allows, from their
    headers (a few MB each, at the pinned revisions). files maps chat/mmproj/
    embed to (repo, sha, path, size). Returns the plan; prints the breakdown."""
    settings = plan_settings(cfg)
    total = smallest_gpu_mib(cfg)
    try:
        chat = vram.model_facts(read_header(*files["chat"][:3]), files["chat"][3])
        mm = vram.projector_facts(read_header(*files["mmproj"][:3]), files["mmproj"][3]) if files.get("mmproj") else None
        emb = (vram.model_facts(read_header(*files["embed"][:3]), files["embed"][3])
               if files.get("embed") and settings["embed_gpu"] else None)
    except (vram.HeaderError, OSError) as e:
        raise CheckFailed(f"can't read a model header: {e}")
    p = vram.plan(total - DRIVER_RESERVE_MIB, total, chat, settings, mm, emb)
    say(f"  info  sized for the smallest GPU the search allows: {total:.0f} MiB "
        f"(gpu_ram/min_vram_gb), {DRIVER_RESERVE_MIB} MiB of it kept by the driver")
    for line in vram.describe(p):
        say(f"        {line}")
    if not p["fits"]:
        options = ["a larger GPU (raise gpu_ram>= in search_params and llama.min_vram_gb)",
                   "a lower llama.ctx_min"]
        if settings["mmproj_offload"] and mm:
            options.append("the projector in system RAM (llama.extra_args = \"--no-mmproj-offload\"; slower images)")
        if emb:
            options.append("the embedding model on the CPU (embedding.gpu = false; slower embeddings)")
        raise CheckFailed(f"models don't fit a {total:.0f} MiB GPU: {p['short_mib']} MiB short even with chat "
                          f"context {p['ctx_min']}. Options (your call): " + "; ".join(options))
    return p


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
    sizes, files = [], {}
    if pins["model_revision"]:
        sz = step(m["file"], lambda: hf_file_size(m["repo"], pins["model_revision"], m["file"]))
        if sz:
            sizes.append(sz)
            files["chat"] = (m["repo"], pins["model_revision"], m["file"], sz)
    if m.get("mmproj_file"):
        mm_repo = m.get("mmproj_repo") or m["repo"]
        pins["mmproj_revision"] = step(f"{mm_repo}@{m.get('mmproj_revision', 'main')}",
                                       lambda: hf_resolve(mm_repo, m.get("mmproj_revision", "main")))
        if pins["mmproj_revision"]:
            sz = step(m["mmproj_file"], lambda: hf_file_size(mm_repo, pins["mmproj_revision"], m["mmproj_file"]))
            if sz:
                sizes.append(sz)
                files["mmproj"] = (mm_repo, pins["mmproj_revision"], m["mmproj_file"], sz)
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
                files["embed"] = (e["repo"], pins["embed_revision"], e["file"], sz)

    weights_gb = sum(sizes) / 1024**3
    if sizes:
        need_disk = weights_gb + 12   # image layers, pyworker venv, logs, headroom
        say(f"  info  weights {weights_gb:.1f} GiB; disk {cfg['template']['disk_gb']} GB (need ~{need_disk:.0f})")
        if cfg["template"]["disk_gb"] < need_disk:
            problems.append(f"template.disk_gb={cfg['template']['disk_gb']} is too small; use >= {need_disk:.0f}")
            say("  FAIL  disk too small")

    say("GPU memory")
    if "chat" in files and ("mmproj" in files or not m.get("mmproj_file")) and ("embed" in files or not e):
        step("models, context caches and buffers fit", lambda: gpu_memory_plan(cfg, files) and None)
    else:
        say("  skip  a model file could not be checked (see above)")

    say("worker code and image")
    pins["orch_ref"] = step(f"github {b['orch_repo']}@{b['orch_ref']}", lambda: gh_resolve(b["orch_repo"], b["orch_ref"]))
    if pins["orch_ref"]:
        for f in ("worker/onstart.sh", "worker/boot.sh", "worker/fetch_model.py", "worker/hf_download.py",
                  "worker/smoke_test.py",
                  "worker/router.py", "worker/pyworker_worker.py", "worker/vram.py"):
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


def parse_warm_hours(text):
    """"15:00-23:00" or "06:30-08:00,15:00-23:00" -> [(start_min, end_min), ...].
    A range may cross midnight ("22:00-02:00"). Empty -> [] (off)."""
    out = []
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        try:
            a, b = part.split("-")
            mins = []
            for t in (a, b):
                h, m = (int(x) for x in t.strip().split(":"))
                if not (0 <= h <= 24 and 0 <= m < 60) or (h == 24 and m):
                    raise ValueError
                mins.append(h * 60 + m)
        except ValueError:
            raise CheckFailed(f"endpoint.warm_hours: can't read {part!r}; use \"HH:MM-HH:MM\", comma-separated")
        if mins[0] == mins[1]:
            raise CheckFailed(f"endpoint.warm_hours: {part!r} is empty")
        out.append((mins[0], mins[1]))
    return out


def warm_now(cfg, now=None):
    """Whether endpoint.warm_hours covers `now` (epoch seconds), in
    endpoint.warm_tz (the machine's clock when unset)."""
    e = cfg["endpoint"]
    windows = parse_warm_hours(e.get("warm_hours", ""))
    if not windows:
        return False
    tz = ZoneInfo(e["warm_tz"]) if e.get("warm_tz") else None
    t = datetime.fromtimestamp(time.time() if now is None else now, tz)
    m = t.hour * 60 + t.minute
    return any((a <= m < b) if a < b else (m >= a or m < b) for a, b in windows)


def endpoint_limits(cfg, now=None):
    """The endpoint settings to send now. During endpoint.warm_hours,
    min_load is endpoint.warm_min_load: the autoscaler then keeps a worker
    running with no traffic, instead of releasing it after
    inactivity_timeout. Every write (apply, pause, resume, watch) goes
    through here, so none of them undoes the other."""
    e = cfg["endpoint"]
    out = {k: e[k] for k in ("max_workers", "cold_workers", "min_load", "target_util",
                             "cold_mult", "inactivity_timeout") if k in e}
    if warm_now(cfg, now):
        out["min_load"] = e.get("warm_min_load", 1)
    return out


# ── instances ────────────────────────────────────────────────────────────────
# Attribution uses only what Vast reports: the autoscaler's worker list for
# the endpoint, and instance ids Vast returned when this tool created them.
# Every id ever seen as a worker or created as a test rental is recorded in
# state.json. Nothing is inferred from env vars, images or templates; anything
# Vast lists that isn't recorded is "unattributed": shown with its cost, and
# destroyed only when named explicitly (sweep --destroy ID).
BOOTING = {"creating", "created", "pending", "loading", "model_loading", "starting"}
GONE = {"exited", "stopped", "offline", "destroyed"}


def _dph(inst):
    try:
        return float(inst.get("dph_total") or 0)
    except (TypeError, ValueError):
        return 0.0


def _status(inst):
    return str(inst.get("actual_status") or inst.get("cur_state") or "?")


def _status_msg(inst):
    return " ".join(str(inst.get("status_msg") or "").split())[:300]


def _describe(inst, note=""):
    msg = _status_msg(inst)
    return (f"instance {inst.get('id')}: {_status(inst)}, {inst.get('gpu_name', '?')}, "
            f"${_dph(inst):.3f}/hr" + (f", label {inst['label']!r}" if inst.get("label") else "")
            + (f", Vast says {msg!r}" if msg else "") + (f" ({note})" if note else ""))


# Words in Vast's status_msg that mean the container failed to start (seen:
# "Secrets fetch failed: network/subprocess error", stuck in loading 47 min).
VAST_ERROR = re.compile(r"\b(error|errors|failed|failure|unable|cannot|denied|no such)\b", re.I)


def vast_error(inst):
    """Vast's status message when it reports a failure for an instance whose
    container isn't running yet, else ""."""
    msg = _status_msg(inst)
    return msg if msg and _status(inst).lower() in BOOTING and VAST_ERROR.search(msg) else ""


def boot_state(inst, booting_for, stuck_s):
    """Classify one instance from what Vast reports for it (actual_status,
    status_msg) and how long it has been in a booting status. Returns
    (state, detail), state being one of:

      "gone"     not listed, or exited/stopped/offline/destroyed
      "running"  the container runs; its log is where to look next
      "error"    still booting, and Vast's message reports a failure
      "stuck"    still booting after stuck_s (0 = never)
      "booting"  still booting, within stuck_s
      "unknown"  a status this tool doesn't know: reported, never acted on
    """
    if not inst:
        return "gone", "not listed by Vast"
    status, msg = _status(inst).lower(), _status_msg(inst)
    detail = f"{status}" + (f", Vast says {msg!r}" if msg else "")
    if status in GONE:
        return "gone", detail
    if status == "running":
        return "running", detail
    if status not in BOOTING:
        return "unknown", detail
    if stuck_s and booting_for >= stuck_s:
        return "stuck", detail
    if vast_error(inst):
        return "error", detail
    return "booting", detail


def booting_limit_s(cfg):
    """How long a test rental may sit in a Vast booting state (image pull,
    container start) before it counts as stuck. Our own boot (downloads,
    model load) runs after Vast reports it running, so it isn't counted."""
    return cfg.get("limits", {}).get("stuck_grace_s", 600)


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
    stuck_s = booting_limit_s(cfg)
    out = {"workers": [], "manual": [], "orphans": [], "young": [], "unattributed": [], "worker_rows": workers,
           "why": {}}
    for i in instances:
        key = str(i.get("id"))
        if i.get("id") in live:
            out["workers"].append(i)
        elif key in manual:
            age = now - manual[key].get("created_at", now)
            if ttl and age >= ttl:
                out["orphans"].append(i)
                out["why"][i.get("id")] = f"test rental past manual_ttl_s={ttl}"
            elif boot_state(i, age, stuck_s)[0] == "stuck":
                # Vast never started its container; it bills meanwhile.
                out["orphans"].append(i)
                out["why"][i.get("id")] = f"test rental still {_status(i)} after {age / 60:.0f} min"
            else:
                out["manual"].append(i)
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
                say(f"  {_describe(i, r.get('why', {}).get(i.get('id'), ''))}")
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
STATUS_EVERY_S = 60          # logs ID: repeat a status line this often while nothing new is printed
VAST_ERROR_CONFIRM_S = 60    # logs ID: an error in Vast's status message must last this long
LOG_FINISH_S = 60            # how long a log fetch in flight may take to finish before we move on


def _log_text(out):
    """vastai logs gives the log text, or a dict when Vast has no log to hand back."""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        return str(out.get("msg") or out.get("error") or json.dumps(out))
    return str(out)


class LogStream(threading.Thread):
    """Fetches one instance's container log in the background from the moment
    following starts, whatever state the instance is in: prints each new line
    once, keeps them all, and notes the first boot marker. Only this thread
    calls `v` (give it its own client)."""

    def __init__(self, v, iid, poll):
        super().__init__(daemon=True)
        self.v, self.iid, self.poll = v, iid, poll
        self.lines, self.marker, self.fatal_line, self.note = [], None, "", None
        self._seen, self._stop_ev, self._lock = set(), threading.Event(), threading.Lock()

    def fetch(self):
        try:
            text = _log_text(self.v.logs(self.iid, tail="1000"))
        except Exception as e:   # no log until the container runs; that's normal early on
            note = f"no log from Vast yet ({type(e).__name__}: {str(e)[:120]})"
            if note != self.note:
                self.note = note
                say(note)
            return
        with self._lock:
            for line in text.splitlines():
                if line in self._seen:
                    continue
                self._seen.add(line)
                self.lines.append(line)
                say(line)
                if "ORCH_FATAL:" in line and not self.marker:
                    self.marker, self.fatal_line = "fatal", line
                elif "ORCH_READY model=" in line and not self.marker:
                    self.marker = "ready"

    def run(self):
        while not self._stop_ev.is_set() and not self.marker:
            self.fetch()
            self._stop_ev.wait(self.poll)

    def finish(self, wait=LOG_FINISH_S):
        """Stop, letting a fetch in flight complete, then take one last copy:
        why a boot failed is usually in its last lines."""
        self._stop_ev.set()
        if self.ident is None:   # never started (logs --once fetched by hand)
            return
        if self.is_alive():
            self.join(wait)
        if not self.is_alive():
            self.fetch()


def follow_instance_logs(v, iid, timeout, once=False, poll=None, stuck_s=600, since=None,
                         clock=time.time, log_v=None):
    """Follow one instance's boot. The container log streams in the
    background (LogStream, on log_v) from the start; meanwhile this loop reads
    what Vast reports for the instance (show_instance) and classifies it with
    boot_state, which needs only actual_status, status_msg and how long it
    has been booting.

    Returns (result, boot): result is "ready", "fatal", "gone", "vast_error",
    "stuck", "timeout" or "once"; boot holds the log lines, the ORCH_FATAL
    line if any and the last instance row Vast gave. The log stream has
    finished (with a last fetch) before this returns.

    "vast_error": Vast's status message has reported a failure for
    VAST_ERROR_CONFIRM_S while the container still isn't running. "stuck":
    Vast has reported the instance booting for stuck_s since `since` (when it
    was rented, if known, else when we first saw it booting)."""
    poll = LOGS_POLL_S if poll is None else poll
    stream, t0 = LogStream(log_v or v, iid, poll), clock()
    booting_since = min(since, t0) if since else None
    last_said, last_line, error_since, last_count, inst = t0, None, None, 0, None
    result, verdict = None, ""
    if once:
        stream.fetch()
    else:
        stream.start()
    try:
        while result is None:
            if stream.marker:
                result = stream.marker
                break
            try:
                inst, known = v.show_instance(iid), True
            except Exception as e:   # not knowing is not "gone": say so and ask again
                say(f"could not read instance {iid} from Vast ({type(e).__name__}: {str(e)[:200]}); retrying")
                known = False
            now = clock()
            if known:
                status = _status(inst).lower() if inst else ""
                booting_since = (booting_since or now) if status in BOOTING else None
                state, detail = boot_state(inst, now - booting_since if booting_since else 0, stuck_s)
                line = f"[vast] instance {iid}: {detail}"
                error_since = (error_since or now) if state == "error" else None
                if state == "gone":
                    result, verdict = "gone", (f"instance {iid} is {status or 'not listed'} and printed no "
                                               "ORCH_READY or ORCH_FATAL")
                elif state == "stuck":
                    result, verdict = "stuck", (
                        f"{line}. No container after {(now - booting_since) / 60:.0f} min "
                        f"(limits.stuck_grace_s={stuck_s}); it still bills")
                elif state == "error" and now - error_since >= VAST_ERROR_CONFIRM_S:
                    result, verdict = "vast_error", (f"{line}. Vast reports an error and never started the "
                                                     "container; it still bills")
                elif line != last_line:
                    say(line)
                    last_line, last_said = line, now
                if result:
                    break
            if once:
                result = stream.marker or "once"
                break
            now = clock()
            if len(stream.lines) != last_count:
                last_count, last_said = len(stream.lines), now
            elif now - last_said >= STATUS_EVERY_S:
                say(f"still waiting, {(now - t0) / 60:.0f} min in: "
                    + (line[len("[vast] "):] if known else f"instance {iid}: Vast state unknown")
                    + (f"; {stream.note}" if stream.note and not stream.lines else ""))
                last_said = now
            if now - t0 >= timeout:
                result, verdict = "timeout", (
                    f"no ORCH_READY or ORCH_FATAL from instance {iid} within {timeout:.0f}s "
                    f"({last_line or 'Vast state unknown'}); the full boot log is /workspace/orch/model.log on it")
                break
            time.sleep(poll)
    finally:
        stream.finish()
    if verdict:
        say(verdict)
    return result, {"lines": list(stream.lines), "fatal_line": stream.fatal_line, "instance": inst}


def save_boot_record(iid, result, boot):
    """Keep what a boot left behind (its log and Vast's last word on the
    instance) next to state.json, before anything destroys the instance."""
    d = os.path.join(os.path.dirname(os.path.abspath(STATE_PATH)), "boots")
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{iid}.log")
        with open(path, "w") as f:
            f.write(f"# instance {iid}: {result} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# last instance row from Vast: " + json.dumps(boot.get("instance"), default=str) + "\n")
            f.write("\n".join(boot.get("lines") or []) + "\n")
        say(f"boot log saved to {path}")
        return path
    except OSError as e:
        say(f"warn: could not save the boot log for instance {iid} ({e})")
        return None


def cmd_logs(cfg, args):
    v = vast()
    if getattr(args, "instance_id", None) is not None:
        # The worker gives up at its own boot and download limits; allow for
        # the image pull before boot.sh starts on top of that.
        timeout = args.timeout or boot_limit_s(cfg) + 900
        rec = load_state().get("manual_instances", {}).get(str(args.instance_id), {})
        r, boot = follow_instance_logs(v, args.instance_id, timeout, once=args.once,
                                       stuck_s=booting_limit_s(cfg), since=rec.get("created_at"), log_v=vast())
        if r in ("vast_error", "stuck"):
            say(f"destroy it with `deploy.py sweep --destroy {args.instance_id}`, or let `watch` do it")
        if r in ("fatal", "gone", "vast_error", "stuck", "timeout"):
            save_boot_record(args.instance_id, r, boot)
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


# A boot that ends in one of these failed because of the host it landed on,
# so renting another host is the fix: Vast never started the container, or
# the worker found the host's download speed couldn't meet the budget.
def host_failed(result, boot):
    return result in ("stuck", "vast_error") or (
        result == "fatal" and "too slow on this host" in boot.get("fatal_line", ""))


def _host_key(row):
    """What identifies the physical machine behind an offer or instance."""
    for k in ("machine_id", "host_id"):
        if row.get(k) is not None:
            return f"{k}={row[k]}"
    return f"offer={row.get('id')}"


def cmd_rent_test(cfg, args):
    """Rent one instance outside serverless (no PyWorker) to prove a config,
    recording the id Vast returns so status, sweep, watch and destroy know it.
    Then follow its boot; if it fails because of its host (see host_failed),
    save its log, destroy it, and rent the next cheapest other host, up to
    limits.rent_attempts rentals in all."""
    v = vast()
    say("preflight")
    pins = check(cfg, v)
    ttl = int(cfg["limits"].get("manual_ttl_s", 14400))
    attempts = max(1, int(cfg["limits"].get("rent_attempts", 3)))
    follow = not getattr(args, "no_follow", False)
    opts = docker_options(cfg, pins) + f" -e ORCH_SKIP_PYWORKER=1 -e ORCH_MANUAL_TTL={ttl}"
    t, w = cfg["template"], cfg["workergroup"]
    label = f"orch-test:{cfg['endpoint']['name']}"
    bad_hosts = []
    for attempt in range(1, attempts + 1):
        offers = [o for o in _rows(v.search_offers(query=w["search_params"] + " rented=False",
                                                   order="dph_total", limit=10, storage=t["disk_gb"]),
                                   "search_offers") if _host_key(o) not in bad_hosts]
        if not offers:
            raise CheckFailed("no offer matches workergroup.search_params right now"
                              + (f" apart from hosts that failed: {', '.join(bad_hosts)}" if bad_hosts else ""))
        o = min(offers, key=lambda o: float(o.get("dph_total") or 0))
        say(f"cheapest match: offer {o.get('id')} {o.get('gpu_name', '?')} ${float(o.get('dph_total') or 0):.3f}/hr"
            + (f" (rental {attempt} of up to {attempts})" if attempt > 1 else ""))
        if args.dry_run:
            say("docker options:\n  " + opts + "\ndry run: nothing rented")
            return
        if attempt == 1:
            again = (f" If a host fails to boot it is destroyed and another rented, up to {attempts} in all."
                     if follow and attempts > 1 else "")
            if not confirm(f"rent it? It stops itself after {ttl}s; `deploy.py destroy` or `sweep` removes it."
                           + again, args.yes):
                return
        with StateLock():
            st = load_state()
            res = v.create_instance(o["id"], image=cfg["llama"]["image"], disk=float(t["disk_gb"]), env=opts,
                                    onstart_cmd=onstart_script(), label=label, ssh=True, direct=True,
                                    cancel_unavail=True)
            iid = res.get("new_contract") if isinstance(res, dict) else None
            if not isinstance(iid, int):
                raise ApiError(f"create_instance returned no instance id: {str(res)[:300]}. Check the dashboard "
                               f"for an instance labelled {label!r} and destroy it if present")
            created = time.time()
            st.setdefault("manual_instances", {})[str(iid)] = {"created_at": created, "label": label}
            save_state(st)
            say(f"instance {iid} rented and recorded")
            inst = _wait_for(lambda: next((i for i in _rows(v.show_instances(), "show_instances")
                                           if i.get("id") == iid), None), f"instance {iid}")
            say(f"  {_describe(inst)}")
        if not follow:
            say(f"follow the boot with: deploy.py logs {iid}")
            return
        result, boot = follow_instance_logs(v, iid, boot_limit_s(cfg) + 900, stuck_s=booting_limit_s(cfg),
                                            since=created, log_v=vast())
        if result == "ready":
            say(f"instance {iid} is ready; it stops itself after {ttl}s, or remove it with `deploy.py destroy`")
            return
        save_boot_record(iid, result, boot)
        if not host_failed(result, boot):
            # Not the host's fault (or not known to be): leave it for a look;
            # it stops itself, and watch/sweep clean up after manual_ttl_s.
            raise CheckFailed(f"instance {iid} did not boot ({result}); see the log above. It was left as is: "
                              f"`deploy.py sweep --destroy {iid}` removes it")
        bad_hosts.append(_host_key(o))
        say(f"instance {iid} failed because of its host ({result}); destroying it")
        with StateLock():
            destroy_and_wait(v, [iid])
            st = load_state()
            st.get("manual_instances", {}).pop(str(iid), None)
            save_state(st)
    raise CheckFailed(f"{attempts} rentals failed to boot on their hosts ({', '.join(bad_hosts)}); "
                      "nothing is left running. See the saved boot logs")


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
        actions.append(f"destroy orphan {_describe(i, r['why'].get(i.get('id'), ''))}")
    if targets:
        destroy_and_wait(v, targets)
        for i in targets:
            st["seen_workers"].pop(str(i), None)
            st["manual_instances"].pop(str(i), None)

    # 3. Warm hours: hold min_load up during them so the autoscaler keeps a
    #    worker running, and put it back after. Written on each change of
    #    window, or when Vast reports a value we didn't set, and read back.
    configured = bool(cfg["endpoint"].get("warm_hours"))
    if ep and (configured or "warm_active" in st):
        active = warm_now(cfg, now)
        want = endpoint_limits(cfg, now)
        reported = ep.get("min_load")
        if st.get("warm_active") != active or (reported is not None and not _same(reported, want["min_load"])):
            v.update_endpoint(ep["id"], endpoint_name=cfg["endpoint"]["name"], **want)
            verify_endpoint(v, ep["id"], want)
            st["warm_active"] = active
            actions.append(f"warm hours {'on' if active else 'off'}: endpoint min_load={want['min_load']}")
        if not configured:   # warm_hours removed from the config: reverted above, now forget it
            st.pop("warm_active", None)

    # 4. Whole-account spend, from Vast's own dph_total, against the budget.
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
    p.add_argument("--no-follow", action="store_true",
                   help="rent-test: rent and record the instance, then exit without following its boot")
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

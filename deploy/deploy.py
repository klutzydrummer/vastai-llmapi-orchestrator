#!/usr/bin/env python3
"""Create / update / inspect the Vast Serverless setup from config.toml.

  deploy.py check     preflight only: every remote reference resolves, files
                      exist, disk fits, the image tag exists, offers match
  deploy.py apply     check, then create/update template, endpoint, workergroup
  deploy.py status    endpoint, workergroups and workers
  deploy.py logs      recent endpoint logs from the autoscaler
  deploy.py pause     stop the endpoint (workers go inactive: storage cost only)
  deploy.py resume    reactivate it
  deploy.py sweep     list instances this tool started that no live worker
                      accounts for (orphans), and destroy them (asks first)
  deploy.py destroy   delete workergroup + endpoint, destroy their instances
                      and wait until they are gone (asks first)

Options: --config PATH (default deploy/config.toml), --yes (no prompts),
--dry-run (apply/sweep: show what would happen, change nothing),
--min-age SECONDS (sweep: leave younger instances alone, default 900).
Needs VAST_API_KEY in the environment.
"""

import argparse
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
STATE_PATH = os.path.join(HERE, "state.json")
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


def docker_options(cfg, pins):
    """The template's 'Docker options' string: port + env, all pinned."""
    m, l, b = cfg["model"], cfg["llama"], cfg["boot"]
    env = {
        # Ownership marker: sweep/destroy only ever touch instances carrying it.
        "ORCH_DEPLOYMENT": cfg["endpoint"]["name"],
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
        "LOAD_TIMEOUT": b["load_timeout_s"],
        "PYWORKER_REF": b["pyworker_ref"],
        "VAST_SDK_VERSION": b["vast_sdk_version"],
    }
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


def image_exists(image):
    """True/False for ghcr.io images; None when the registry can't be checked."""
    if not image.startswith("ghcr.io/"):
        return None
    name, _, tag = image[len("ghcr.io/"):].rpartition(":")
    try:
        tok = requests.get(f"https://ghcr.io/token?scope=repository:{name}:pull", timeout=20).json()["token"]
        r = requests.head(f"https://ghcr.io/v2/{name}/manifests/{tag}", timeout=20, headers={
            "Authorization": f"Bearer {tok}",
            "Accept": ", ".join([
                "application/vnd.oci.image.index.v1+json",
                "application/vnd.docker.distribution.manifest.list.v2+json",
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.v2+json"])})
        return r.status_code == 200
    except Exception:
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

    weights_gb = sum(sizes) / 1024**3
    if sizes:
        need_disk = weights_gb + 12   # image layers, pyworker venv, logs, headroom
        say(f"  info  weights {weights_gb:.1f} GiB; disk {cfg['template']['disk_gb']} GB (need ~{need_disk:.0f})")
        if cfg["template"]["disk_gb"] < need_disk:
            problems.append(f"template.disk_gb={cfg['template']['disk_gb']} is too small; use >= {need_disk:.0f}")
            say("  FAIL  disk too small")
        if weights_gb + 2.5 > l["min_vram_gb"]:
            say(f"  warn  weights {weights_gb:.1f} GiB leave little room for KV cache under "
                f"min_vram_gb={l['min_vram_gb']}")

    say("worker code and image")
    pins["orch_ref"] = step(f"github {b['orch_repo']}@{b['orch_ref']}", lambda: gh_resolve(b["orch_repo"], b["orch_ref"]))
    if pins["orch_ref"]:
        for f in ("worker/onstart.sh", "worker/boot.sh", "worker/fetch_model.py", "worker/smoke_test.py"):
            def _has(f=f):
                if not gh_raw_exists(b["orch_repo"], pins["orch_ref"], f):
                    raise CheckFailed("not found at that commit (push it first)")
            step(f"  {f}", _has)
    def _pyworker():
        if not gh_raw_exists("vast-ai/pyworker", b["pyworker_ref"], "workers/llama/worker.py"):
            raise CheckFailed("workers/llama/worker.py missing at that ref")
    step(f"vast-ai/pyworker@{b['pyworker_ref'][:12]}", _pyworker)
    ok = image_exists(l["image"])
    if ok is False:
        say(f"  FAIL  image {l['image']} not found")
        problems.append(f"image {l['image']} not found")
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
                    f"cuda {o.get('cuda_max_good', '?')}  rel {(o.get('reliability') or o.get('reliability2') or 0):.3f}")
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


class StateLock:
    """Stops two apply/destroy/sweep runs from racing each other into duplicates."""

    def __enter__(self):
        self.path = STATE_PATH + ".lock"
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise CheckFailed(f"{self.path} exists: another apply/destroy/sweep is running, or one crashed. "
                              "If none is running, delete that file and retry")
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


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
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


# ── instances ────────────────────────────────────────────────────────────────
def _env(inst):
    env = inst.get("extra_env") or {}
    if isinstance(env, list):
        env = {p[0]: p[1] for p in env if isinstance(p, (list, tuple)) and len(p) >= 2}
    return env if isinstance(env, dict) else {}


def owned(inst, cfg, st):
    """True for instances this deployment started: serverless workers and
    manual test rentals made with the printed docker options."""
    env = _env(inst)
    if env.get("ORCH_DEPLOYMENT") == cfg["endpoint"]["name"]:
        return True
    # Started before the ORCH_DEPLOYMENT marker existed.
    if env.get("ORCH_REF") and env.get("SERVED_MODEL_NAME") == cfg["model"]["served_name"]:
        return True
    tpl = inst.get("template_hash_id")
    return bool(tpl) and tpl in st.get("template_hashes", [])


def _age(inst):
    if isinstance(inst.get("duration"), (int, float)):
        return inst["duration"]
    if isinstance(inst.get("start_date"), (int, float)):
        return time.time() - inst["start_date"]
    return float("inf")


def _dph(inst):
    try:
        return float(inst.get("dph_total") or 0)
    except (TypeError, ValueError):
        return 0.0


def _describe(inst):
    env = _env(inst)
    kind = "manual test rental" if env.get("ORCH_SKIP_PYWORKER") == "1" else "worker"
    age = _age(inst)
    age_s = "?" if age == float("inf") else f"{age / 60:.0f} min"
    return (f"instance {inst.get('id')}: {kind}, {inst.get('actual_status') or inst.get('cur_state') or '?'}, "
            f"{inst.get('gpu_name', '?')}, ${_dph(inst):.3f}/hr, up {age_s}")


def live_worker_ids(v, endpoint_id):
    """Instance ids the autoscaler counts as this endpoint's workers. Raises
    when it can't say, so nothing is destroyed on a guess."""
    out = v.get_endpoint_workers(endpoint_id)
    if isinstance(out, dict) and out.get("error_msg"):
        raise ApiError(f"get_endpoint_workers: {out['error_msg']}")
    return {r.get("id") for r in _rows(out, "get_endpoint_workers") if isinstance(r, dict)}


def survey(v, cfg, st, ep, min_age):
    """Sort this deployment's instances into workers, orphans and too-new-to-judge."""
    instances = _rows(v.show_instances(), "show_instances")
    mine = [i for i in instances if owned(i, cfg, st)]
    live = live_worker_ids(v, ep["id"]) if ep else set()
    out = {"workers": [], "orphans": [], "young": [],
           "others": [i for i in instances if not owned(i, cfg, st)]}
    for i in mine:
        if i.get("id") in live:
            out["workers"].append(i)
        elif _age(i) >= min_age:
            out["orphans"].append(i)
        else:
            out["young"].append(i)
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

    ep_fields = {k: e[k] for k in ("max_workers", "cold_workers", "min_load", "target_util",
                                   "cold_mult", "inactivity_timeout") if k in e}
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

    search = (w["search_params"] + " verified=True rentable=True rented=False").strip()
    if groups:
        g = groups[0]
        v.update_workergroup(g["id"], template_hash=tpl_hash, search_params=search,
                             gpu_ram=w.get("gpu_ram"), endpoint_id=ep_id)
        say(f"workergroup {g['id']} now uses template {tpl_hash}")
        say("existing workers keep running the old template until replaced; "
            "use the dashboard's update-workers (or destroy + apply) to roll them")
        st["workergroup_id"] = g["id"]
    else:
        st["workergroup_pending"] = ep_id
        save_state(st)
        blob = {"client_id": "me", "template_hash": tpl_hash, "search_params": search,
                "gpu_ram": w.get("gpu_ram"), "endpoint_id": ep_id, "endpoint_name": e["name"],
                "test_workers": w["test_workers"], "cold_workers": e["cold_workers"],
                "autoscaler_instance": "prod"}
        r = v.client.post("/autojobs/", json_data=blob)
        r.raise_for_status()
        made = _wait_for(lambda: find_workergroups(v, ep_id), f"workergroup for endpoint {ep_id}")
        st.pop("workergroup_pending", None)
        st["workergroup_id"] = made[0].get("id")
        say(f"workergroup {st['workergroup_id']} created")
    save_state(st)
    say("\ndone. watch it with: deploy.py status   (first worker: download + load + benchmark)")


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
        r = survey(v, cfg, st, ep, args.min_age)
    except CheckFailed as e:
        say(f"(could not check instances: {e})")
        return
    _report(r)


def _report(r):
    for label, key in (("serverless workers", "workers"), ("ORPHANS (no live worker accounts for them)",
                                                           "orphans"), ("too new to judge", "young")):
        if r[key]:
            say(f"{label}:")
            for i in r[key]:
                say(f"  {_describe(i)}")
    mine = r["workers"] + r["orphans"] + r["young"]
    say(f"this deployment: {len(mine)} instance(s), ${sum(_dph(i) for i in mine):.3f}/hr")
    if r["others"]:
        say(f"other instances on the account (left alone): {len(r['others'])}, "
            f"${sum(_dph(i) for i in r['others']):.3f}/hr")
    if r["orphans"]:
        say("run `deploy.py sweep` to destroy the orphans")


def cmd_sweep(cfg, args):
    v = vast()
    with StateLock():
        st = load_state()
        ep = find_endpoint(v, cfg["endpoint"]["name"], st, allow_state_id=True)
        r = survey(v, cfg, st, ep, args.min_age)
        _report(r)
        if not r["orphans"]:
            say("no orphans")
            return
        if args.dry_run:
            say("dry run: nothing destroyed")
            return
        cost = sum(_dph(i) for i in r["orphans"])
        if not confirm(f"destroy {len(r['orphans'])} orphaned instance(s) (${cost:.3f}/hr)?", args.yes):
            return
        destroy_and_wait(v, [i["id"] for i in r["orphans"]])


def cmd_logs(cfg, args):
    v = vast()
    ep = find_endpoint(v, cfg["endpoint"]["name"], load_state(), allow_state_id=True)
    if not ep:
        raise SystemExit("endpoint not found")
    out = v.get_endpt_logs(ep["id"], level=1, tail=200)
    say(json.dumps(out, indent=2) if not isinstance(out, str) else out)


def _set_state(cfg, state):
    v = vast()
    ep = find_endpoint(v, cfg["endpoint"]["name"], load_state(), allow_state_id=True)
    if not ep:
        raise SystemExit("endpoint not found")
    v.update_endpoint(ep["id"], endpoint_name=cfg["endpoint"]["name"], endpoint_state=state)
    say(f"endpoint {ep['id']} -> {state}")


def cmd_destroy(cfg, args):
    v = vast()
    with StateLock():
        st = load_state()
        ep = find_endpoint(v, cfg["endpoint"]["name"], st, allow_state_id=True)
        groups = find_workergroups(v, ep["id"]) if ep else []
        mine = [i for i in _rows(v.show_instances(), "show_instances") if owned(i, cfg, st)]
        if not ep and not groups and not mine:
            say("no endpoint and no instances from this deployment; nothing to do")
            return
        for i in mine:
            say(f"  {_describe(i)}")
        what = f"endpoint {ep['id']} and {len(groups)} workergroup(s)" if ep else "no endpoint (already gone)"
        if not confirm(f"delete {what}, and destroy {len(mine)} instance(s) listed above "
                       "(manual test rentals included)? Cached weights go away.", args.yes):
            return
        failed = []
        for kind, obj_id, fn in ([("workergroup", g["id"], v.delete_workergroup) for g in groups]
                                 + ([("endpoint", ep["id"], v.delete_endpoint)] if ep else [])):
            try:
                fn(obj_id)
                say(f"{kind} {obj_id} deleted")
            except Exception as e:   # keep going: the instances are what bill
                say(f"{kind} {obj_id}: delete failed ({e})")
                failed.append(f"{kind} {obj_id}")
        # Deleting a workergroup doesn't destroy its instances (per the Vast
        # SDK), so destroy them here and confirm. A second pass catches any
        # the autoscaler launched while the groups were being deleted.
        done = set()
        for _ in range(2):
            ids = {i["id"] for i in _rows(v.show_instances(), "show_instances") if owned(i, cfg, st)} - done
            if not ids:
                break
            destroy_and_wait(v, ids)
            done |= ids
        if failed:
            raise CheckFailed(f"instances are gone, but deleting {', '.join(failed)} failed; the autoscaler "
                              "may start new workers. Re-run destroy or delete them in the dashboard")
        for k in ("endpoint_id", "endpoint_name", "workergroup_id", "endpoint_pending", "workergroup_pending"):
            st.pop(k, None)
        save_state(st)
        say("destroyed; no instances from this deployment remain")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "apply", "status", "logs", "pause", "resume", "sweep", "destroy"])
    p.add_argument("--config", default=os.path.join(HERE, "config.toml"))
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--min-age", type=float, default=900,
                   help="sweep/status: instances younger than this many seconds are never called orphans")
    args = p.parse_args()
    cfg = load_config(args.config)
    try:
        {"check": cmd_check, "apply": cmd_apply, "status": cmd_status, "logs": cmd_logs,
         "pause": lambda c, a: _set_state(c, "stopped"),
         "resume": lambda c, a: _set_state(c, "active"),
         "sweep": cmd_sweep, "destroy": cmd_destroy}[args.command](cfg, args)
    except CheckFailed as e:
        say(f"\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create / update / inspect the Vast Serverless setup from config.toml.

  deploy.py check     preflight only: every remote reference resolves, files
                      exist, disk fits, the image tag exists, offers match
  deploy.py apply     check, then create/update template, endpoint, workergroup
  deploy.py status    endpoint, workergroups and workers
  deploy.py logs      recent endpoint logs from the autoscaler
  deploy.py pause     stop the endpoint (workers go inactive: storage cost only)
  deploy.py resume    reactivate it
  deploy.py destroy   delete workergroup + endpoint (asks first)

Options: --config PATH (default deploy/config.toml), --yes (no prompts),
--dry-run (apply: print what would be sent, change nothing).
Needs VAST_API_KEY in the environment.
"""

import argparse
import json
import os
import sys
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
def load_state():
    try:
        return json.load(open(STATE_PATH))
    except Exception:
        return {}


def save_state(st):
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=2)


def find_endpoint(v, name):
    rows = v.show_endpoints()
    for row in rows if isinstance(rows, list) else []:
        if row.get("endpoint_name") == name:
            return row
    return None


def find_workergroups(v, endpoint_id):
    rows = v.show_workergroups()
    return [w for w in (rows if isinstance(rows, list) else []) if w.get("endpoint_id") == endpoint_id]


def confirm(prompt, yes):
    if yes:
        return True
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_check(cfg, args):
    say("preflight")
    check(cfg)
    say("\nall checks passed")


def cmd_apply(cfg, args):
    v = vast()
    say("preflight")
    pins = check(cfg, v)
    e, w, t = cfg["endpoint"], cfg["workergroup"], cfg["template"]
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

    st = load_state()
    tpl = v.create_template(
        name=t["name"], image=cfg["llama"]["image"], env=opts, onstart_cmd=onstart_script(),
        ssh=True, direct=True, disk_space=float(t["disk_gb"]),
        search_params=w["search_params"], desc="vastai-llmapi-orchestrator worker", public=False)
    tpl_obj = tpl.get("template") if isinstance(tpl, dict) else None
    tpl_hash = (tpl_obj or {}).get("hash_id") or (tpl_obj or {}).get("hash") or tpl.get("hash_id")
    if not tpl_hash:
        raise SystemExit(f"template creation returned no hash: {tpl}")
    say(f"template {tpl_hash} ({t['name']})")

    ep_fields = {k: e[k] for k in ("max_workers", "cold_workers", "min_load", "target_util",
                                   "cold_mult", "inactivity_timeout") if k in e}
    ep = find_endpoint(v, e["name"])
    if ep:
        v.update_endpoint(ep["id"], endpoint_name=e["name"], **ep_fields)
        say(f"endpoint {ep['id']} updated")
        ep_id = ep["id"]
    else:
        v.create_endpoint(endpoint_name=e["name"], **ep_fields)
        ep = find_endpoint(v, e["name"])
        if not ep:
            raise SystemExit("endpoint was created but can't be found")
        ep_id = ep["id"]
        say(f"endpoint {ep_id} created")

    search = (w["search_params"] + " verified=True rentable=True rented=False").strip()
    groups = find_workergroups(v, ep_id)
    if groups:
        for g in groups:
            v.update_workergroup(g["id"], template_hash=tpl_hash, search_params=w["search_params"],
                                 gpu_ram=w.get("gpu_ram"), endpoint_id=ep_id)
            say(f"workergroup {g['id']} now uses template {tpl_hash}")
        say("existing workers keep running the old template until replaced; "
            "use the dashboard's update-workers (or destroy + apply) to roll them")
    else:
        blob = {"client_id": "me", "template_hash": tpl_hash, "search_params": search,
                "gpu_ram": w.get("gpu_ram"), "endpoint_id": ep_id, "endpoint_name": e["name"],
                "test_workers": w.get("test_workers", 1), "cold_workers": e.get("cold_workers"),
                "autoscaler_instance": "prod"}
        r = v.client.post("/autojobs/", json_data=blob)
        r.raise_for_status()
        say(f"workergroup created: {r.json()}")

    st.update({"template_hash": tpl_hash, "endpoint_id": ep_id, "endpoint_name": e["name"], "pins": pins})
    save_state(st)
    say("\ndone. watch it with: deploy.py status   (first worker: download + load + benchmark)")


def cmd_status(cfg, args):
    v = vast()
    ep = find_endpoint(v, cfg["endpoint"]["name"])
    if not ep:
        say("endpoint not found")
        return
    say(json.dumps({k: ep.get(k) for k in ("id", "endpoint_name", "endpoint_state", "max_workers",
                                           "cold_workers", "min_load", "inactivity_timeout")}, indent=2))
    for g in find_workergroups(v, ep["id"]):
        say(json.dumps({k: g.get(k) for k in ("id", "template_hash", "search_params", "gpu_ram",
                                              "test_workers", "cold_workers")}, indent=2))
    try:
        import asyncio
        from vastai import CoroutineServerless

        async def workers():
            async with CoroutineServerless(api_key=os.environ["VAST_API_KEY"]) as c:
                e = await c.get_endpoint(cfg["endpoint"]["name"])
                return await c.get_endpoint_workers(e)
        for wk in asyncio.run(workers()):
            say(f"worker {wk.id}: {wk.status}  load {wk.cur_load:.0f}  perf {wk.perf:.0f}  reqs {wk.reqs_working}")
    except Exception as e:
        say(f"(could not list workers: {e})")


def cmd_logs(cfg, args):
    v = vast()
    ep = find_endpoint(v, cfg["endpoint"]["name"])
    if not ep:
        raise SystemExit("endpoint not found")
    out = v.get_endpt_logs(ep["id"], level=1, tail=200)
    say(json.dumps(out, indent=2) if not isinstance(out, str) else out)


def _set_state(cfg, state):
    v = vast()
    ep = find_endpoint(v, cfg["endpoint"]["name"])
    if not ep:
        raise SystemExit("endpoint not found")
    v.update_endpoint(ep["id"], endpoint_name=cfg["endpoint"]["name"], endpoint_state=state)
    say(f"endpoint {ep['id']} -> {state}")


def cmd_destroy(cfg, args):
    v = vast()
    ep = find_endpoint(v, cfg["endpoint"]["name"])
    if not ep:
        say("endpoint not found; nothing to do")
        return
    groups = find_workergroups(v, ep["id"])
    if not confirm(f"delete endpoint {ep['id']} and {len(groups)} workergroup(s)? "
                   "Their instances and cached weights go away.", args.yes):
        return
    for g in groups:
        v.delete_workergroup(g["id"])
        say(f"workergroup {g['id']} deleted")
    v.delete_endpoint(ep["id"])
    say(f"endpoint {ep['id']} deleted")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "apply", "status", "logs", "pause", "resume", "destroy"])
    p.add_argument("--config", default=os.path.join(HERE, "config.toml"))
    p.add_argument("--yes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    try:
        {"check": cmd_check, "apply": cmd_apply, "status": cmd_status, "logs": cmd_logs,
         "pause": lambda c, a: _set_state(c, "stopped"),
         "resume": lambda c, a: _set_state(c, "active"),
         "destroy": cmd_destroy}[args.command](cfg, args)
    except CheckFailed as e:
        say(f"\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

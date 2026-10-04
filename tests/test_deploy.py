#!/usr/bin/env python3
"""deploy.py tests against an in-memory fake of the Vast API: no duplicate
endpoints or workergroups, no destroying on a guess, orphans found and
confirmed gone, spend limits enforced. Run: python3 tests/test_deploy.py"""

import builtins
import copy
import os
import re
import sys
import tempfile
import time
import tomllib
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
import deploy  # noqa: E402

deploy.POLL_S = 0
deploy.CONFIRM_WAIT_S = 0
deploy.DESTROY_WAIT_S = 0
deploy.LOCK_WAIT_S = 0
deploy.say = lambda msg="": None

with open(os.path.join(os.path.dirname(__file__), "..", "deploy", "config.example.toml"), "rb") as f:
    BASE_CFG = tomllib.load(f)
NAME = BASE_CFG["endpoint"]["name"]
PINS = {"model_revision": "a" * 40, "orch_ref": "b" * 40, "mmproj_revision": "c" * 40, "embed_revision": "d" * 40}


class Resp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeVast:
    """In-memory Vast: endpoints and workergroups keep the fields they were
    given (so read-back verification has something to compare), workers are
    the autoscaler's list, instances are the account's instances."""

    def __init__(self):
        self.endpoints, self.groups, self.instances = [], [], []
        self.workers = {}          # instance id -> autoscaler status
        self.calls = []
        self.fail = set()          # method names that answer with an error dict
        self.sticky = set()        # instance ids that survive destroy
        self.ignore = set()        # endpoint fields the "server" silently ignores
        self.next_id = 100
        self.client = types.SimpleNamespace(post=self._post)

    def _id(self):
        self.next_id += 1
        return self.next_id

    def _err(self, name):
        return {"error": f"{name} exploded"} if name in self.fail else None

    def show_endpoints(self):
        return self._err("show_endpoints") or copy.deepcopy(self.endpoints)

    def show_workergroups(self):
        return self._err("show_workergroups") or copy.deepcopy(self.groups)

    def show_instances(self):
        return self._err("show_instances") or copy.deepcopy(self.instances)

    def get_endpoint_workers(self, id):
        if "get_endpoint_workers" in self.fail:
            return {"error_msg": "not ready"}
        return [{"id": i, "status": s} for i, s in self.workers.items()]

    def create_template(self, **kw):
        self.calls.append(("create_template",))
        return {"success": True, "template": {"hash_id": f"tpl{self._id()}"}}

    def _apply_fields(self, row, kw):
        for k, val in kw.items():
            if val is not None and k not in self.ignore:
                row[k] = val

    def create_endpoint(self, **kw):
        self.calls.append(("create_endpoint", kw))
        eid = self._id()
        row = {"id": eid, "endpoint_state": "active"}
        self._apply_fields(row, kw)
        self.endpoints.append(row)
        return {"success": True, "result": eid}

    def update_endpoint(self, id, **kw):
        self.calls.append(("update_endpoint", id, kw))
        for row in self.endpoints:
            if row["id"] == id:
                self._apply_fields(row, kw)

    def update_workergroup(self, id, **kw):
        self.calls.append(("update_workergroup", id, kw))
        for g in self.groups:
            if g["id"] == id and kw.get("template_hash"):
                g["template_hash"] = kw["template_hash"]

    def _post(self, path, json_data=None):
        self.calls.append(("create_workergroup", json_data))
        gid = self._id()
        self.groups.append({"id": gid, **{k: json_data[k] for k in (
            "endpoint_id", "template_hash", "test_workers", "cold_workers", "max_workers", "min_load")}})
        return Resp({"success": True, "id": gid})

    def delete_workergroup(self, id):
        self.calls.append(("delete_workergroup", id))
        self.groups = [g for g in self.groups if g["id"] != id]

    def delete_endpoint(self, id):
        """Like Vast: deletes the endpoint's workergroups and destroys its workers."""
        self.calls.append(("delete_endpoint", id))
        self.endpoints = [e for e in self.endpoints if e["id"] != id]
        self.groups = [g for g in self.groups if g["endpoint_id"] != id]
        gone = [i for i in self.workers if i not in self.sticky]
        failed = [i for i in self.workers if i in self.sticky]
        self.instances = [i for i in self.instances if i["id"] not in gone]
        self.workers = {}
        return {"success": True, "deleted_workers": gone, "failed_workers": failed}

    def destroy_instance(self, id):
        self.calls.append(("destroy_instance", id))
        if id not in self.sticky:
            self.instances = [i for i in self.instances if i["id"] != id]

    def search_offers(self, **kw):
        return [{"id": 777, "gpu_name": "RTX 4090", "dph_total": 0.31}]

    def create_instance(self, offer, **kw):
        self.calls.append(("create_instance", offer, kw))
        iid = self._id()
        self.instances.append(inst(iid, label=kw.get("label")))
        return {"success": True, "new_contract": iid}

    def made(self, name):
        return [c for c in self.calls if c[0] == name]


def inst(id, status="running", dph=0.3, label=None):
    return {"id": id, "actual_status": status, "dph_total": dph, "gpu_name": "RTX 4090", "label": label}


PASSED, FAILED = [], []


def case(fn):
    with tempfile.TemporaryDirectory() as d:
        deploy.STATE_PATH = os.path.join(d, "state.json")
        try:
            fn()
            PASSED.append(fn.__name__)
            print(f"PASS  {fn.__doc__}")
        except Exception as e:
            FAILED.append(fn.__name__)
            print(f"FAIL  {fn.__doc__}: {type(e).__name__}: {e}")
    return fn


def args(**kw):
    a = {"yes": True, "dry_run": False, "min_age": 900, "destroy": []}
    a.update(kw)
    return types.SimpleNamespace(**a)


def raises(fn, exc=deploy.CheckFailed, text=""):
    try:
        fn()
    except exc as e:
        assert text in str(e), f"wrong message: {e}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def apply(v, cfg=None):
    deploy._apply(v, cfg or BASE_CFG, PINS, "-p 3000:3000")


def with_lock(fn):
    with deploy.StateLock():
        fn()


# ── apply: no duplicates ──────────────────────────────────────────────────────
@case
def fresh_apply_then_reapply():
    """apply creates one endpoint + workergroup; a second apply updates them"""
    v = FakeVast()
    apply(v)
    apply(v)
    assert len(v.made("create_endpoint")) == 1 and len(v.made("create_workergroup")) == 1, v.calls
    assert len(v.endpoints) == 1 and len(v.groups) == 1
    assert len(v.made("update_endpoint")) == 1 and len(v.made("update_workergroup")) == 1
    st = deploy.load_state()
    assert st["endpoint_id"] == v.endpoints[0]["id"] and st["workergroup_id"] == v.groups[0]["id"], st
    assert "endpoint_pending" not in st and "workergroup_pending" not in st, st


@case
def workergroup_calls_match_the_sdk():
    """update passes the raw search (the SDK adds defaults); create sends every limit"""
    v = FakeVast()
    apply(v)
    blob = v.made("create_workergroup")[0][1]
    e, w = BASE_CFG["endpoint"], BASE_CFG["workergroup"]
    for k, want in (("max_workers", e["max_workers"]), ("min_load", e["min_load"]),
                    ("cold_workers", e["cold_workers"]), ("test_workers", w["test_workers"])):
        assert blob[k] == want, (k, blob)
    assert blob["search_params"].endswith("verified=True rentable=True rented=False")
    assert deploy.load_state()["workergroup_id"] == v.groups[0]["id"]
    apply(v)
    upd = v.made("update_workergroup")[0][2]
    assert upd["search_params"] == w["search_params"], upd


@case
def endpoint_list_error_creates_nothing():
    """an error from show_endpoints stops apply before anything is created"""
    v = FakeVast()
    v.fail.add("show_endpoints")
    raises(lambda: apply(v), deploy.ApiError, "instead of a list")
    assert not v.calls, v.calls


@case
def workergroup_list_error_creates_nothing():
    """an error from show_workergroups stops apply before anything is created"""
    v = FakeVast()
    apply(v)
    v.calls.clear()
    v.fail.add("show_workergroups")
    raises(lambda: apply(v), deploy.ApiError)
    assert not v.calls, v.calls


@case
def duplicate_endpoints_refused():
    """two endpoints with the configured name: refuse, change nothing"""
    v = FakeVast()
    v.endpoints = [{"id": 1, "endpoint_name": NAME}, {"id": 2, "endpoint_name": NAME}]
    raises(lambda: apply(v), deploy.ApiError, "2 endpoints")
    assert not v.calls


@case
def duplicate_workergroups_refused():
    """two workergroups on the endpoint: refuse, change nothing"""
    v = FakeVast()
    v.endpoints = [{"id": 1, "endpoint_name": NAME}]
    v.groups = [{"id": 5, "endpoint_id": 1}, {"id": 6, "endpoint_id": 1}]
    raises(lambda: apply(v), deploy.ApiError, "2 workergroups")
    assert not v.calls


@case
def renamed_endpoint_refused():
    """the endpoint state.json recorded was renamed: refuse instead of creating a second"""
    v = FakeVast()
    apply(v)
    v.endpoints[0]["endpoint_name"] = "renamed"
    v.calls.clear()
    raises(lambda: apply(v), deploy.ApiError, "now named")
    assert not v.made("create_endpoint")


@case
def unconfirmed_endpoint_blocks_rerun():
    """an endpoint that was created but never showed up blocks the next apply"""
    v = FakeVast()
    v.show_endpoints = lambda: []          # created, but the list never shows it
    raises(lambda: apply(v), deploy.ApiError, "hasn't shown up")
    assert deploy.load_state().get("endpoint_pending") == NAME
    v2 = FakeVast()
    raises(lambda: apply(v2), deploy.ApiError, "never saw it appear")
    assert not v2.calls


@case
def unconfirmed_workergroup_blocks_rerun():
    """a workergroup that was created but never showed up blocks the next apply"""
    v = FakeVast()
    v.show_workergroups = lambda: []
    raises(lambda: apply(v), deploy.ApiError, "hasn't shown up")
    v.calls.clear()
    raises(lambda: apply(v), deploy.ApiError, "never saw it appear")
    assert not v.made("create_workergroup")


@case
def concurrent_runs_refused():
    """a second apply while one holds the lock is refused"""
    v = FakeVast()
    with deploy.StateLock():
        raises(lambda: with_lock(lambda: apply(v)), deploy.CheckFailed, "holds")
    assert not v.calls
    with_lock(lambda: apply(v))      # lock released afterwards


@case
def lock_freed_when_holder_dies():
    """a run killed while holding the lock (e.g. a stopped container) doesn't block the next"""
    import subprocess
    code = ("import fcntl,os,sys; f=open(sys.argv[1],'a+'); "
            "fcntl.flock(f,fcntl.LOCK_EX); os.kill(os.getpid(),9)")
    subprocess.run([sys.executable, "-c", code, deploy.STATE_PATH + ".lock"])
    with_lock(lambda: apply(FakeVast()))


@case
def corrupt_state_refused():
    """a corrupt state.json is an error, not an empty slate"""
    with open(deploy.STATE_PATH, "w") as f:
        f.write("{not json")
    raises(lambda: apply(FakeVast()), deploy.CheckFailed, "unreadable")


# ── spend limits ──────────────────────────────────────────────────────────────
def limited(**changes):
    cfg = copy.deepcopy(BASE_CFG)
    for path, val in changes.items():
        section, key = path.split("__")
        if val is None:
            cfg[section].pop(key, None)
        else:
            cfg[section][key] = val
    return cfg


@case
def example_config_within_limits():
    """the example config passes and reports its worst case"""
    worst = deploy.check_limits(BASE_CFG)
    assert abs(worst - 1.20) < 1e-9, worst


@case
def missing_limits_refused():
    """missing max_workers, cold_workers, test_workers or [limits] is refused"""
    for path in ("endpoint__max_workers", "endpoint__cold_workers", "workergroup__test_workers"):
        raises(lambda: deploy.check_limits(limited(**{path: None})), text=path.split("__")[1])
    cfg = copy.deepcopy(BASE_CFG)
    del cfg["limits"]
    raises(lambda: deploy.check_limits(cfg), text="max_hourly_usd")


@case
def price_ceiling_required():
    """search_params without a dph_total ceiling is refused"""
    sp = re.sub(r"dph_total<=[0-9.]+", "dph_total>0.01", BASE_CFG["workergroup"]["search_params"])
    assert sp != BASE_CFG["workergroup"]["search_params"]
    raises(lambda: deploy.check_limits(limited(workergroup__search_params=sp)), text="price ceiling")
    assert deploy.price_ceiling("gpu_ram>=22 dph_total < 0.5 dph_total<=0.3") == 0.3


@case
def worst_case_over_budget_refused():
    """max_workers x price above max_hourly_usd, or above max_workers_cap, is refused"""
    raises(lambda: deploy.check_limits(limited(limits__max_hourly_usd=0.5)), text="worst case")
    raises(lambda: deploy.check_limits(limited(endpoint__max_workers=3)), text="max_workers_cap")


def deployed():
    v = FakeVast()
    apply(v)
    v.calls.clear()
    return v


def seen(v, *ids, ago=3600):
    """Make state.json record ids as workers first seen `ago` seconds back."""
    st = deploy.load_state()
    for i in ids:
        st.setdefault("seen_workers", {})[str(i)] = time.time() - ago
    deploy.save_state(st)


# ── verify after write ────────────────────────────────────────────────────────
@case
def ignored_setting_is_caught():
    """if Vast reports a different max_workers than asked, apply fails instead of trusting it"""
    v = FakeVast()
    real = v.create_endpoint

    def create(**kw):
        res = real(**kw)
        v.endpoints[-1]["max_workers"] = 20      # server kept its default
        return res
    v.create_endpoint = create
    v.ignore.add("max_workers")
    raises(lambda: apply(v), deploy.ApiError, "max_workers: asked 1, Vast reports 20")


@case
def unreported_fields_are_named_not_assumed():
    """a field Vast doesn't report back is listed as unconfirmed"""
    msgs = []
    deploy.say = msgs.append
    try:
        missing = deploy._verify({"id": 1, "max_workers": 1}, {"max_workers": 1, "test_workers": 1}, "wg 1")
    finally:
        deploy.say = lambda msg="": None
    assert missing == ["test_workers"] and any("can't confirm" in m for m in msgs), msgs


@case
def pause_sends_every_limit_and_verifies():
    """pause sends the configured limits with the state and reads it back"""
    v = deployed()
    deploy.vast = lambda: v
    deploy._set_state(BASE_CFG, "stopped")
    kw = v.made("update_endpoint")[-1][2]
    assert kw["endpoint_state"] == "stopped" and kw["max_workers"] == BASE_CFG["endpoint"]["max_workers"], kw
    assert v.endpoints[0]["endpoint_state"] == "stopped"
    v.ignore.add("endpoint_state")
    raises(lambda: deploy._set_state(BASE_CFG, "active"), deploy.ApiError, "endpoint_state")


# ── sweep: facts only ────────────────────────────────────────────────────────
@case
def sweep_destroys_only_recorded_orphans():
    """sweep destroys former workers it recorded, never unrecorded instances"""
    v = deployed()
    v.instances = [inst(1), inst(2), inst(3), inst(4)]
    v.workers = {1: "IDLE"}
    seen(v, 1, 2)
    seen(v, 3, ago=60)                 # recorded only a minute ago
    deploy.vast = lambda: v
    deploy.cmd_sweep(BASE_CFG, args())
    assert [c[1] for c in v.made("destroy_instance")] == [2], v.calls
    assert sorted(i["id"] for i in v.instances) == [1, 3, 4]


@case
def sweep_records_workers_it_sees():
    """sweep records every worker id the autoscaler reports"""
    v = deployed()
    v.instances = [inst(5)]
    v.workers = {5: "LOADING"}
    deploy.vast = lambda: v
    deploy.cmd_sweep(BASE_CFG, args())
    assert "5" in deploy.load_state()["seen_workers"] and not v.made("destroy_instance")


@case
def sweep_destroy_named_unrecorded():
    """an unrecorded instance is destroyed only when named, and a live worker never"""
    v = deployed()
    v.instances = [inst(1), inst(9)]
    v.workers = {1: "IDLE"}
    deploy.vast = lambda: v
    deploy.cmd_sweep(BASE_CFG, args())
    assert not v.made("destroy_instance")
    deploy.cmd_sweep(BASE_CFG, args(destroy=[9]))
    assert [c[1] for c in v.made("destroy_instance")] == [9]
    raises(lambda: deploy.cmd_sweep(BASE_CFG, args(destroy=[1])), text="live worker")


@case
def sweep_dry_run_destroys_nothing():
    """sweep --dry-run only reports"""
    v = deployed()
    v.instances = [inst(2)]
    seen(v, 2)
    deploy.vast = lambda: v
    deploy.cmd_sweep(BASE_CFG, args(dry_run=True))
    assert not v.made("destroy_instance")


@case
def sweep_refuses_without_worker_list():
    """if the autoscaler can't list workers, sweep destroys nothing"""
    v = deployed()
    v.instances = [inst(2)]
    seen(v, 2)
    v.fail.add("get_endpoint_workers")
    deploy.vast = lambda: v
    raises(lambda: deploy.cmd_sweep(BASE_CFG, args()), deploy.ApiError)
    assert not v.made("destroy_instance")


@case
def sweep_reports_survivors():
    """an instance that won't go away makes sweep fail loudly"""
    v = deployed()
    v.instances = [inst(2)]
    seen(v, 2)
    v.sticky.add(2)
    deploy.vast = lambda: v
    raises(lambda: deploy.cmd_sweep(BASE_CFG, args()), deploy.CheckFailed, "still exist")


# ── rent-test ────────────────────────────────────────────────────────────────
@case
def rent_test_records_the_id_vast_returns():
    """rent-test records the instance id from Vast; past its TTL it becomes an orphan"""
    v = FakeVast()
    deploy.vast = lambda: v
    real_check = deploy.check
    deploy.check = lambda cfg, v=None: PINS
    try:
        deploy.cmd_rent_test(BASE_CFG, args())
    finally:
        deploy.check = real_check
    (_, offer, kw), = v.made("create_instance")
    assert offer == 777 and "ORCH_SKIP_PYWORKER=1" in kw["env"] and kw["cancel_unavail"], kw
    iid = v.instances[0]["id"]
    st = deploy.load_state()
    assert str(iid) in st["manual_instances"], st
    r = deploy.survey(v, BASE_CFG, st, None, 900)
    assert [i["id"] for i in r["manual"]] == [iid]
    r = deploy.survey(v, BASE_CFG, st, None, 900, now=time.time() + 14401)
    assert [i["id"] for i in r["orphans"]] == [iid]


# ── watch ────────────────────────────────────────────────────────────────────
def tick(v, now=None):
    st = deploy.load_state()
    out = deploy.watch_tick(v, BASE_CFG, st, now)
    deploy.save_state(st)
    return out


@case
def watch_destroys_worker_stuck_booting():
    """a worker Vast reports loading past the boot deadline is destroyed; idle and error ones are not"""
    v = deployed()
    v.instances = [inst(1, dph=0.1), inst(2, dph=0.1), inst(3, dph=0.1)]
    v.workers = {1: "LOADING", 2: "IDLE", 3: "ERROR"}
    t0 = time.time()
    assert not tick(v, t0)
    # a worker still inside its download allowance is left alone
    assert not tick(v, t0 + BASE_CFG["boot"]["deadline_s"] + BASE_CFG["limits"]["stuck_grace_s"] + 1)
    limit = (BASE_CFG["boot"]["deadline_s"] + BASE_CFG["boot"]["download_max_s"]
             + BASE_CFG["limits"]["stuck_grace_s"])
    acts = tick(v, t0 + limit + 1)
    assert [c[1] for c in v.made("destroy_instance")] == [1], (acts, v.calls)


@case
def watch_ignores_unknown_status():
    """a status string the watchdog doesn't know is left alone"""
    v = deployed()
    v.instances = [inst(1)]
    v.workers = {1: "SOMETHING_NEW"}
    t0 = time.time()
    tick(v, t0)
    tick(v, t0 + 10 ** 6)
    assert not v.made("destroy_instance")


@case
def watch_pauses_on_overspend():
    """account burn over max_hourly_usd pauses the endpoint, read back; unrecorded instances untouched"""
    v = deployed()
    v.instances = [inst(1, dph=0.8), inst(9, dph=0.8)]
    v.workers = {1: "IDLE"}
    acts = tick(v)
    assert any("paused" in a for a in acts), acts
    assert v.endpoints[0]["endpoint_state"] == "stopped"
    assert not v.made("destroy_instance")


@case
def watch_within_budget_does_nothing():
    """under budget with healthy workers, the watchdog changes nothing"""
    v = deployed()
    v.instances = [inst(1, dph=0.3)]
    v.workers = {1: "IDLE"}
    assert not tick(v) and not v.made("update_endpoint") and not v.made("destroy_instance")


# ── destroy ───────────────────────────────────────────────────────────────────
@case
def destroy_removes_recorded_instances_and_confirms():
    """destroy deletes the endpoint (Vast takes its workers), then destroys recorded leftovers"""
    v = deployed()
    v.instances = [inst(1), inst(2), inst(9)]
    v.workers = {1: "IDLE"}
    seen(v, 2)
    deploy.vast = lambda: v
    deploy.cmd_destroy(BASE_CFG, args())
    assert v.made("delete_endpoint") and not v.made("delete_workergroup"), v.calls
    assert not v.groups
    assert sorted(c[1] for c in v.made("destroy_instance")) == [2]
    assert [i["id"] for i in v.instances] == [9]
    st = deploy.load_state()
    assert "endpoint_id" not in st and not st["seen_workers"]


@case
def destroy_retries_failed_workers():
    """workers Vast reports as failed to delete are destroyed, and survivors reported"""
    v = deployed()
    v.instances = [inst(1)]
    v.workers = {1: "IDLE"}
    v.sticky.add(1)
    deploy.vast = lambda: v
    raises(lambda: deploy.cmd_destroy(BASE_CFG, args()), deploy.CheckFailed, "still exist")
    assert ("destroy_instance", 1) in v.calls


@case
def destroy_follows_renamed_endpoint():
    """destroy finds the endpoint via state.json even after a rename"""
    v = deployed()
    v.endpoints[0]["endpoint_name"] = "renamed"
    deploy.vast = lambda: v
    deploy.cmd_destroy(BASE_CFG, args())
    assert v.made("delete_endpoint") and not v.endpoints


@case
def destroy_without_endpoint_still_cleans_recorded():
    """endpoint already gone: destroy still destroys recorded instances, not others"""
    v = FakeVast()
    v.instances = [inst(4), inst(5)]
    seen(v, 4)
    deploy.vast = lambda: v
    deploy.cmd_destroy(BASE_CFG, args())
    assert [c[1] for c in v.made("destroy_instance")] == [4] and [i["id"] for i in v.instances] == [5]


@case
def missing_image_tag_suggests_newest_older_build():
    """a llama.cpp release without an image points at the newest older build that has one"""
    real = deploy._ghcr_token, deploy._ghcr_has
    asked = []
    deploy._ghcr_token = lambda name: "tok"
    deploy._ghcr_has = lambda name, tag, tok: asked.append(tag) or tag == "server-cuda-b11371"
    try:
        alt = deploy.nearest_older_image("ghcr.io/ggml-org/llama.cpp:server-cuda-b11379")
        assert alt == "ghcr.io/ggml-org/llama.cpp:server-cuda-b11371", alt
        assert asked[0] == "server-cuda-b11378" and asked[-1] == "server-cuda-b11371", asked
        assert deploy.nearest_older_image("ghcr.io/ggml-org/llama.cpp:server-cuda") is None
        deploy._ghcr_has = lambda name, tag, tok: False
        assert deploy.nearest_older_image("ghcr.io/x/y:server-cuda-b100", tries=5) is None
    finally:
        deploy._ghcr_token, deploy._ghcr_has = real


@case
def embedding_model_goes_into_the_template_pinned():
    """[embedding] becomes EMBED_* template env with the pinned revision; without it, none"""
    opts = deploy.docker_options(BASE_CFG, PINS)
    e = BASE_CFG["embedding"]
    for part in (f"-e EMBED_SERVED_NAME={e['served_name']}", f"-e EMBED_REPO={e['repo']}",
                 f"-e EMBED_FILE={e['file']}", "-e EMBED_REVISION=" + "d" * 40,
                 f"-e EMBED_CTX={e['ctx']}", f"-e EMBED_POOLING={e['pooling']}"):
        assert part in opts, (part, opts)
    cfg = copy.deepcopy(BASE_CFG)
    del cfg["embedding"]
    assert "EMBED_" not in deploy.docker_options(cfg, PINS)


@case
def cheaper_example_config_within_limits():
    """the WaifuGemma4 example runs on 24 GB cards at $0.80/hr worst case, with the same projector and embeddings"""
    with open(os.path.join(os.path.dirname(__file__), "..", "deploy", "config.waifugemma4.example.toml"), "rb") as f:
        cfg = tomllib.load(f)
    assert abs(deploy.check_limits(cfg) - 0.80) < 1e-9
    assert cfg["embedding"] == BASE_CFG["embedding"]
    assert (cfg["model"]["mmproj_repo"], cfg["model"]["mmproj_file"]) == \
        (BASE_CFG["model"]["mmproj_repo"], BASE_CFG["model"]["mmproj_file"])
    assert set(cfg) == set(BASE_CFG), set(cfg) ^ set(BASE_CFG)


@case
def download_limits_reach_the_worker():
    """[boot] download budget goes into the template env; no fixed speed floor unless set; Ampere or newer"""
    opts = deploy.docker_options(BASE_CFG, PINS)
    assert f"-e DOWNLOAD_MAX_S={BASE_CFG['boot']['download_max_s']}" in opts, opts
    assert "DOWNLOAD_MIN_MBPS" not in opts, opts
    cfg = copy.deepcopy(BASE_CFG)
    cfg["boot"]["download_min_mbps"] = 3
    assert "-e DOWNLOAD_MIN_MBPS=3" in deploy.docker_options(cfg, PINS)
    del cfg["boot"]["download_max_s"]
    assert deploy.boot_limit_s(cfg) == cfg["boot"]["deadline_s"] + 3600
    with open(os.path.join(os.path.dirname(__file__), "..", "deploy", "config.waifugemma4.example.toml"), "rb") as f:
        waifu = tomllib.load(f)
    for c in (BASE_CFG, waifu):
        assert "compute_cap>=800" in c["workergroup"]["search_params"], c["workergroup"]["search_params"]
        assert "download_min_mbps" not in c["boot"], c["boot"]
    # each budget leaves room for the slowest Hub speed seen so far (~6 MB/s)
    # on that config's weights (about 29 and 17 GB)
    assert BASE_CFG["boot"]["download_max_s"] * 6e6 > 29e9 and waifu["boot"]["download_max_s"] * 6e6 > 17e9


class LogVast:
    """vastai logs returns the whole (tail of the) container log each call."""
    def __init__(self, logs, status="running"):
        self.logs_seq, self.status, self.calls = list(logs), status, 0

    def logs(self, instance_id, tail=None):
        self.calls += 1
        out = self.logs_seq[min(self.calls, len(self.logs_seq)) - 1]
        if isinstance(out, Exception):
            raise out
        return out

    def show_instance(self, id):
        return {"id": id, "actual_status": self.status} if self.status else None


def follow(v, **kw):
    out, real = [], deploy.say
    deploy.say = out.append
    try:
        return deploy.follow_instance_logs(v, 5, kw.pop("timeout", 60), poll=0, **kw), out
    finally:
        deploy.say = real


@case
def confirm_without_terminal_stops_cleanly():
    """with no terminal, a prompt stops with a hint to pass --yes instead of crashing, and nothing is rented"""
    real_stdin, real_input = sys.stdin, builtins.input
    class NoTTY:
        def isatty(self):
            return False
    class TTY:
        def isatty(self):
            return True
    try:
        sys.stdin = NoTTY()
        raises(lambda: deploy.confirm("rent it?", False), SystemExit, "pass --yes")
        assert deploy.confirm("rent it?", True)
        sys.stdin = TTY()
        def eof(_):
            raise EOFError
        builtins.input = eof
        raises(lambda: deploy.confirm("rent it?", False), SystemExit, "pass --yes")
        builtins.input = real_input
        sys.stdin = NoTTY()
        v = FakeVast()
        deploy.vast = lambda: v
        real_check = deploy.check
        deploy.check = lambda cfg, v=None: PINS
        try:
            raises(lambda: deploy.cmd_rent_test(BASE_CFG, args(yes=False)), SystemExit, "pass --yes")
        finally:
            deploy.check = real_check
        assert not v.made("create_instance") and not deploy.load_state().get("manual_instances")
    finally:
        sys.stdin = real_stdin
        builtins.input = real_input


@case
def logs_follows_an_instance_until_ready():
    """logs ID prints each new container log line once and stops at ORCH_READY"""
    v = LogVast([RuntimeError("Result not ready"),
                 "ssh setup\n[00:01:00] [boot] boot start",
                 "ssh setup\n[00:01:00] [boot] boot start\n[00:02:00] [fetch] model.gguf: 1.00/16.00 GiB (6%)",
                 "ssh setup\n[00:01:00] [boot] boot start\n[00:02:00] [fetch] model.gguf: 1.00/16.00 GiB (6%)"
                 "\n[00:20:00] [boot] ORCH_READY model=waifugemma4"])
    r, out = follow(v)
    assert r == "ready", (r, out)
    assert out.count("[00:01:00] [boot] boot start") == 1 and out[-1].endswith("ORCH_READY model=waifugemma4"), out
    assert any("no log from Vast yet" in o for o in out), out


@case
def logs_reports_fatal_gone_and_timeout():
    """logs ID returns fatal on ORCH_FATAL, gone when the instance stops silently, timeout otherwise"""
    assert follow(LogVast(["[t] [boot] ORCH_FATAL: model fetch failed (exit 4)"]))[0] == "fatal"
    r, out = follow(LogVast(["ssh setup"], status="exited"))
    assert r == "gone" and "exited" in out[-1], out
    assert follow(LogVast(["ssh setup"], status=None))[0] == "gone"
    r, out = follow(LogVast(["ssh setup"]), timeout=0)
    assert r == "timeout" and "model.log" in out[-1], out
    r, out = follow(LogVast(["ssh setup", "never read"]), once=True)
    assert r == "once" and out == ["ssh setup"], out
    # the env var naming the marker is not the marker
    assert follow(LogVast(["ORCH_FATAL_GRACE=600"]), timeout=0)[0] == "timeout"


@case
def logs_command_exit_code_follows_the_marker():
    """deploy.py logs ID exits 1 on ORCH_FATAL and 0 on ORCH_READY"""
    deploy.vast = lambda: LogVast(["[t] [boot] ORCH_FATAL: smoke test failed"])
    real = deploy.say
    deploy.say = lambda *a: None
    try:
        raises(lambda: deploy.cmd_logs(BASE_CFG, args(instance_id=5, timeout=60, once=False)), SystemExit)
        deploy.vast = lambda: LogVast(["[t] [boot] ORCH_READY model=x"])
        deploy.cmd_logs(BASE_CFG, args(instance_id=5, timeout=60, once=False))
    finally:
        deploy.say = real


print(f"---- {len(PASSED)} passed, {len(FAILED)} failed")
sys.exit(1 if FAILED else 0)

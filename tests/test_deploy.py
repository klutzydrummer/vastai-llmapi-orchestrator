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

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
import deploy  # noqa: E402

deploy.POLL_S = 0
deploy.CONFIRM_WAIT_S = 0
deploy.DESTROY_WAIT_S = 0
deploy.LOCK_WAIT_S = 0
deploy.say = lambda msg="": None

with open(os.path.join(os.path.dirname(__file__), "..", "deploy", "config.example.toml"), "rb") as f:
    BASE_CFG = tomllib.load(f)
# The example's schedule is tested on its own (schedule tests below); the
# rest of the suite runs unscheduled, so the clock doesn't change what it sees.
assert BASE_CFG["endpoint"]["cold_hours"] == "", "the examples leave cold_hours off (README, Schedules)"
EXAMPLE_COLD_HOURS = "Mon-Fri 14:00-24:00, Sat-Sun 08:00-24:00"
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
        self.offers = [{"id": 777, "machine_id": 1, "gpu_name": "RTX 4090", "dph_total": 0.31}]
        self.boot_plan = []        # per rental: (actual_status, status_msg, log text or exception)
        self.create_errors = {}    # offer id -> (exception to raise, whether the offer then disappears,
                                   #              whether an instance is created anyway)
        self.offer_rounds = []     # successive search_offers answers, before falling back to self.offers
        self.boot_logs = {}

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
        if self.offer_rounds:
            self.offers = self.offer_rounds.pop(0)
        return copy.deepcopy(self.offers[:kw.get("limit", 10)])

    def create_instance(self, offer, **kw):
        self.calls.append(("create_instance", offer, kw))
        if offer in self.create_errors:
            exc, gone, created = self.create_errors.pop(offer)
            if gone:
                self.offers = [o for o in self.offers if o["id"] != offer]
            if created:
                self.instances.append(inst(self._id(), label=kw.get("label")))
            raise exc
        iid = self._id()
        row = inst(iid, label=kw.get("label"))
        if self.boot_plan:
            row["actual_status"], row["status_msg"], self.boot_logs[iid] = self.boot_plan.pop(0)
        self.instances.append(row)
        return {"success": True, "new_contract": iid}

    def show_instance(self, id):
        return next((copy.deepcopy(i) for i in self.instances if i["id"] == id), None)

    def logs(self, id, tail=None):
        out = self.boot_logs.get(id, RuntimeError("Result not ready"))
        if isinstance(out, Exception):
            raise out
        return out

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
    assert abs(worst - 0.80) < 1e-9, worst


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
def example_configs_cap_the_download_price():
    """both example configs carry an inet_down_cost<= ceiling ($/GB) of at most 0.01"""
    for name in ("config.example.toml", "config.waifugemma4.example.toml"):
        with open(os.path.join(os.path.dirname(__file__), "..", "deploy", name), "rb") as f:
            cap = deploy.download_price_ceiling(tomllib.load(f)["workergroup"]["search_params"])
        assert cap is not None and cap <= 0.01, (name, cap)
    assert deploy.download_price_ceiling("inet_down>=500 inet_down_cost <0.02 inet_down_cost<=0.005") == 0.005
    assert deploy.download_price_ceiling("inet_down>=500 dph_total<=0.4") is None


def run_check(search_params, offers):
    """check() with the Hub, GitHub and registry answered locally; returns (output, facts)."""
    cfg = limited(workergroup__search_params=search_params)
    sizes = {cfg["model"]["file"]: 16_000_000_000, cfg["model"]["mmproj_file"]: 1_200_000_000,
             cfg["embedding"]["file"]: 1_500_000_000}
    v = FakeVast()
    v.offers = offers
    stubs = {"hf_resolve": lambda repo, rev: "a" * 40, "hf_file_size": lambda repo, sha, path: sizes[path],
             "gpu_memory_plan": lambda cfg, files: {"fits": True}, "gh_resolve": lambda repo, ref: "b" * 40,
             "gh_raw_exists": lambda repo, sha, path: True, "image_exists": lambda image: True}
    real = {k: getattr(deploy, k) for k in stubs}, deploy.say
    out, facts = [], {}
    for k, fn in stubs.items():
        setattr(deploy, k, fn)
    deploy.say = out.append
    try:
        deploy.check(cfg, v, facts)
    finally:
        for k, fn in real[0].items():
            setattr(deploy, k, fn)
        deploy.say = real[1]
    return out, facts


@case
def check_shows_download_and_cold_start_cost_per_offer():
    """check: each offer shows download $ (inet_down_cost x weight GB) and one cold start's cost; bandwidth is named as outside the limits"""
    sp = BASE_CFG["workergroup"]["search_params"]
    assert deploy.download_price_ceiling(sp) is not None
    offers = [{"id": 1, "gpu_name": "RTX 3090", "gpu_ram": 24576, "dph_total": 0.30, "inet_down_cost": 0.01},
              {"id": 2, "gpu_name": "RTX 3090", "gpu_ram": 24576, "dph_total": 0.31}]
    out, facts = run_check(sp, offers)
    assert facts["weight_bytes"] == 18_700_000_000, facts
    rows = [o for o in out if "RTX 3090" in o]
    # 18.7 GB x $0.01 = $0.19; boot 71 s + 18.7e9 / 29.4e6 s = 707 s, x $0.30/hr = $0.06
    assert "dl $0.19" in rows[0] and "cold $0.25" in rows[0], rows
    assert "dl     ?" in rows[1] and "cold     ?" in rows[1], rows
    assert any("billed apart from dph_total" in o and "at most $0.19 at inet_down_cost<=0.01" in o for o in out), out
    assert not any("no inet_down_cost<= filter" in o for o in out), out
    out, _ = run_check(re.sub(r"\s*inet_down_cost<=[0-9.]+", "", sp), offers)
    assert any("warn  no inet_down_cost<= filter" in o for o in out), out


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
    deploy.check = lambda cfg, v=None, facts=None: PINS
    try:
        deploy.cmd_rent_test(BASE_CFG, args(no_follow=True))
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


def rent_with_boots(*boots, offers=None, v=None, weight_bytes=None, **kw):
    v = v or FakeVast()
    if offers:
        v.offers = offers
    v.boot_plan = list(boots)
    deploy.vast = lambda: v
    real = deploy.check, deploy.VAST_ERROR_CONFIRM_S, deploy.LOGS_POLL_S, deploy.say

    def fake_check(cfg, v=None, facts=None):
        if weight_bytes and facts is not None:
            facts["weight_bytes"] = weight_bytes
        return PINS
    deploy.check = fake_check
    deploy.VAST_ERROR_CONFIRM_S, deploy.LOGS_POLL_S = 0, 0
    out = []
    deploy.say = out.append
    try:
        try:
            deploy.cmd_rent_test(BASE_CFG, args(**kw))
            err = None
        except deploy.CheckFailed as e:
            err = str(e)
    finally:
        deploy.check, deploy.VAST_ERROR_CONFIRM_S, deploy.LOGS_POLL_S, deploy.say = real
    return v, out, err


THREE_HOSTS = [{"id": 777, "machine_id": 1, "gpu_name": "RTX 3090", "dph_total": 0.16},
               {"id": 778, "machine_id": 2, "gpu_name": "RTX 3090", "dph_total": 0.17},
               {"id": 779, "machine_id": 3, "gpu_name": "RTX 3090", "dph_total": 0.18}]


@case
def rent_test_replaces_a_host_that_fails_to_boot():
    """rent-test: a host where Vast says 'Secrets fetch failed' has its log saved and is destroyed; the next cheapest other host is rented and boots"""
    v, out, err = rent_with_boots(("loading", "Secrets fetch failed: network/subprocess error", RuntimeError("x")),
                                  ("running", "", "ssh setup\n[t] [boot] ORCH_READY model=x"), offers=THREE_HOSTS)
    assert err is None, (err, out)
    assert [c[1] for c in v.made("create_instance")] == [777, 778], v.calls
    assert [c[1] for c in v.made("destroy_instance")] == [101], v.calls
    assert [i["id"] for i in v.instances] == [102], v.instances
    assert set(deploy.load_state()["manual_instances"]) == {"102"}
    saved = os.path.join(os.path.dirname(deploy.STATE_PATH), "boots", "101.log")
    assert "Secrets fetch failed" in open(saved).read(), out
    assert any("is ready" in o for o in out), out


@case
def rent_test_keeps_a_boot_that_failed_for_another_reason():
    """rent-test: an ORCH_FATAL that isn't about the host is reported and the instance kept for a look, not replaced"""
    v, out, err = rent_with_boots(("running", "", "[t] [boot] ORCH_FATAL: smoke test failed"), offers=THREE_HOSTS)
    assert err and "did not boot (fatal)" in err, (err, out)
    assert len(v.made("create_instance")) == 1 and not v.made("destroy_instance"), v.calls
    # a host too slow for the download budget is the host's fault: replaced
    v, out, err = rent_with_boots(
        ("running", "", "[t] [boot] ORCH_FATAL: weights download too slow on this host (see the [fetch] line above)"),
        ("running", "", "[t] [boot] ORCH_READY model=x"), offers=THREE_HOSTS)
    assert err is None and [c[1] for c in v.made("create_instance")] == [777, 778], (err, v.calls)


@case
def rent_test_stops_after_rent_attempts():
    """rent-test: after limits.rent_attempts hosts fail it stops with nothing left running"""
    bad = ("loading", "Secrets fetch failed", RuntimeError("x"))
    v, out, err = rent_with_boots(bad, bad, bad, bad, offers=THREE_HOSTS + [dict(THREE_HOSTS[0], id=780, machine_id=4)])
    assert err and "3 attempts failed" in err and "nothing is left running" in err, (err, out)
    assert len(v.made("create_instance")) == 3 and not v.instances, v.calls


def vast_400(offer):
    """What the vastai SDK raised on rental attempt for offer 50484116 (issue 16)."""
    r = requests.Response()
    r.status_code, r._content = 400, b'{"success": false, "error": "invalid_args", "msg": "offer no longer available"}'
    r.url = f"https://console.vast.ai/api/v0/asks/{offer}/"
    return requests.exceptions.HTTPError(f"400 Client Error: Bad Request for url: {r.url}", response=r)


@case
def rent_test_tries_the_next_offer_when_one_is_taken():
    """rent-test: Vast refuses an offer that is then gone; Vast's answer is printed and the next offer is rented"""
    ready = ("running", "", "[t] [boot] ORCH_READY model=x")
    v = FakeVast()
    v.create_errors = {777: (vast_400(777), True, False)}
    v, out, err = rent_with_boots(ready, offers=THREE_HOSTS, v=v)
    assert err is None, (err, out)
    assert [c[1] for c in v.made("create_instance")] == [777, 778], v.calls
    assert any("HTTP 400" in o and "offer no longer available" in o for o in out), out
    assert any("no longer listed" in o and "nothing was rented" in o for o in out), out
    assert set(deploy.load_state()["manual_instances"]) == {str(v.instances[0]["id"])}


@case
def rent_test_stops_when_a_listed_offer_is_refused():
    """rent-test: Vast refuses an offer that is still listed; it stops with Vast's answer, nothing rented, no retry"""
    v = FakeVast()
    v.create_errors = {777: (vast_400(777), False, False)}
    v, out, err = rent_with_boots(offers=THREE_HOSTS, v=v)
    assert err and "still listed" in err and "offer no longer available" in err, (err, out)
    assert len(v.made("create_instance")) == 1 and not v.instances, v.calls


@case
def rent_test_stops_when_a_refused_create_left_an_instance():
    """rent-test: a create that errors but leaves a new instance on the account stops, names it, and is not retried"""
    v = FakeVast()
    v.create_errors = {777: (requests.exceptions.ConnectionError("reset"), True, True)}
    v, out, err = rent_with_boots(offers=THREE_HOSTS, v=v)
    assert err and "didn't have before" in err and str(v.instances[0]["id"]) in err, (err, out)
    assert len(v.made("create_instance")) == 1 and not v.made("destroy_instance"), v.calls


@case
def rent_test_searches_again_right_before_renting():
    """rent-test: the offer is re-searched after the prompt; a cheaper offer that went away is not rented"""
    ready = ("running", "", "[t] [boot] ORCH_READY model=x")
    v = FakeVast()
    v.offer_rounds = [list(THREE_HOSTS), THREE_HOSTS[1:]]
    v, out, err = rent_with_boots(ready, v=v)
    assert err is None and [c[1] for c in v.made("create_instance")] == [778], (err, v.calls)
    assert any("offers changed" in o for o in out), out


@case
def rent_test_counts_taken_offers_as_attempts():
    """rent-test: offers taken one after another use up limits.rent_attempts, with nothing rented"""
    v = FakeVast()
    v.create_errors = {o["id"]: (vast_400(o["id"]), True, False) for o in THREE_HOSTS}
    v, out, err = rent_with_boots(offers=THREE_HOSTS + [dict(THREE_HOSTS[0], id=780, machine_id=4, dph_total=0.19)], v=v)
    assert err and "3 attempts failed" in err and "offer 779 taken" in err, (err, out)
    assert len(v.made("create_instance")) == 3 and not v.instances, v.calls


@case
def rent_test_prints_the_download_cost_of_the_pick():
    """rent-test: the picked offer's line shows what downloading the weights costs there, apart from $/hr"""
    ready = ("running", "", "[t] [boot] ORCH_READY model=x")
    offers = [dict(THREE_HOSTS[0], inet_down_cost=0.0247), dict(THREE_HOSTS[1])]
    v, out, err = rent_with_boots(ready, offers=offers, weight_bytes=18.7e9)
    assert err is None, (err, out)
    assert any(o.startswith("cheapest match: offer 777") and "+ $0.46 to download" in o for o in out), out
    # an offer without inet_down_cost says so instead of claiming it's free
    assert "price unknown" in deploy._offer_line(THREE_HOSTS[1], 18.7e9)
    assert "download" not in deploy._offer_line(THREE_HOSTS[1])


@case
def rent_test_hold_on_fatal_sets_the_grace_for_that_rental_only():
    """rent-test --hold-on-fatal 30 sets ORCH_FATAL_GRACE=1800 on the rental and says what it costs; without it, no override"""
    bad = ("running", "", "[t] [boot] ORCH_FATAL: smoke test failed")
    v, out, err = rent_with_boots(bad, offers=THREE_HOSTS, hold_on_fatal=30)
    (_, _, kw), = v.made("create_instance")
    assert "-e ORCH_FATAL_GRACE=1800" in kw["env"], kw["env"]
    assert any(o.startswith("!! --hold-on-fatal 30") and "$0.160/hr" in o and "$0.08" in o
               and "manual_ttl_s=14400" in o for o in out), out
    assert err and "stops itself 30 min after the failure" in err, err
    v, out, err = rent_with_boots(bad, offers=THREE_HOSTS)
    (_, _, kw), = v.made("create_instance")
    assert "ORCH_FATAL_GRACE" not in kw["env"] and not any(o.startswith("!!") for o in out), kw["env"]


@case
def rent_test_hold_on_fatal_is_bounded_by_manual_ttl():
    """rent-test --hold-on-fatal longer than limits.manual_ttl_s, or not above 0, is refused before anything is rented"""
    for hold in (241, 0, -5):
        v, out, err = rent_with_boots(offers=THREE_HOSTS, hold_on_fatal=hold)
        assert err and "--hold-on-fatal" in err and not v.made("create_instance"), (hold, err)
    v, out, err = rent_with_boots(("running", "", "[t] [boot] ORCH_READY model=x"), offers=THREE_HOSTS,
                                  hold_on_fatal=240)
    assert err is None, err


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


def warm_cfg(**endpoint):
    cfg = copy.deepcopy(BASE_CFG)
    cfg["endpoint"].update({"warm_hours": "15:00-23:00", "schedule_tz": "UTC", **endpoint})
    return cfg


def utc(h, m=0):
    import datetime as dt
    return dt.datetime(2026, 10, 5, h, m, tzinfo=dt.timezone.utc).timestamp()


@case
def warm_hours_hold_min_load_through_watch():
    """warm hours: watch raises endpoint min_load at the start, reads it back, leaves it, and restores it after"""
    cfg = warm_cfg()
    v = FakeVast()
    deploy._apply(v, cfg, PINS, "-p 3000:3000")
    v.calls.clear()
    def wtick(now, c=cfg):
        st = deploy.load_state()
        out = deploy.watch_tick(v, c, st, now)
        deploy.save_state(st)
        return out
    acts = wtick(utc(16))
    (_, _, kw), = v.made("update_endpoint")
    assert kw["min_load"] == 1 and kw["max_workers"] == cfg["endpoint"]["max_workers"], kw
    assert v.endpoints[0]["min_load"] == 1 and "warm hours on" in acts[0], acts
    v.calls.clear()
    assert not wtick(utc(22, 59)) and not v.made("update_endpoint")
    # someone changed it behind our back: put back
    v.endpoints[0]["min_load"] = 0
    wtick(utc(17))
    assert v.endpoints[0]["min_load"] == 1
    acts = wtick(utc(23, 0))
    assert v.endpoints[0]["min_load"] == cfg["endpoint"]["min_load"] and "warm hours off" in acts[-1], acts
    # pause/resume during warm hours send the warm value, so they don't undo it
    assert deploy.endpoint_limits(cfg, utc(15))["min_load"] == 1
    assert deploy.endpoint_limits(cfg, utc(14, 59))["min_load"] == cfg["endpoint"]["min_load"]
    # windows may cross midnight; no warm_hours means nothing changes
    assert deploy.warm_now(warm_cfg(warm_hours="22:00-02:00"), utc(1, 30))
    assert not deploy.warm_now(warm_cfg(warm_hours="22:00-02:00"), utc(2))
    v.calls.clear()
    plain = copy.deepcopy(BASE_CFG)
    wtick(utc(16), plain)
    assert not v.made("update_endpoint") and "warm_active" not in deploy.load_state()


@case
def warm_hours_settings_are_checked():
    """bad warm_hours, warm_tz or warm_min_load are refused before anything runs"""
    raises(lambda: deploy.check_limits(warm_cfg(warm_hours="3pm-11pm")), deploy.CheckFailed, "warm_hours")
    raises(lambda: deploy.check_limits(warm_cfg(warm_hours="15:00-15:00")), deploy.CheckFailed, "empty")
    raises(lambda: deploy.check_limits(warm_cfg(schedule_tz="Mars/Olympus")), deploy.CheckFailed, "schedule_tz")
    raises(lambda: deploy.check_limits(warm_cfg(schedule_tz="")), deploy.CheckFailed, "needs endpoint.schedule_tz")
    raises(lambda: deploy.check_limits(warm_cfg(warm_tz="UTC")), deploy.CheckFailed, "now endpoint.schedule_tz")
    raises(lambda: deploy.check_limits(warm_cfg(warm_min_load=0)), deploy.CheckFailed, "warm_min_load")
    deploy.check_limits(warm_cfg())


def at(day, h, m=0):
    """UTC epoch seconds on the week of Monday 2026-10-05; day 0 = Monday."""
    import datetime as dt
    return dt.datetime(2026, 10, 5 + day, h % 24, m, tzinfo=dt.timezone.utc).timestamp() + (h // 24) * 86400


def cold_cfg(**endpoint):
    cfg = copy.deepcopy(BASE_CFG)
    cfg["endpoint"].update({"cold_hours": EXAMPLE_COLD_HOURS, "schedule_tz": "UTC", **endpoint})
    return cfg


@case
def schedules_take_days_of_the_week():
    """warm_hours/cold_hours: day ranges (also wrapping ones), every-day ranges, ranges past midnight"""
    c = cold_cfg()
    assert EXAMPLE_COLD_HOURS == "Mon-Fri 14:00-24:00, Sat-Sun 08:00-24:00"
    for day, h, m, want in [(0, 13, 59, False), (0, 14, 0, True), (0, 23, 59, True), (1, 0, 0, False),
                            (4, 23, 59, True), (5, 0, 0, False), (5, 7, 59, False), (5, 8, 0, True),
                            (6, 23, 59, True), (7, 0, 1, False), (2, 6, 30, False)]:
        assert deploy.cold_now(c, at(day, h, m)) is want, (day, h, m, want)
    w = warm_cfg(warm_hours="Fri-Mon 22:00-02:00")
    assert deploy.warm_now(w, at(0, 23)) and deploy.warm_now(w, at(1, 1, 59)) and not deploy.warm_now(w, at(1, 22))
    assert deploy.warm_now(w, at(4, 22)) and deploy.warm_now(w, at(6, 1)) and not deploy.warm_now(w, at(3, 23))
    every = warm_cfg(warm_hours="sat 10:00-12:00, 18:00-19:00")
    assert deploy.warm_now(every, at(2, 18, 30)) and deploy.warm_now(every, at(5, 11)) and not deploy.warm_now(every, at(2, 11))
    # the zone counts: 14:00 in Chicago is 19:00 UTC in October
    assert deploy.cold_now(cold_cfg(schedule_tz="America/Chicago"), at(0, 19))
    assert not deploy.cold_now(cold_cfg(schedule_tz="America/Chicago"), at(0, 18, 59))


@case
def schedule_settings_are_checked():
    """bad day names, cold_hours with cold_workers above 0, or a bad cold_hours_workers are refused"""
    for bad in ("Mon-Fry 14:00-24:00", "Mon-Wed-Fri 14:00-15:00", "Mon Tue 14:00-15:00", "Mo 14:00-15:00"):
        raises(lambda: deploy.check_limits(cold_cfg(cold_hours=bad)), deploy.CheckFailed, "cold_hours")
    raises(lambda: deploy.check_limits(cold_cfg(cold_workers=1)), deploy.CheckFailed, "cold_workers = 0")
    for n in (0, 2, True, 1.0):
        raises(lambda: deploy.check_limits(cold_cfg(cold_hours_workers=n)), deploy.CheckFailed, "cold_hours_workers")
    raises(lambda: deploy.check_limits(cold_cfg(schedule_tz="")), deploy.CheckFailed, "needs endpoint.schedule_tz")
    deploy.check_limits(cold_cfg())
    # the shipped examples pass as they are
    for name in ("config.example.toml", "config.waifugemma4.example.toml"):
        with open(os.path.join(os.path.dirname(__file__), "..", "deploy", name), "rb") as f:
            deploy.check_limits(tomllib.load(f))


@case
def cold_hours_keep_a_worker_only_after_one_started():
    """cold hours: cold_workers goes to 1 only once a worker exists, back to 0 after; stopped workers are removed outside them"""
    cfg = cold_cfg()
    v = FakeVast()
    deploy._apply(v, cfg, PINS, "-p 3000:3000")
    assert v.endpoints[0]["cold_workers"] == 0
    v.calls.clear()
    def wtick(now):
        st = deploy.load_state()
        out = deploy.watch_tick(v, cfg, st, now)
        deploy.save_state(st)
        return out
    # in cold hours with no worker: nothing for the autoscaler to rent ahead of a request
    wtick(at(0, 14, 5))
    assert v.endpoints[0]["cold_workers"] == 0
    v.calls.clear()
    assert not wtick(at(0, 14, 6)) and not v.made("update_endpoint")
    # a request started a worker: keep it when it idles
    v.instances, v.workers = [inst(1)], {1: "IDLE"}
    acts = wtick(at(0, 15))
    assert v.endpoints[0]["cold_workers"] == 1 and "cold hours holding" in acts[0], acts
    # apply/pause/resume while holding send the held value, so they don't undo it
    st = deploy.load_state()
    assert deploy.endpoint_limits(cfg, at(0, 16), st)["cold_workers"] == 1
    assert deploy.endpoint_limits(cfg, at(1, 0), st)["cold_workers"] == 0
    assert deploy.endpoint_limits(cfg, at(0, 16))["cold_workers"] == 0
    # it went cold: kept during cold hours
    v.instances = [inst(1, status="stopped")]
    v.workers = {1: "STOPPED"}
    v.calls.clear()
    assert not wtick(at(0, 20)) and not v.made("destroy_instance")
    # someone changed it behind our back: put back
    v.endpoints[0]["cold_workers"] = 0
    wtick(at(0, 21))
    assert v.endpoints[0]["cold_workers"] == 1
    # midnight: cold_workers back to 0, and the stopped worker is removed
    v.calls.clear()
    acts = wtick(at(1, 0))
    assert v.endpoints[0]["cold_workers"] == 0, acts
    assert [c[1] for c in v.made("destroy_instance")] == [1], (acts, v.calls)
    # a running worker outside cold hours is left to idle out
    v.instances, v.workers = [inst(2)], {2: "IDLE"}
    v.calls.clear()
    wtick(at(1, 9))
    assert not v.made("destroy_instance") and v.endpoints[0]["cold_workers"] == 0
    # removing cold_hours from the config reverts and forgets
    plain = copy.deepcopy(BASE_CFG)
    st = deploy.load_state()
    deploy.watch_tick(v, plain, st, at(1, 15))
    assert "cold_held" not in st and v.endpoints[0]["cold_workers"] == 0


@case
def cold_hours_drop_an_unavail_worker():
    """cold hours: a kept worker Vast reports unavail stops the hold (cold_workers 0 first) and is destroyed"""
    cfg = cold_cfg()
    v = FakeVast()
    deploy._apply(v, cfg, PINS, "-p 3000:3000")
    def wtick(now):
        st = deploy.load_state()
        out = deploy.watch_tick(v, cfg, st, now)
        deploy.save_state(st)
        return out
    v.instances, v.workers = [inst(1)], {1: "IDLE"}
    wtick(at(0, 15))
    assert v.endpoints[0]["cold_workers"] == 1
    v.instances, v.workers = [inst(1, status="stopped")], {1: "STOPPED"}
    v.calls.clear()
    assert not wtick(at(0, 16)) and not v.made("destroy_instance")
    for row_status, inst_fields in (("unavail", {}), ("STOPPED", {"cur_state": "unavail"}),
                                    ("STOPPED", {"actual_status": "unavailable"})):
        v.instances = [dict(inst(1, status="stopped"), **inst_fields)]
        v.workers = {1: row_status}
        v.endpoints[0]["cold_workers"] = 1
        st = deploy.load_state()
        st["cold_held"] = True
        deploy.save_state(st)
        v.calls.clear()
        acts = wtick(at(0, 17))
        updates = [i for i, c in enumerate(v.calls) if c[0] == "update_endpoint"]
        destroys = [i for i, c in enumerate(v.calls) if c[0] == "destroy_instance"]
        assert v.endpoints[0]["cold_workers"] == 0 and updates and destroys, (row_status, acts, v.calls)
        assert updates[0] < destroys[0], v.calls   # floor lowered before the worker goes
        assert any("unavail" in a for a in acts), acts
        v.instances, v.workers = [inst(1, status="stopped")], {1: "STOPPED"}
    # a plain stopped worker in cold hours is still kept
    v.instances, v.workers = [inst(2)], {2: "IDLE"}
    wtick(at(0, 18))
    v.instances, v.workers = [inst(2, status="stopped")], {2: "STOPPED"}
    v.calls.clear()
    wtick(at(0, 19))
    assert not v.made("destroy_instance") and v.endpoints[0]["cold_workers"] == 1


@case
def stopped_instances_show_their_storage_rate():
    """status: a stopped instance bills storage (storage_cost x disk_space), not its running dph_total"""
    run = inst(1, dph=0.485)
    stopped = dict(inst(2, status="exited", dph=0.485), storage_cost=0.15, disk_space=48)
    unknown = inst(3, status="stopped", dph=0.485)
    assert deploy._rate(run) == 0.485
    assert abs(deploy._rate(stopped) - 0.01) < 1e-9
    assert deploy._rate(unknown) is None
    assert "storage $0.010/hr" in deploy._describe(stopped) and "0.485" not in deploy._describe(stopped)
    assert "storage rate not reported" in deploy._describe(unknown)
    lines = []
    real = deploy.say
    deploy.say = lines.append
    try:
        deploy._report({"workers": [run, stopped, unknown], "manual": [], "orphans": [], "young": [],
                        "unattributed": [], "why": {}})
    finally:
        deploy.say = real
    assert "this deployment: 3 instance(s), $0.495/hr" in lines, lines
    assert any("plus storage Vast didn't report" in x for x in lines), lines


# ── GPU memory preflight ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fakes"))
import gguf  # noqa: E402

GGUF_DIR = tempfile.mkdtemp(prefix="test_deploy_gguf.")
KIND = {"chat.gguf": "gemma4", "mmproj.gguf": "gemma4-mmproj", "embed.gguf": "qwen3-embed"}
for _name, _kind in KIND.items():
    gguf.write(os.path.join(GGUF_DIR, _name), gguf.KINDS[_kind]())


def local_header(repo, sha, path):
    return deploy.vram.read_header(deploy.vram.FileSource(os.path.join(GGUF_DIR, path)))


def gemma_cfg(**llama):
    with open(os.path.join(os.path.dirname(__file__), "..", "deploy", "config.waifugemma4.example.toml"), "rb") as f:
        cfg = tomllib.load(f)
    cfg["llama"].update(llama)
    return cfg


def gemma_files(chat_size=14_970_000_000):
    return {"chat": ("r", "a" * 40, "chat.gguf", chat_size), "mmproj": ("r", "c" * 40, "mmproj.gguf", 1_194_800_000),
            "embed": ("r", "d" * 40, "embed.gguf", 639_153_184)}


@case
def check_sizes_for_the_smallest_allowed_gpu():
    """check: WaifuGemma4 + BF16 projector + GPU embedding fit the smallest GPU gpu_ram>=22 allows (24 GiB)"""
    cfg = gemma_cfg()
    assert deploy.smallest_gpu_mib(cfg) == 24576
    p = deploy.gpu_memory_plan(cfg, gemma_files(), read_header=local_header)
    assert p["fits"] and p["total_mib"] == 24576 and p["free_mib"] == 24576 - deploy.DRIVER_RESERVE_MIB
    assert "embedding weights" in p["parts_mib"] and p["ubatch"] == 1120


@case
def default_example_fits_its_smallest_gpu():
    """check: the default example's Pantheon Q4_K_M (18,750,500,384 bytes) fits 24 GiB above ctx_min"""
    assert deploy.smallest_gpu_mib(BASE_CFG) == 24576
    p = deploy.gpu_memory_plan(BASE_CFG, gemma_files(18_750_500_384), read_header=local_header)
    assert p["fits"] and BASE_CFG["llama"]["ctx_min"] <= p["ctx"] <= BASE_CFG["llama"]["ctx"], p


@case
def check_fails_a_set_that_cannot_fit_and_lists_options():
    """check: a Q8_0 chat model on 24 GiB fails, says by how much and lists the owner's options without picking one"""
    try:
        deploy.gpu_memory_plan(gemma_cfg(), gemma_files(26_900_000_000), read_header=local_header)
        raise AssertionError("expected CheckFailed")
    except deploy.CheckFailed as e:
        msg = str(e)
    assert re.search(r"[0-9]+ MiB short", msg), msg
    for opt in ("larger GPU", "llama.ctx_min", "--no-mmproj-offload", "embedding.gpu = false"):
        assert opt in msg, (opt, msg)


@case
def check_counts_settings_that_move_parts_off_the_gpu():
    """check: --no-mmproj-offload and embedding.gpu = false drop those parts from the GPU budget"""
    cfg = gemma_cfg(extra_args="--no-mmproj-offload")
    cfg["embedding"]["gpu"] = False
    p = deploy.gpu_memory_plan(cfg, gemma_files(), read_header=local_header)
    assert not any(k.startswith(("embedding", "projector")) for k in p["parts_mib"]), p["parts_mib"]


@case
def check_validates_gpu_settings():
    """check: ctx_min outside 256 per slot..ctx, an unknown cache type or a non-bool embedding.gpu are refused"""
    raises(lambda: deploy.plan_settings(gemma_cfg(ctx_min=65536)), text="llama.ctx_min")
    raises(lambda: deploy.plan_settings(gemma_cfg(ctx_min=256)), text="llama.ctx_min")
    raises(lambda: deploy.plan_settings(gemma_cfg(cache_type="q9")), text="llama.cache_type")
    cfg = gemma_cfg()
    cfg["embedding"]["gpu"] = "yes"
    raises(lambda: deploy.plan_settings(cfg), text="embedding.gpu")


@case
def gpu_settings_reach_the_worker_env():
    """docker options carry ctx_min, image_tokens, embedding gpu and cache type to the worker"""
    opts = deploy.docker_options(gemma_cfg(), PINS)
    for kv in ("LLAMA_CTX_MIN=16384", "LLAMA_IMAGE_TOKENS=1120", "EMBED_GPU=1", "EMBED_CACHE_TYPE=f16"):
        assert f"-e {kv}" in opts, kv
    cfg = gemma_cfg()
    cfg["embedding"]["gpu"] = False
    assert "-e EMBED_GPU=0" in deploy.docker_options(cfg, PINS)


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
    # on that config's weights (about 21 and 17 GB)
    assert BASE_CFG["boot"]["download_max_s"] * 6e6 > 21e9 and waifu["boot"]["download_max_s"] * 6e6 > 17e9


class LogVast:
    """vastai logs returns the whole (tail of the) container log each call.
    status may be a list of (actual_status, status_msg), one per show_instance
    call, the last repeating."""
    def __init__(self, logs, status="running"):
        self.logs_seq, self.status, self.calls, self.shows = list(logs), status, 0, 0

    def logs(self, instance_id, tail=None):
        self.calls += 1
        out = self.logs_seq[min(self.calls, len(self.logs_seq)) - 1]
        if isinstance(out, Exception):
            raise out
        return out

    def show_instance(self, id):
        if isinstance(self.status, list):
            self.shows += 1
            status, msg = self.status[min(self.shows, len(self.status)) - 1]
            return {"id": id, "actual_status": status, "status_msg": msg}
        return {"id": id, "actual_status": self.status} if self.status else None


def ticking(step):
    """A clock that moves on `step` seconds each time it is read."""
    t = [1_000_000.0]
    def clock():
        t[0] += step
        return t[0]
    return clock


def follow(v, **kw):
    out, real = [], deploy.say
    deploy.say = out.append
    try:
        return deploy.follow_instance_logs(v, 5, kw.pop("timeout", 60), poll=0, **kw)[0], out
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
        deploy.check = lambda cfg, v=None, facts=None: PINS
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
    # the log streams from the start, whatever Vast says about the instance
    v = LogVast(["ssh setup\n[t] [boot] ORCH_READY model=x"], status=[("loading", "")])
    assert follow(v, timeout=10 ** 6)[0] == "ready"


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
    assert r == "once" and out == ["ssh setup", "[vast] instance 5: running"], out
    # the env var naming the marker is not the marker
    assert follow(LogVast(["ORCH_FATAL_GRACE=600"]), timeout=0)[0] == "timeout"


SECRETS = "Secrets fetch failed: network/subprocess error"


@case
def logs_stops_on_an_error_vast_reports():
    """rental 54214819: Vast says 'Secrets fetch failed' while loading; logs ID prints it and exits within a few minutes"""
    v = LogVast([RuntimeError("Result not ready")], status=[("loading", SECRETS)])
    r, out = follow(v, clock=ticking(20), timeout=10 ** 6)
    assert r == "vast_error", (r, out)
    assert any(o.startswith("[vast] instance 5: loading") and SECRETS in o for o in out), out
    assert "still bills" in out[-1], out
    # the log was being fetched all along, though Vast never started a container
    assert v.calls >= 1, v.calls
    # a message that clears once the container starts isn't an error
    v = LogVast(["ssh setup", "ssh setup", "ssh setup\n[t] [boot] ORCH_READY model=x"],
                status=[("loading", "Error response from daemon: retrying"), ("running", "")])
    assert follow(v, clock=ticking(20), timeout=10 ** 6)[0] == "ready"


@case
def logs_stops_on_an_instance_stuck_booting():
    """an instance Vast keeps reporting as loading, with no error, counts as stuck after stuck_grace_s from its rental"""
    v = LogVast([RuntimeError("Result not ready")], status=[("loading", "Pulling image layer 3/9")])
    r, out = follow(v, clock=ticking(20), timeout=10 ** 6, stuck_s=600)
    assert r == "stuck" and "stuck_grace_s=600" in out[-1], (r, out)
    # issue 13: a status line about once a minute while waiting, with context
    waits = [o for o in out if o.startswith("still waiting")]
    assert len(waits) >= 5 and "loading" in waits[-1] and "Pulling image" in waits[-1] \
        and "no log from Vast yet" in waits[-1], out
    # rented 590 s before logs started: stuck almost at once
    clock = ticking(20)
    rented = clock() - 590
    r, out = follow(LogVast([RuntimeError("x")], status=[("loading", "")]), clock=clock, timeout=10 ** 6,
                    stuck_s=600, since=rented)
    assert r == "stuck" and len([o for o in out if o.startswith("still waiting")]) == 0, (r, out)
    # once the container runs, the download and model load have their own limits
    r, out = follow(LogVast(["ssh setup"], status=[("running", "")]), clock=ticking(20), timeout=3000,
                    stuck_s=600)
    assert r == "timeout", (r, out)


@case
def describe_and_vast_error_read_the_status_message():
    """status lists carry Vast's status message; only a booting instance's error message counts as an error"""
    i = dict(inst(1, status="loading"), status_msg=SECRETS)
    assert SECRETS in deploy._describe(i) and deploy.vast_error(i) == SECRETS
    assert deploy.vast_error(dict(i, actual_status="running")) == ""
    assert deploy.vast_error(dict(i, status_msg="Pulling image")) == ""


@case
def watch_destroys_test_rental_stuck_loading():
    """watch destroys a rent-test instance Vast still reports loading after stuck_grace_s; running or younger ones stay"""
    v = FakeVast()
    t0 = time.time()
    v.instances = [dict(inst(1, status="loading"), status_msg=SECRETS), inst(2, status="loading"),
                   inst(3, status="running")]
    st = deploy.load_state()
    grace = BASE_CFG["limits"]["stuck_grace_s"]
    st["manual_instances"] = {"1": {"created_at": t0 - grace - 1}, "2": {"created_at": t0 - grace + 60},
                              "3": {"created_at": t0 - grace - 1}}
    deploy.save_state(st)
    acts = tick(v, t0)
    assert [c[1] for c in v.made("destroy_instance")] == [1], (acts, v.calls)
    assert "still loading" in acts[0] and SECRETS in acts[0], acts
    assert set(deploy.load_state()["manual_instances"]) == {"2", "3"}


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

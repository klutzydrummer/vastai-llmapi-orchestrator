#!/usr/bin/env python3
"""deploy.py tests against an in-memory fake of the Vast API: no duplicate
endpoints or workergroups, no destroying on a guess, orphans found and
confirmed gone, spend limits enforced. Run: python3 tests/test_deploy.py"""

import copy
import os
import sys
import tempfile
import tomllib
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
import deploy  # noqa: E402

deploy.POLL_S = 0
deploy.CONFIRM_WAIT_S = 0
deploy.DESTROY_WAIT_S = 0
deploy.say = lambda msg="": None

with open(os.path.join(os.path.dirname(__file__), "..", "deploy", "config.example.toml"), "rb") as f:
    BASE_CFG = tomllib.load(f)
NAME = BASE_CFG["endpoint"]["name"]
PINS = {"model_revision": "a" * 40, "orch_ref": "b" * 40, "mmproj_revision": "c" * 40}


class Resp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeVast:
    def __init__(self):
        self.endpoints, self.groups, self.instances, self.workers = [], [], [], []
        self.calls = []
        self.fail = set()          # method names that answer with an error dict
        self.sticky = set()        # instance ids that survive destroy
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
        return [{"id": i} for i in self.workers]

    def create_template(self, **kw):
        self.calls.append(("create_template",))
        return {"success": True, "template": {"hash_id": f"tpl{self._id()}"}}

    def create_endpoint(self, **kw):
        self.calls.append(("create_endpoint", kw))
        eid = self._id()
        self.endpoints.append({"id": eid, "endpoint_name": kw["endpoint_name"]})
        return {"success": True, "result": eid}

    def update_endpoint(self, id, **kw):
        self.calls.append(("update_endpoint", id, kw))

    def update_workergroup(self, id, **kw):
        self.calls.append(("update_workergroup", id, kw))

    def _post(self, path, json_data=None):
        self.calls.append(("create_workergroup", json_data))
        self.groups.append({"id": self._id(), "endpoint_id": json_data["endpoint_id"]})
        return Resp({"success": True})

    def delete_workergroup(self, id):
        self.calls.append(("delete_workergroup", id))
        self.groups = [g for g in self.groups if g["id"] != id]

    def delete_endpoint(self, id):
        self.calls.append(("delete_endpoint", id))
        self.endpoints = [e for e in self.endpoints if e["id"] != id]

    def destroy_instance(self, id):
        self.calls.append(("destroy_instance", id))
        if id not in self.sticky:
            self.instances = [i for i in self.instances if i["id"] != id]

    def made(self, name):
        return [c for c in self.calls if c[0] == name]


def inst(id, age=3600, env=None, dph=0.3):
    return {"id": id, "duration": age, "dph_total": dph, "actual_status": "running", "gpu_name": "RTX 4090",
            "extra_env": {"ORCH_DEPLOYMENT": NAME, **(env or {})} if env is not False else {}}


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
    a = {"yes": True, "dry_run": False, "min_age": 900}
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
        raises(lambda: with_lock(lambda: apply(v)), deploy.CheckFailed, "another apply")
    assert not v.calls
    with_lock(lambda: apply(v))      # lock released afterwards


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
    sp = BASE_CFG["workergroup"]["search_params"].replace("dph_total<=0.40", "dph_total>0.01")
    raises(lambda: deploy.check_limits(limited(workergroup__search_params=sp)), text="price ceiling")
    assert deploy.price_ceiling("gpu_ram>=22 dph_total < 0.5 dph_total<=0.3") == 0.3


@case
def worst_case_over_budget_refused():
    """max_workers x price above max_hourly_usd, or above max_workers_cap, is refused"""
    raises(lambda: deploy.check_limits(limited(limits__max_hourly_usd=0.5)), text="worst case")
    raises(lambda: deploy.check_limits(limited(endpoint__max_workers=3)), text="max_workers_cap")


# ── sweep ─────────────────────────────────────────────────────────────────────
def deployed():
    v = FakeVast()
    apply(v)
    v.calls.clear()
    return v


@case
def sweep_destroys_only_orphans():
    """sweep destroys old marked instances the autoscaler doesn't count, nothing else"""
    v = deployed()
    v.instances = [inst(1), inst(2), inst(3, age=60), inst(4, env=False),
                   inst(5, env={"ORCH_SKIP_PYWORKER": "1"})]
    v.workers = [1]
    deploy.vast = lambda: v
    deploy.cmd_sweep(BASE_CFG, args())
    destroyed = sorted(c[1] for c in v.made("destroy_instance"))
    assert destroyed == [2, 5], destroyed
    assert sorted(i["id"] for i in v.instances) == [1, 3, 4]


@case
def sweep_dry_run_destroys_nothing():
    """sweep --dry-run only reports"""
    v = deployed()
    v.instances = [inst(2)]
    deploy.vast = lambda: v
    deploy.cmd_sweep(BASE_CFG, args(dry_run=True))
    assert not v.made("destroy_instance")


@case
def sweep_refuses_without_worker_list():
    """if the autoscaler can't list workers, sweep destroys nothing"""
    v = deployed()
    v.instances = [inst(1), inst(2)]
    v.fail.add("get_endpoint_workers")
    deploy.vast = lambda: v
    raises(lambda: deploy.cmd_sweep(BASE_CFG, args()), deploy.ApiError)
    assert not v.made("destroy_instance")


@case
def sweep_without_endpoint_treats_all_as_orphans():
    """with no endpoint left, every marked instance is an orphan"""
    v = FakeVast()
    v.instances = [inst(7), inst(8, env=False)]
    deploy.vast = lambda: v
    deploy.cmd_sweep(BASE_CFG, args())
    assert [c[1] for c in v.made("destroy_instance")] == [7]


@case
def sweep_reports_survivors():
    """an instance that won't go away makes sweep fail loudly"""
    v = deployed()
    v.instances = [inst(2)]
    v.sticky.add(2)
    deploy.vast = lambda: v
    raises(lambda: deploy.cmd_sweep(BASE_CFG, args()), deploy.CheckFailed, "still exist")


@case
def legacy_and_template_ownership():
    """instances without the marker are recognised by old env vars or a recorded template"""
    st = {"template_hashes": ["tplX"]}
    legacy = {"id": 1, "extra_env": [["ORCH_REF", "abc"], ["SERVED_MODEL_NAME", BASE_CFG["model"]["served_name"]]]}
    by_tpl = {"id": 2, "template_hash_id": "tplX", "extra_env": {}}
    stranger = {"id": 3, "extra_env": {"ORCH_REF": "abc", "SERVED_MODEL_NAME": "other"}}
    assert deploy.owned(legacy, BASE_CFG, st) and deploy.owned(by_tpl, BASE_CFG, st)
    assert not deploy.owned(stranger, BASE_CFG, st)


# ── destroy ───────────────────────────────────────────────────────────────────
@case
def destroy_removes_instances_and_confirms():
    """destroy deletes group + endpoint, then destroys and confirms every instance"""
    v = deployed()
    v.instances = [inst(1), inst(2, age=10), inst(9, env=False)]
    v.workers = [1]
    deploy.vast = lambda: v
    deploy.cmd_destroy(BASE_CFG, args())
    assert v.made("delete_workergroup") and v.made("delete_endpoint")
    assert sorted(c[1] for c in v.made("destroy_instance")) == [1, 2]
    assert [i["id"] for i in v.instances] == [9]
    assert "endpoint_id" not in deploy.load_state()


@case
def destroy_follows_renamed_endpoint():
    """destroy finds the endpoint via state.json even after a rename"""
    v = deployed()
    v.endpoints[0]["endpoint_name"] = "renamed"
    deploy.vast = lambda: v
    deploy.cmd_destroy(BASE_CFG, args())
    assert v.made("delete_endpoint") and not v.endpoints


@case
def destroy_without_endpoint_still_cleans_instances():
    """endpoint already gone: destroy still destroys leftover instances"""
    v = FakeVast()
    v.instances = [inst(4)]
    deploy.vast = lambda: v
    deploy.cmd_destroy(BASE_CFG, args())
    assert [c[1] for c in v.made("destroy_instance")] == [4] and not v.instances


@case
def destroy_reports_survivors():
    """an instance that survives destroy is reported, and state is kept"""
    v = deployed()
    v.instances = [inst(1)]
    v.sticky.add(1)
    deploy.vast = lambda: v
    raises(lambda: deploy.cmd_destroy(BASE_CFG, args()), deploy.CheckFailed, "still exist")
    assert deploy.load_state().get("endpoint_id")


@case
def docker_options_carry_marker():
    """the template env carries the ORCH_DEPLOYMENT ownership marker"""
    assert f"-e ORCH_DEPLOYMENT={NAME}" in deploy.docker_options(BASE_CFG, PINS)


print(f"---- {len(PASSED)} passed, {len(FAILED)} failed")
sys.exit(1 if FAILED else 0)

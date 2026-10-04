#!/usr/bin/env python3
"""Download budget checks in worker/fetch_model.py, with the speeds real
rentals saw. Run: python3 tests/test_fetch.py"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
import fetch_model as fm  # noqa: E402

fm.log = lambda msg: None
MB, GiB = 1e6, 1024**3
PASSED, FAILED = [], []


def case(fn):
    try:
        fn()
        PASSED.append(fn.__name__)
        print(f"PASS  {fn.__doc__}")
    except Exception as e:
        FAILED.append(fn.__name__)
        print(f"FAIL  {fn.__doc__}: {type(e).__name__}: {e}")
    return fn


@case
def ramping_download_is_kept():
    """rental 54166295 (8.5, 19.4, 20.6 MB/s): kept on its last-minute speed, though its average was 16 MB/s"""
    left = 15.6 * GiB + 1.3 * GiB - 1.4 * GiB   # chat model so far, plus mmproj and embeddings still to come
    assert fm.too_slow(90, left, 20.6 * MB, max_s=3600, min_rate=0) is None


@case
def steady_slow_download_within_budget_is_kept():
    """rental 54167523 (6.2, 8.4, 7.9 MB/s): ~36 min projected against a 60 min budget is kept"""
    assert fm.too_slow(90, 16.8e9 - 0.7e9, 7.9 * MB, max_s=3600, min_rate=0) is None


@case
def host_that_would_miss_the_budget_fails():
    """2 MB/s for 17 GB would take ~2.4 h: fails early and says how long it would take"""
    why = fm.too_slow(90, 17e9, 2 * MB, max_s=3600, min_rate=0)
    assert why and "past DOWNLOAD_MAX_S=3600" in why and "min more" in why, why
    assert fm.too_slow(90, 17e9, 0, max_s=3600, min_rate=0), "no data at all must fail"


@case
def budget_already_spent_counts():
    """time already spent counts: 50 min in, 10 GB left at 20 MB/s (8 min) misses a 55 min budget"""
    assert fm.too_slow(50 * 60, 10e9, 20 * MB, max_s=55 * 60, min_rate=0)


@case
def optional_floor_still_works():
    """DOWNLOAD_MIN_MBPS, when set, is a floor on the recent speed"""
    why = fm.too_slow(90, 1e9, 4 * MB, max_s=3600, min_rate=5 * MB)
    assert why and "below DOWNLOAD_MIN_MBPS=5" in why, why


def ramping_download(recent_s):
    """20 KB in the first two seconds, then 1 MB/s, 5 MB in all, against a 10 s
    budget, checked from 3 s: the average then projects ~15 s, the last
    second's speed ~7 s."""
    with tempfile.TemporaryDirectory() as d:
        dest = os.path.join(d, "f")
        writer = ("import sys,time\n"
                  "f=open(sys.argv[1],'wb')\n"
                  "f.write(b'x'*20000); f.flush(); time.sleep(2)\n"
                  "for _ in range(50): f.write(b'x'*100000); f.flush(); time.sleep(0.1)\n")
        saved = fm.PROBE_S, fm.RECENT_S, fm.MAX_S, fm.MIN_RATE, fm.STARTED, fm.PROGRESS_S
        fm.PROBE_S, fm.RECENT_S, fm.MAX_S, fm.MIN_RATE, fm.PROGRESS_S = 3, recent_s, 10, 0, 1
        fm.STARTED = time.time()
        fm.LATER[0] = 0
        try:
            return fm.run_with_progress([sys.executable, "-c", writer, dest], lambda: fm.on_disk(dest),
                                        "f", 5_020_000)
        finally:
            fm.PROBE_S, fm.RECENT_S, fm.MAX_S, fm.MIN_RATE, fm.STARTED, fm.PROGRESS_S = saved


@case
def run_with_progress_judges_recent_speed():
    """a download that starts slow and speeds up is kept on its recent speed; judged on its average it would fail"""
    assert ramping_download(recent_s=1) == 0
    try:
        ramping_download(recent_s=100)
    except fm.TooSlow:
        return
    raise AssertionError("with the whole download as the window, the slow start should have failed it")


print(f"---- {len(PASSED)} passed, {len(FAILED)} failed")
sys.exit(1 if FAILED else 0)

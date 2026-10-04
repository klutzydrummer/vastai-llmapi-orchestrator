#!/usr/bin/env python3
"""Download and verify the GGUF weights a worker needs.

Every file is checked against the size and sha256 that the Hugging Face Hub
reports for the pinned revision, so a truncated download, a wrong filename or a
re-uploaded file can never reach llama-server. A verified file gets a small
marker next to it; on a cold restart the marker plus a size check is enough and
the hash is not recomputed.

Configuration (environment):
  MODEL_REPO, MODEL_FILE, MODEL_REVISION      main GGUF (revision defaults to main)
  MMPROJ_REPO, MMPROJ_FILE, MMPROJ_REVISION   vision projector (optional; repo
                                              defaults to MODEL_REPO)
  EMBED_REPO, EMBED_FILE, EMBED_REVISION      embedding model (optional)
  MODELS_DIR            where files go (default /workspace/models)
  PATHS_ENV             where to write MODEL_PATH=/MMPROJ_PATH=/EMBED_PATH= (default
                        /workspace/orch/paths.env)
  HF_TOKEN              optional, for gated/private repos
  HF_ENDPOINT           default https://huggingface.co
  DOWNLOAD_MAX_ATTEMPTS default 5
  DOWNLOAD_CONNECTIONS  aria2c connections per file, default 16
  MIN_FREE_GB_AFTER     free space to leave on disk, default 2
  DOWNLOAD_PROGRESS_S   seconds between progress lines, default 30
  DOWNLOAD_METHOD       auto (default): huggingface_hub + hf_xet when HF_PYTHON
                        is set, else aria2c/curl; hf; direct (aria2c/curl only)
  HF_PYTHON             python with huggingface_hub[hf_xet] (boot.sh sets it)
  DOWNLOAD_MIN_MBPS     give up when a file's average speed after
                        DOWNLOAD_PROBE_S (default 90) seconds is below this many
                        MB/s, default 25; 0 = no minimum. Another host may be faster.
  DOWNLOAD_STALL_S      restart an attempt that has received nothing for this
                        long, default 120
  DOWNLOAD_MAX_S        give up when all downloads together take longer, default 3600

Exit status: 0 ok, 2 bad configuration / file not found on the Hub,
3 not enough disk, 4 download failed, 5 verification failed, 6 too slow.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
MODELS_DIR = os.environ.get("MODELS_DIR", "/workspace/models")
PATHS_ENV = os.environ.get("PATHS_ENV", "/workspace/orch/paths.env")
MAX_ATTEMPTS = int(os.environ.get("DOWNLOAD_MAX_ATTEMPTS", "5"))
CONNECTIONS = max(1, min(16, int(os.environ.get("DOWNLOAD_CONNECTIONS", "16"))))
MIN_FREE_AFTER = float(os.environ.get("MIN_FREE_GB_AFTER", "2")) * 1024**3
PROGRESS_S = max(1.0, float(os.environ.get("DOWNLOAD_PROGRESS_S", "30")))
METHOD = os.environ.get("DOWNLOAD_METHOD", "auto").strip() or "auto"
HF_PYTHON = os.environ.get("HF_PYTHON", "").strip()
HF_DOWNLOAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_download.py")
MIN_RATE = float(os.environ.get("DOWNLOAD_MIN_MBPS", "25")) * 1e6
PROBE_S = float(os.environ.get("DOWNLOAD_PROBE_S", "90"))
STALL_S = float(os.environ.get("DOWNLOAD_STALL_S", "120"))
MAX_S = float(os.environ.get("DOWNLOAD_MAX_S", "3600"))
STARTED = time.time()

EXIT_CONFIG, EXIT_DISK, EXIT_DOWNLOAD, EXIT_VERIFY, EXIT_SLOW = 2, 3, 4, 5, 6


class FetchError(Exception):
    def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code


class TooSlow(FetchError):
    """Not retried: the same host will be just as slow on the next attempt."""
    def __init__(self, msg):
        super().__init__(msg, EXIT_SLOW)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] [fetch] {msg}", flush=True)


def _headers():
    h = {"User-Agent": "vastai-llmapi-orchestrator/1"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h


def _http_json(url, data=None, timeout=30):
    req = urllib.request.Request(url, data=data, headers=_headers())
    if data is not None:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def hub_file_info(repo, revision, path):
    """Return (size, sha256) the Hub reports for repo@revision:path."""
    q_repo = urllib.parse.quote(repo, safe="/")
    q_rev = urllib.parse.quote(revision, safe="")
    last = None
    for attempt in range(1, 4):
        try:
            items = _http_json(
                f"{HF_ENDPOINT}/api/models/{q_repo}/paths-info/{q_rev}",
                data=urllib.parse.urlencode({"paths": path}).encode(),
            )
            break
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise FetchError(f"{repo}@{revision}: HTTP {e.code} from the Hub "
                                 f"(missing repo/revision, or HF_TOKEN needed)", EXIT_CONFIG)
            last = e
        except Exception as e:  # network blip
            last = e
        time.sleep(3 * attempt)
    else:
        raise FetchError(f"could not query the Hub for {repo}: {last}", EXIT_DOWNLOAD)

    for it in items if isinstance(items, list) else []:
        if it.get("path") == path and it.get("type", "file") == "file":
            lfs = it.get("lfs") or {}
            sha = lfs.get("oid") or lfs.get("sha256")
            size = lfs.get("size", it.get("size"))
            if not sha or not size:
                raise FetchError(f"{repo}:{path} is not an LFS file; cannot verify it", EXIT_CONFIG)
            return int(size), sha.lower()
    raise FetchError(f"{path} does not exist in {repo}@{revision} "
                     f"(check the exact filename, including case)", EXIT_CONFIG)


def sha256_of(path):
    log(f"hashing {os.path.basename(path)} ({os.path.getsize(path) / 1024**3:.2f} GiB)")
    h = hashlib.sha256()
    t0 = time.time()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    log(f"hashed {os.path.basename(path)} in {time.time() - t0:.0f}s")
    return h.hexdigest()


def is_gguf(path):
    with open(path, "rb") as f:
        return f.read(4) == b"GGUF"


def marker_path(dest):
    return dest + ".verified"


def marker_ok(dest, size, sha):
    try:
        m = json.load(open(marker_path(dest)))
        return m.get("sha256") == sha and m.get("size") == size and os.path.getsize(dest) == size
    except Exception:
        return False


def write_marker(dest, size, sha):
    with open(marker_path(dest), "w") as f:
        json.dump({"sha256": sha, "size": size, "verified_at": int(time.time())}, f)


def on_disk(path):
    """Bytes actually written: aria2c writes parts at their offsets, so the
    file's apparent size jumps ahead of what has arrived."""
    try:
        st = os.stat(path)
        return min(st.st_size, st.st_blocks * 512)
    except OSError:
        return 0


def partial_bytes(local_dir):
    """Bytes huggingface_hub has written so far: it downloads into
    LOCAL_DIR/.cache/huggingface/download/*.incomplete and renames at the end."""
    total = 0
    for root, _, files in os.walk(os.path.join(local_dir, ".cache", "huggingface", "download")):
        total += sum(on_disk(os.path.join(root, f)) for f in files if f.endswith(".incomplete"))
    return total


def _stop(p):
    p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def run_with_progress(cmd, measure, name, size):
    """Run a downloader. Logs progress every PROGRESS_S seconds, restarts an
    attempt that has stalled, and gives up early on a host too slow to finish
    in reasonable time instead of downloading until the boot deadline."""
    p = subprocess.Popen(cmd)
    t0 = last_t = moved_t = time.time()
    start_b = last_b = moved_b = measure()
    tick = min(5.0, PROGRESS_S)
    while True:
        try:
            return p.wait(timeout=tick)
        except subprocess.TimeoutExpired:
            pass
        now, got = time.time(), measure()
        avg = (got - start_b) / max(now - t0, 1e-6)
        if got > moved_b:
            moved_t, moved_b = now, got
        if now - last_t >= PROGRESS_S:
            rate = (got - last_b) / max(now - last_t, 1e-6)
            left = f", ~{(size - got) / avg / 60:.0f} min left" if avg > 0 else ""
            log(f"{name}: {got / 1024**3:.2f}/{size / 1024**3:.2f} GiB "
                f"({100 * got / max(size, 1):.0f}%), {rate / 1e6:.1f} MB/s{left}")
            last_t, last_b = now, got
        if now - moved_t >= STALL_S:
            _stop(p)
            raise FetchError(f"{name}: no data for {STALL_S:.0f}s, restarting the download", EXIT_DOWNLOAD)
        if MIN_RATE > 0 and now - t0 >= PROBE_S and avg < MIN_RATE:
            _stop(p)
            raise TooSlow(f"{name}: {avg / 1e6:.1f} MB/s average over {now - t0:.0f}s is below "
                          f"DOWNLOAD_MIN_MBPS={MIN_RATE / 1e6:g}; the remaining "
                          f"{(size - got) / 1024**3:.1f} GiB would take ~{(size - got) / max(avg, 1) / 60:.0f} min "
                          f"on this host")
        if now - STARTED >= MAX_S:
            _stop(p)
            raise TooSlow(f"downloads have taken over DOWNLOAD_MAX_S={MAX_S:.0f}s")


def direct_cmd(url, dest):
    tmp_dir, name = os.path.dirname(dest), os.path.basename(dest)
    if shutil.which("aria2c"):
        # No per-connection speed floor (--lowest-speed-limit): Hugging Face's
        # CDN serves each connection slowly and aria2c dropped most of them
        # (error 5), leaving one or two. Stalls are caught above instead.
        cmd = ["aria2c", f"-x{CONNECTIONS}", f"-s{CONNECTIONS}", "-k1M", "-c",
               "--file-allocation=none", "--auto-file-renaming=false",
               "--allow-overwrite=true", "--summary-interval=0",
               "--console-log-level=warn", "--download-result=hide",
               "--max-tries=5", "--retry-wait=5", "--timeout=60",
               "--connect-timeout=30", "-d", tmp_dir, "-o", name]
        if HF_TOKEN:
            cmd += ["--header", f"Authorization: Bearer {HF_TOKEN}"]
        return cmd + [url]
    cmd = ["curl", "-fL", "--retry", "5", "--retry-delay", "5",
           "--connect-timeout", "30", "-C", "-", "-sS", "-o", dest]
    if HF_TOKEN:
        cmd += ["-H", f"Authorization: Bearer {HF_TOKEN}"]
    return cmd + [url]


def use_hf():
    if METHOD == "direct":
        return False
    ok = bool(HF_PYTHON) and os.access(HF_PYTHON, os.X_OK)
    if METHOD == "hf" and not ok:
        raise FetchError(f"DOWNLOAD_METHOD=hf but HF_PYTHON ({HF_PYTHON or 'unset'}) is not runnable", EXIT_CONFIG)
    return ok


def download(repo, revision, path, url, dest, size, hf):
    name = os.path.basename(dest)
    if hf:
        local_dir = os.path.join(MODELS_DIR, repo.replace("/", "__"))
        # A partial file from an earlier direct attempt is no use to hf_xet;
        # free its space, and drop leftovers from killed attempts.
        for f in (dest, dest + ".aria2"):
            if os.path.exists(f):
                os.remove(f)
        dl = os.path.join(local_dir, ".cache", "huggingface", "download")
        for root, _, files in os.walk(dl):
            for f in files:
                if f.endswith(".incomplete"):
                    os.remove(os.path.join(root, f))
        cmd = [HF_PYTHON, HF_DOWNLOAD, repo, path, revision, local_dir]
        rc = run_with_progress(cmd, lambda: partial_bytes(local_dir) + on_disk(dest), name, size)
    else:
        rc = run_with_progress(direct_cmd(url, dest), lambda: on_disk(dest), name, size)
    got = os.path.getsize(dest) if os.path.exists(dest) else 0
    if rc != 0 or got != size:
        raise FetchError(f"download of {name} incomplete (exit {rc}, {got}/{size} bytes)", EXIT_DOWNLOAD)


def fetch(repo, revision, path):
    size, sha = hub_file_info(repo, revision, path)
    dest = os.path.join(MODELS_DIR, repo.replace("/", "__"), path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    gb = size / 1024**3
    log(f"{repo}@{revision[:12]}:{path} — {gb:.2f} GiB, sha256 {sha[:12]}…")

    if marker_ok(dest, size, sha):
        log(f"{path}: already verified, skipping")
        return dest

    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have > size:
        log(f"{path}: local file is larger than expected, discarding")
        os.remove(dest)
        have = 0
    if have == size and not os.path.exists(dest + ".aria2"):
        if sha256_of(dest) == sha:
            write_marker(dest, size, sha)
            log(f"{path}: existing file verified")
            return dest
        log(f"{path}: existing file failed verification, re-downloading")
        os.remove(dest)
        have = 0

    free = shutil.disk_usage(os.path.dirname(dest)).free
    if free - (size - have) < MIN_FREE_AFTER:
        raise FetchError(f"not enough disk for {path}: need {(size - have) / 1024**3:.1f} GiB "
                         f"plus headroom, {free / 1024**3:.1f} GiB free", EXIT_DISK)

    url = (f"{HF_ENDPOINT}/{urllib.parse.quote(repo, safe='/')}/resolve/"
           f"{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(path)}")
    hf, hf_failures = use_hf(), 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        how = "huggingface_hub/hf_xet" if hf else ("aria2c" if shutil.which("aria2c") else "curl")
        try:
            t0 = time.time()
            log(f"{path}: downloading with {how}")
            download(repo, revision, path, url, dest, size, hf)
            dt = time.time() - t0
            fetched = size if hf else size - have   # hf_xet starts from zero
            log(f"{path}: downloaded in {dt:.0f}s ({fetched / max(dt, 1e-6) / 1e6:.1f} MB/s)")
            break
        except TooSlow:
            raise
        except FetchError as e:
            if attempt == MAX_ATTEMPTS:
                raise
            if hf:
                hf_failures += 1
                if hf_failures >= 2 and METHOD != "hf":
                    hf = False
                    log(f"{path}: huggingface_hub failed twice, switching to direct download")
            wait = min(60, 5 * 2 ** (attempt - 1))
            log(f"{e}; retry {attempt}/{MAX_ATTEMPTS - 1} in {wait}s")
            time.sleep(wait)

    if not is_gguf(dest):
        os.remove(dest)
        raise FetchError(f"{path} is not a GGUF file (bad magic)", EXIT_VERIFY)
    if sha256_of(dest) != sha:
        os.remove(dest)
        raise FetchError(f"{path} sha256 mismatch after download; file removed", EXIT_VERIFY)
    write_marker(dest, size, sha)
    log(f"{path}: verified")
    return dest


def main():
    model_repo = os.environ.get("MODEL_REPO", "").strip()
    model_file = os.environ.get("MODEL_FILE", "").strip()
    if not model_repo or not model_file:
        raise FetchError("MODEL_REPO and MODEL_FILE must be set", EXIT_CONFIG)
    model_rev = os.environ.get("MODEL_REVISION", "").strip() or "main"
    mm_file = os.environ.get("MMPROJ_FILE", "").strip()
    mm_repo = os.environ.get("MMPROJ_REPO", "").strip() or model_repo
    mm_rev = os.environ.get("MMPROJ_REVISION", "").strip() or "main"

    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {"MODEL_PATH": fetch(model_repo, model_rev, model_file)}
    if mm_file:
        paths["MMPROJ_PATH"] = fetch(mm_repo, mm_rev, mm_file)
    embed_file = os.environ.get("EMBED_FILE", "").strip()
    if embed_file:
        embed_repo = os.environ.get("EMBED_REPO", "").strip()
        if not embed_repo:
            raise FetchError("EMBED_FILE is set but EMBED_REPO is not", EXIT_CONFIG)
        embed_rev = os.environ.get("EMBED_REVISION", "").strip() or "main"
        paths["EMBED_PATH"] = fetch(embed_repo, embed_rev, embed_file)

    os.makedirs(os.path.dirname(PATHS_ENV), exist_ok=True)
    with open(PATHS_ENV, "w") as f:
        for k, v in paths.items():
            f.write(f"{k}={v}\n")
    log("all files present and verified")


if __name__ == "__main__":
    try:
        main()
    except FetchError as e:
        log(f"FAILED: {e}")
        sys.exit(e.code)

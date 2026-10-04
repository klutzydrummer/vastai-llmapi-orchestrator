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

Exit status: 0 ok, 2 bad configuration / file not found on the Hub,
3 not enough disk, 4 download failed, 5 verification failed.
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

EXIT_CONFIG, EXIT_DISK, EXIT_DOWNLOAD, EXIT_VERIFY = 2, 3, 4, 5


class FetchError(Exception):
    def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code


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


def run_with_progress(cmd, dest, size):
    """Run the downloader, logging how far it has got every PROGRESS_S seconds,
    so a slow or stalled download shows up in the log while it happens."""
    p = subprocess.Popen(cmd)
    t0 = last_t = time.time()
    last_b = start_b = on_disk(dest)
    while True:
        try:
            return p.wait(timeout=PROGRESS_S)
        except subprocess.TimeoutExpired:
            pass
        now, got = time.time(), on_disk(dest)
        rate = (got - last_b) / max(now - last_t, 1e-6)
        avg = (got - start_b) / max(now - t0, 1e-6)
        left = f", ~{(size - got) / avg / 60:.0f} min left" if avg > 0 else ""
        log(f"{os.path.basename(dest)}: {got / 1024**3:.2f}/{size / 1024**3:.2f} GiB "
            f"({100 * got / max(size, 1):.0f}%), {rate / 1e6:.1f} MB/s{left}")
        last_t, last_b = now, got


def download(url, dest, size):
    tmp_dir, name = os.path.dirname(dest), os.path.basename(dest)
    if shutil.which("aria2c"):
        cmd = ["aria2c", f"-x{CONNECTIONS}", f"-s{CONNECTIONS}", "-k1M", "-c",
               "--file-allocation=none", "--auto-file-renaming=false",
               "--allow-overwrite=true", "--summary-interval=0",
               "--console-log-level=warn", "--download-result=hide",
               "--max-tries=5", "--retry-wait=5", "--timeout=60",
               "--connect-timeout=30", "--lowest-speed-limit=512K",
               "-d", tmp_dir, "-o", name]
        if HF_TOKEN:
            cmd += ["--header", f"Authorization: Bearer {HF_TOKEN}"]
        cmd.append(url)
    else:
        cmd = ["curl", "-fL", "--retry", "5", "--retry-delay", "5",
               "--connect-timeout", "30", "--speed-limit", "524288",
               "--speed-time", "60", "-C", "-", "-sS", "-o", dest]
        if HF_TOKEN:
            cmd += ["-H", f"Authorization: Bearer {HF_TOKEN}"]
        cmd.append(url)
    rc = run_with_progress(cmd, dest, size)
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
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            t0 = time.time()
            download(url, dest, size)
            log(f"{path}: downloaded in {time.time() - t0:.0f}s")
            break
        except FetchError as e:
            if attempt == MAX_ATTEMPTS:
                raise
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

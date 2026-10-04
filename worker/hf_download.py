#!/usr/bin/env python3
"""Download one file from the Hugging Face Hub with huggingface_hub + hf_xet.

Usage: hf_download.py REPO PATH REVISION LOCAL_DIR [PROGRESS_FILE]

Run by fetch_model.py with the venv boot.sh builds (HF_PYTHON). Large GGUF
files on the Hub are stored with Xet; hf_xet fetches their chunks in parallel
straight from Xet storage, which Hugging Face recommends over plain HTTP range
requests (those go through a CDN bridge that throttles single connections).

The file lands at LOCAL_DIR/PATH. fetch_model.py still checks its size and
sha256 against what the Hub reports for the pinned revision; this script only
moves bytes. HF_TOKEN in the environment is picked up by huggingface_hub.

hf_xet holds chunks in memory and writes the file late and out of order, so
the file's size says little about how fast data is arriving. With
PROGRESS_FILE, this script writes the bytes huggingface_hub reports as
received there (an integer, about once a second) for fetch_model.py's speed
and stall checks.
"""

import os
import sys
import threading
import time

# Before the import: these are read when huggingface_hub loads.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # fetch_model.py logs progress
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")  # no second copy of the weights on disk

from huggingface_hub import hf_hub_download  # noqa: E402
from huggingface_hub.utils.tqdm import tqdm as hf_tqdm  # noqa: E402

PROGRESS_FILE = sys.argv[5] if len(sys.argv) > 5 else ""
_lock = threading.Lock()
_counts = {}            # progress bar -> bytes it has been told about
_last_write = [0.0]


def _write_progress(force=False):
    now = time.time()
    if not PROGRESS_FILE or (not force and now - _last_write[0] < 1):
        return
    _last_write[0] = now
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(max(_counts.values(), default=0)))
    os.replace(tmp, PROGRESS_FILE)


class CountingTqdm(hf_tqdm):
    """huggingface_hub's progress bar class, counting every update even when
    the bar itself is disabled (a disabled tqdm ignores update()). For an Xet
    file huggingface_hub drives two bars: bytes received and bytes written;
    for a plain HTTP download, one. The larger count is the progress."""

    def __init__(self, *args, **kwargs):
        with _lock:
            _counts[id(self)] = 0
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        with _lock:
            _counts[id(self)] = _counts.get(id(self), 0) + (n or 0)
            _write_progress()
        return super().update(n)


def main():
    repo, path, revision, local_dir = sys.argv[1:5]
    out = hf_hub_download(repo_id=repo, filename=path, revision=revision, local_dir=local_dir,
                          tqdm_class=CountingTqdm)
    with _lock:
        _write_progress(force=True)
    print(out, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download one file from the Hugging Face Hub with huggingface_hub + hf_xet.

Usage: hf_download.py REPO PATH REVISION LOCAL_DIR

Run by fetch_model.py with the venv boot.sh builds (HF_PYTHON). Large GGUF
files on the Hub are stored with Xet; hf_xet fetches their chunks in parallel
straight from Xet storage, which Hugging Face recommends over plain HTTP range
requests (those go through a CDN bridge that throttles single connections).

The file lands at LOCAL_DIR/PATH. fetch_model.py still checks its size and
sha256 against what the Hub reports for the pinned revision; this script only
moves bytes. HF_TOKEN in the environment is picked up by huggingface_hub.
"""

import os
import sys

# Before the import: these are read when huggingface_hub loads.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # fetch_model.py logs progress
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")  # no second copy of the weights on disk

from huggingface_hub import hf_hub_download  # noqa: E402


def main():
    repo, path, revision, local_dir = sys.argv[1:5]
    out = hf_hub_download(repo_id=repo, filename=path, revision=revision, local_dir=local_dir)
    print(out, flush=True)


if __name__ == "__main__":
    main()

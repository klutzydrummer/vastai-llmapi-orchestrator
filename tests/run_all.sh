#!/usr/bin/env bash
# Runs every offline test. Needs python3 with vastai==1.8.3 and aiohttp.
set -e
cd "$(dirname "$0")/.."
bash -n worker/boot.sh worker/onstart.sh
python3 -m py_compile worker/fetch_model.py worker/smoke_test.py shim/shim.py deploy/deploy.py
bash tests/test_boot.sh
python3 tests/test_shim.py

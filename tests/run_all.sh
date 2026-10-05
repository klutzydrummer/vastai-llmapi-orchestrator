#!/usr/bin/env bash
# Runs every offline test. Needs bash, python3 with vastai==1.8.3 and aiohttp
# (shim/requirements.txt), and curl, git, openssl, pkill (procps), setsid
# (util-linux) and timeout (coreutils). In Docker: docker compose run --rm test
set -e
cd "$(dirname "$0")/.."

missing=()
for b in python3 curl git openssl pkill setsid timeout; do
    command -v "$b" >/dev/null 2>&1 || missing+=("$b")
done
python3 -c 'import vastai, aiohttp' 2>/dev/null || missing+=("python: pip install -r shim/requirements.txt")
if [ ${#missing[@]} -gt 0 ]; then
    printf 'missing test prerequisites:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 2
fi

# Each suite gets a hard limit so a broken environment fails instead of hanging.
run(){ local limit="$1"; shift
    timeout --kill-after=10 "$limit" "$@" || { rc=$?; [ $rc -eq 124 ] && echo "TIMEOUT after ${limit}s: $*" >&2; exit $rc; }
}
bash -n worker/boot.sh worker/onstart.sh
python3 -m py_compile worker/fetch_model.py worker/smoke_test.py worker/vram.py worker/router.py shim/shim.py deploy/deploy.py
run 600 bash tests/test_boot.sh
run 120 python3 tests/test_shim.py
run 120 python3 tests/test_deploy.py
run 60 python3 tests/test_fetch.py
run 60 python3 tests/test_vram.py
run 60 python3 tests/test_smoke.py

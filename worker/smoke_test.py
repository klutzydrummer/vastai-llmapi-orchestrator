#!/usr/bin/env python3
"""Prove the loaded model actually works before the worker takes traffic.

Runs against the local llama-server (default http://127.0.0.1:18000):
  1. /props  — when the server reports modalities, vision must be on if an
               mmproj was configured (catches a projector that silently failed).
  2. text    — a short chat completion must return non-empty text.
  3. image   — a chat completion with an inline PNG must succeed and return
               text. The image is a solid red square; if the answer doesn't
               mention red that is logged as a warning, or treated as a failure
               with SMOKE_IMAGE_STRICT=1.

Exit 0 when everything passes, 1 otherwise. Output is plain log lines.
"""

import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

BASE = os.environ.get("LLAMA_URL", "http://127.0.0.1:18000").rstrip("/")
EXPECT_VISION = bool(os.environ.get("MMPROJ_PATH", "").strip())
STRICT_IMAGE = os.environ.get("SMOKE_IMAGE_STRICT", "0") == "1"
TIMEOUT = float(os.environ.get("SMOKE_TIMEOUT", "180"))
MODEL = os.environ.get("SERVED_MODEL_NAME", "model")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] [smoke] {msg}", flush=True)


def red_png(n=64):
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * n for _ in range(n))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode()[:500]
        except Exception:
            detail = ""
        return e.code, {"_error": detail}


def reply_text(resp):
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    return ((msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")).strip()


def check_props():
    status, props = call("/props")
    if status != 200:
        log(f"/props returned {status}; skipping modality check")
        return True
    mods = props.get("modalities")
    if isinstance(mods, dict) and EXPECT_VISION and not mods.get("vision"):
        log("FAIL: an mmproj is configured but the server reports vision disabled")
        return False
    log(f"props ok (modalities={mods})")
    return True


def check_text():
    t0 = time.time()
    status, resp = call("/v1/chat/completions", {
        "model": MODEL, "temperature": 0, "max_tokens": 24,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
    })
    text = reply_text(resp)
    if status != 200 or not text:
        log(f"FAIL: text request -> HTTP {status}: {resp.get('_error') or str(resp)[:300]}")
        return False
    log(f"text ok in {time.time() - t0:.1f}s: {text[:80]!r}")
    return True


def check_image():
    t0 = time.time()
    url = "data:image/png;base64," + base64.b64encode(red_png()).decode()
    status, resp = call("/v1/chat/completions", {
        "model": MODEL, "temperature": 0, "max_tokens": 24,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "What single color fills this image? Answer with one word."},
        ]}],
    })
    text = reply_text(resp)
    if status != 200 or not text:
        log(f"FAIL: image request -> HTTP {status}: {resp.get('_error') or str(resp)[:300]}")
        return False
    if "red" not in text.lower():
        if STRICT_IMAGE:
            log(f"FAIL: image answer did not mention red: {text[:80]!r}")
            return False
        log(f"warning: image answer did not mention red: {text[:80]!r}")
    log(f"image ok in {time.time() - t0:.1f}s: {text[:80]!r}")
    return True


def main():
    ok = check_props() and check_text()
    if ok and EXPECT_VISION:
        ok = check_image()
    log("all checks passed" if ok else "checks failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

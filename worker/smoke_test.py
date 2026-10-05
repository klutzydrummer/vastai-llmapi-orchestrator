#!/usr/bin/env python3
"""Prove the loaded model actually works before the worker takes traffic.

Runs against the local llama-server (default http://127.0.0.1:18000):
  1. /props  — when the server reports modalities, vision must be on if an
               mmproj was configured (catches a projector that silently failed).
  2. text    — two short chat completions must each return clean text: not
               empty, no chat-template markup (<|...|>, <start_of_turn>, ...)
               and not starting with a role name, which is what a template or
               thinking-mode mismatch leaks into replies (rental 54245444:
               '<|thought|>\n', 'user\n...'). On such a reply the prompt as
               the server renders it (/apply-template) is logged.
  3. image   — a chat completion with an inline PNG must succeed and return
               clean text. The image is a solid red square; if the answer doesn't
               mention red that is logged as a warning, or treated as a failure
               with SMOKE_IMAGE_STRICT=1.
  4. embedding — when EMBED_SERVED_NAME is set, /v1/embeddings must return
               one finite, non-zero vector per input, all the same length.

Exit 0 when everything passes, 1 otherwise. Output is plain log lines.
"""

import base64
import json
import math
import os
import re
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
EMBED_MODEL = os.environ.get("EMBED_SERVED_NAME", "").strip()


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
    """The reply's content. Reasoning (reasoning_content) doesn't count: with
    reasoning_budget 0 the answer must be in the content."""
    try:
        return resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


# Chat-template markup that must never reach a client: <|...|> tokens,
# Gemma's <start_of_turn>/<end_of_turn>, <|channel>-style openers, think tags.
MARKUP = re.compile(r"<\|[^<>\s]{0,40}\|?>|<(start|end)_of_turn>|</?(think|thought)>", re.I)
ROLE_PREFIX = re.compile(r"^\s*(user|model|assistant|system)\s*(\n|:)", re.I)


def garbled(text):
    """Why a reply looks like leaked template or thinking output, or ""."""
    if not text.strip():
        return "empty reply"
    m = MARKUP.search(text)
    if m:
        return f"chat-template markup {m.group(0)!r} in the reply"
    if ROLE_PREFIX.match(text):
        return "reply starts with a role name"
    if not re.search(r"[^\W_]", text):
        return "no words in the reply"
    return ""


def show_template(messages):
    """Log the prompt as the server renders it, to diagnose a garbled reply."""
    status, resp = call("/apply-template", {"messages": messages})
    if status == 200 and isinstance(resp.get("prompt"), str):
        log(f"the server renders this prompt as: {resp['prompt'][-400:]!r}")


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
    messages = [{"role": "user", "content": "Reply with the single word: pong"}]
    # Twice: rental 54245444's first reply leaked a thinking token, later ones didn't.
    for n in (1, 2):
        t0 = time.time()
        status, resp = call("/v1/chat/completions", {
            "model": MODEL, "temperature": 0, "max_tokens": 24, "messages": messages})
        if status != 200:
            log(f"FAIL: text request -> HTTP {status}: {resp.get('_error') or str(resp)[:300]}")
            return False
        text = reply_text(resp)
        why = garbled(text)
        if why:
            log(f"FAIL: text request {n}: {why}: {text[:200]!r}")
            show_template(messages)
            return False
        if "pong" not in text.lower():
            log(f"warning: text reply {n} did not say pong: {text[:80]!r}")
        log(f"text {n} ok in {time.time() - t0:.1f}s: {text[:80]!r}")
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
    if status != 200:
        log(f"FAIL: image request -> HTTP {status}: {resp.get('_error') or str(resp)[:300]}")
        return False
    text = reply_text(resp)
    why = garbled(text)
    if why:
        log(f"FAIL: image request: {why}: {text[:200]!r}")
        return False
    if "red" not in text.lower():
        if STRICT_IMAGE:
            log(f"FAIL: image answer did not mention red: {text[:80]!r}")
            return False
        log(f"warning: image answer did not mention red: {text[:80]!r}")
    log(f"image ok in {time.time() - t0:.1f}s: {text[:80]!r}")
    return True


def check_embedding():
    t0 = time.time()
    inputs = ["The quick brown fox.", "A red square."]
    status, resp = call("/v1/embeddings", {"model": EMBED_MODEL, "input": inputs})
    try:
        vecs = [d["embedding"] for d in sorted(resp["data"], key=lambda d: d.get("index", 0))]
    except (KeyError, TypeError):
        vecs = []
    if status != 200 or len(vecs) != len(inputs):
        log(f"FAIL: embedding request -> HTTP {status}: {resp.get('_error') or str(resp)[:300]}")
        return False
    dims = {len(v) for v in vecs}
    bad = [v for v in vecs if not v or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
           or not any(v)]
    if len(dims) != 1 or bad:
        log(f"FAIL: embedding vectors are empty, non-finite, all zero or of different lengths ({sorted(dims)})")
        return False
    log(f"embedding ok in {time.time() - t0:.1f}s: {len(vecs)} x {dims.pop()} dims")
    return True


def main():
    ok = check_props() and check_text()
    if ok and EXPECT_VISION:
        ok = check_image()
    if ok and EMBED_MODEL:
        ok = check_embedding()
    log("all checks passed" if ok else "checks failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

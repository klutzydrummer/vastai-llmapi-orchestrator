#!/usr/bin/env python3
"""Prove the loaded model actually works before the worker takes traffic.

Runs against the local llama-server (default http://127.0.0.1:18000):
  1. /props  — when the server reports modalities, vision must be on if an
               mmproj was configured (catches a projector that silently failed).
  2. text    — a short chat completion must return a clean reply (below).
  3. image   — a chat completion with an inline PNG must succeed and return
               a clean reply. The image is a solid red square; if the answer
               doesn't mention red that is logged as a warning, or treated as
               a failure with SMOKE_IMAGE_STRICT=1.
  4. embedding — when EMBED_SERVED_NAME is set, /v1/embeddings must return
               one finite, non-zero vector per input, all the same length.

A clean reply has answer text (`content`) that contains no chat-template
tokens (<|thought|>, <start_of_turn>, <eos>, ...) and doesn't start with a role
name ("user\n", "model:", ...); those mean the template or the thinking
settings are wrong and every client would see the garbage. With thinking on
(LLAMA_REASONING_BUDGET other than 0) a reply that is still thinking when
max_tokens runs out is fine: the reasoning text is judged instead. When a
reply is refused, the raw reply and the tail of the prompt the server built
(/apply-template) are logged, so the boot log shows the cause.

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
THINKING = os.environ.get("LLAMA_REASONING_BUDGET", "0").strip() not in ("", "0")

# Chat-template tokens that must never reach a client: <|...|>, <|turn>,
# <turn|> (Gemma 4 / ChatML style) and Gemma's <start_of_turn>, <eos>, ...
TEMPLATE_TOKEN = re.compile(r"<\|[^<>\s]{0,40}>|<[^<>\s|]{1,40}\|>|<(?:start_of_turn|end_of_turn|bos|eos|pad)>")
ROLE_PREFIX = re.compile(r"\s*(?:user|model|assistant|system)\s*(?:\n|:)", re.IGNORECASE)


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


def judge(resp):
    """(text, problem) for a chat completion: problem is None for a clean
    reply, else what is wrong with it."""
    try:
        choice = resp["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError):
        return "", "no choices[0].message in the response"
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    text = content
    if not content.strip():
        if not (THINKING and reasoning.strip()):
            return "", ("empty answer; only reasoning came back although thinking is off "
                        "(reasoning_budget 0)" if reasoning.strip() else "empty answer")
        text = reasoning    # thinking on and cut off by max_tokens while still thinking
    tok = TEMPLATE_TOKEN.search(text)
    if tok:
        rest = TEMPLATE_TOKEN.sub("", text).strip()
        return text, (f"chat-template token {tok.group(0)!r} in the answer"
                      + ("" if rest else ", and nothing else"))
    m = ROLE_PREFIX.match(text)
    if m:
        return text, f"answer starts with a role name ({m.group(0).strip()!r})"
    return text.strip(), None


def explain(resp, messages):
    """Logs what a refused reply was made of: the raw message, why generation
    stopped, and the end of the prompt the chat template produced."""
    try:
        choice = resp["choices"][0]
        log(f"  raw message: {json.dumps(choice.get('message'))[:400]}")
        log(f"  finish_reason: {choice.get('finish_reason')!r}")
    except (KeyError, IndexError, TypeError):
        log(f"  raw response: {str(resp)[:400]}")
    status, tmpl = call("/apply-template", {"messages": messages})
    if status == 200 and isinstance(tmpl.get("prompt"), str):
        log(f"  prompt the template built ends with: {tmpl['prompt'][-200:]!r}")
    else:
        log(f"  /apply-template returned {status}; can't show the prompt")


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


def chat(messages):
    """Sends one chat completion; returns the clean reply text, or None after
    logging why it failed."""
    status, resp = call("/v1/chat/completions", {
        "model": MODEL, "temperature": 0, "max_tokens": 24, "messages": messages})
    if status != 200:
        log(f"FAIL: HTTP {status}: {resp.get('_error') or str(resp)[:300]}")
        return None
    text, problem = judge(resp)
    if problem:
        log(f"FAIL: garbled reply, {problem}: {text[:80]!r}")
        explain(resp, messages)
        return None
    return text


def check_text():
    t0 = time.time()
    text = chat([{"role": "user", "content": "Reply with the single word: pong"}])
    if text is None:
        log("FAIL: text request")
        return False
    log(f"text ok in {time.time() - t0:.1f}s: {text[:80]!r}")
    return True


def check_image():
    t0 = time.time()
    url = "data:image/png;base64," + base64.b64encode(red_png()).decode()
    text = chat([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "What single color fills this image? Answer with one word."},
    ]}])
    if text is None:
        log("FAIL: image request")
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

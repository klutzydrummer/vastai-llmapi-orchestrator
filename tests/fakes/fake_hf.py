#!/usr/bin/env python3
"""Tiny stand-in for the Hugging Face Hub: paths-info + resolve (with Range).

Usage: fake_hf.py PORT DIR [--corrupt NAME] [--slow SECS] [--stall-once] [--no-ranges]
Every file in DIR is served as repo "test/repo". With --corrupt NAME the Hub
reports a wrong sha256 for that file, to exercise verification failure. With
--slow SECS each download is spread over about SECS seconds. With
--stall-once the first download sends half the file, then hangs. With
--no-ranges bounded range requests (header reads) get a 403.
"""
import hashlib
import json
import os
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, ROOT = int(sys.argv[1]), sys.argv[2]
CORRUPT = sys.argv[sys.argv.index("--corrupt") + 1] if "--corrupt" in sys.argv else None
SLOW = float(sys.argv[sys.argv.index("--slow") + 1]) if "--slow" in sys.argv else 0
STALL = {"left": 1 if "--stall-once" in sys.argv else 0}
NO_RANGES = "--no-ranges" in sys.argv


def info(name):
    p = os.path.join(ROOT, name)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if name == CORRUPT:
        sha = "0" * 64
    size = os.path.getsize(p)
    return {"type": "file", "path": name, "size": size, "oid": "x",
            "lfs": {"oid": sha, "size": size, "pointerSize": 130}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if "/paths-info/" not in self.path:
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        paths = urllib.parse.parse_qs(body).get("paths", [])
        out = [info(p) for p in paths if os.path.exists(os.path.join(ROOT, p))]
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if "/resolve/" not in self.path:
            self.send_error(404)
            return
        name = urllib.parse.unquote(self.path.split("/resolve/", 1)[1].split("/", 1)[1])
        p = os.path.join(ROOT, name)
        if not os.path.exists(p):
            self.send_error(404)
            return
        data = open(p, "rb").read()
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes=") and rng.split("-", 1)[1].strip():
            # A bounded range is a header read (worker/vram.py): served at once,
            # never slowed or stalled, so the download knobs only affect downloads.
            if NO_RANGES:
                self.send_error(403)
                return
            a, b = (int(x) for x in rng[6:].split("-"))
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{a + len(body) - 1}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0] or 0)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        body = data[start:]
        if STALL["left"]:
            STALL["left"] -= 1
            self.wfile.write(body[:len(body) // 2])
            self.wfile.flush()
            time.sleep(600)
            return
        if not SLOW:
            self.wfile.write(body)
            return
        step = max(1, len(body) // 20)
        for i in range(0, len(body), step):
            self.wfile.write(body[i:i + step])
            self.wfile.flush()
            time.sleep(SLOW / 20)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

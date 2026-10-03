#!/usr/bin/env python3
"""Stand-in for llama-server, enough to drive worker/boot.sh end to end.

Behaviour knobs (env):
  FAKE_LLAMA_MODE  ok (default) | crash (exit during load) | novision |
                   textfail | die_after_ready
  FAKE_LLAMA_LOAD_SECS  seconds of 503 before /health turns 200 (default 2)
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = os.environ.get("FAKE_LLAMA_MODE", "ok")
LOAD = float(os.environ.get("FAKE_LLAMA_LOAD_SECS", "2"))

HELP = """usage: llama-server [options]
-m, --model FNAME
-mm, --mmproj FILE
-c, --ctx-size N
-np, --parallel N
-kvu, --kv-unified, -no-kvu, --no-kv-unified
-fa, --flash-attn [on|off|auto]
--jinja, --no-jinja
--reasoning-budget N
--no-webui
"""

argv = sys.argv[1:]
if "--help" in argv:
    print(HELP)
    sys.exit(0)
if "--list-devices" in argv:
    print("Available devices:\n  CUDA0: FAKE GPU (24000 MiB, 23000 MiB free)")
    sys.exit(0)


def arg(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


port = int(arg("--port", "8080"))
has_mmproj = "--mmproj" in argv
for f in (arg("-m"), arg("--mmproj")):
    if f and not os.path.exists(f):
        print(f"error loading model: {f} missing", flush=True)
        sys.exit(1)
print(f"fake llama-server args: {' '.join(argv)}", flush=True)
t_start = time.time()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        loaded = time.time() - t_start >= LOAD
        if self.path == "/health":
            self._json(200 if loaded else 503, {"status": "ok" if loaded else "loading"})
        elif self.path == "/props":
            self._json(200, {"modalities": {"vision": has_mmproj and MODE != "novision"}})
        else:
            self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if MODE == "textfail":
            self._json(500, {"error": {"message": "boom"}})
            return
        content = body["messages"][-1]["content"]
        is_image = isinstance(content, list)
        if is_image and not has_mmproj:
            self._json(500, {"error": {"message": "image input is not supported"}})
            return
        text = "Red" if is_image else "pong"
        self._json(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})


srv = ThreadingHTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print("main: server is listening", flush=True)

if MODE == "crash":
    time.sleep(0.5)
    print("error loading model: simulated", flush=True)
    sys.exit(1)
time.sleep(LOAD)
print("main: model loaded", flush=True)
if MODE == "die_after_ready":
    time.sleep(6)
    sys.exit(3)
while True:
    time.sleep(3600)

#!/usr/bin/env python3
"""Local router in front of the llama-servers on one worker.

The PyWorker forwards every request to a single model server port (18000 in
workers/openai/core.py). boot.sh runs the chat model (and, when one is
configured, the embedding model) as separate llama-server processes and
starts this router on that port:

  POST /v1/embeddings, /embeddings, /embedding   -> EMBED_URL (CHAT_URL without one)
  GET|POST /orch/info                            runtime facts for the shim's
                                                 status page (answered here)
  everything else                                -> CHAT_URL
  GET  /health                                   200 only when every server is healthy

Responses are passed through unchanged, streamed chunk by chunk (SSE works).
Standard library only: the llama.cpp image has python3 but no extra packages.

Environment: ROUTER_PORT (default 18000), CHAT_URL, EMBED_URL (empty = no
embedding server), ORCH_DIR (where boot.sh keeps plan-*.json).
"""

import http.client
import json
import os
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("ROUTER_PORT", "18000"))
CHAT = urllib.parse.urlsplit(os.environ.get("CHAT_URL", "http://127.0.0.1:18010"))
EMBED = urllib.parse.urlsplit(os.environ["EMBED_URL"]) if os.environ.get("EMBED_URL", "").strip() else None
ORCH_DIR = os.environ.get("ORCH_DIR", "/workspace/orch")
EMBED_PATHS = {"/v1/embeddings", "/embeddings", "/embedding"}
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "upgrade",
       "proxy-connection", "te", "trailer"}
TIMEOUT = 3600


def healthy(target):
    try:
        c = http.client.HTTPConnection(target.hostname, target.port, timeout=5)
        c.request("GET", "/health")
        ok = c.getresponse().status == 200
        c.close()
        return ok
    except OSError:
        return False


def get_json(target, path):
    """GET a llama-server JSON endpoint; None when it isn't there."""
    try:
        c = http.client.HTTPConnection(target.hostname, target.port, timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        body = r.read()
        c.close()
        return json.loads(body) if r.status == 200 else None
    except (OSError, ValueError):
        return None


def _server_info(target):
    props, models = get_json(target, "/props") or {}, get_json(target, "/v1/models") or {}
    gen = props.get("default_generation_settings") or {}
    meta = next((m.get("meta") for m in models.get("data") or [] if m.get("meta")), None) or {}
    return {
        "model": next((m.get("id") for m in models.get("data") or []), None),
        "model_path": props.get("model_path"),
        "slots": props.get("total_slots"),
        "ctx_per_slot": gen.get("n_ctx"),
        "ctx_train": meta.get("n_ctx_train"),
        "n_embd": meta.get("n_embd"),
        "n_params": meta.get("n_params"),
        "size_bytes": meta.get("size"),
        "vision": bool((props.get("modalities") or {}).get("vision")),
        "build": props.get("build_info"),
        "chat_template_caps": props.get("chat_template_caps"),
    }


def info():
    """What a client (SillyTavern) needs to know about this worker, from the
    llama-servers themselves and boot.sh's memory plan."""
    out = {"chat": _server_info(CHAT)}
    if out["chat"]["slots"] and out["chat"]["ctx_per_slot"]:
        out["chat"]["ctx_total"] = out["chat"]["slots"] * out["chat"]["ctx_per_slot"]
    slots = get_json(CHAT, "/slots")
    if isinstance(slots, list):
        out["chat"]["busy_slots"] = sum(1 for s in slots if s.get("is_processing"))
    if EMBED:
        out["embedding"] = _server_info(EMBED)
    for stage in ("chat", "local"):
        try:
            with open(os.path.join(ORCH_DIR, f"plan-{stage}.json")) as f:
                out["memory_plan"] = json.load(f)
            break
        except (OSError, ValueError):
            continue
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        name, used, total = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        out["gpu"] = {"name": name, "memory_used_mib": int(float(used)), "memory_total_mib": int(float(total))}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return out


class Router(BaseHTTPRequestHandler):
    # HTTP/1.0: each response ends by closing the connection, so a streamed body
    # needs no length or chunked framing of its own.
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _plain(self, code, text):
        data = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _forward(self, method):
        path = urllib.parse.urlsplit(self.path).path
        if method == "GET" and path == "/health":
            up = healthy(CHAT) and (EMBED is None or healthy(EMBED))
            self._plain(200 if up else 503, '{"status":"ok"}' if up else '{"status":"loading"}')
            return
        if path == "/orch/info":
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                self.rfile.read(n)
            self._plain(200, json.dumps(info()))
            return
        target = EMBED if EMBED and path in EMBED_PATHS else CHAT
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP | {"host"}}
        try:
            c = http.client.HTTPConnection(target.hostname, target.port, timeout=TIMEOUT)
            c.request(method, self.path, body=body, headers=headers)
            r = c.getresponse()
        except OSError as e:
            self._plain(502, '{"error":{"message":"router: upstream unreachable: %s"}}' % type(e).__name__)
            return
        self.send_response(r.status, r.reason)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        length = r.getheader("Content-Length")
        if length is not None:
            self.send_header("Content-Length", length)
        self.end_headers()
        try:
            while True:
                chunk = r.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError:
            pass  # client went away; drop the upstream request with it
        finally:
            c.close()

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Router)
    srv.daemon_threads = True
    print(f"[router] :{PORT} -> chat {CHAT.netloc}, embeddings {EMBED.netloc if EMBED else 'none'}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())

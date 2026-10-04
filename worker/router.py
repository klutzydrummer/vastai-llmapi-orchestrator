#!/usr/bin/env python3
"""Local router in front of two llama-servers on one worker.

The PyWorker forwards every request to a single model server port (18000 in
workers/openai/core.py). When an embedding model is configured, boot.sh runs
the chat model and the embedding model as separate llama-server processes and
starts this router on that port:

  POST /v1/embeddings, /embeddings, /embedding   -> EMBED_URL
  everything else                                -> CHAT_URL
  GET  /health                                   200 only when both are healthy

Responses are passed through unchanged, streamed chunk by chunk (SSE works).
Standard library only: the llama.cpp image has python3 but no extra packages.

Environment: ROUTER_PORT (default 18000), CHAT_URL, EMBED_URL.
"""

import http.client
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("ROUTER_PORT", "18000"))
CHAT = urllib.parse.urlsplit(os.environ.get("CHAT_URL", "http://127.0.0.1:18010"))
EMBED = urllib.parse.urlsplit(os.environ.get("EMBED_URL", "http://127.0.0.1:18011"))
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
            up = healthy(CHAT) and healthy(EMBED)
            self._plain(200 if up else 503, '{"status":"ok"}' if up else '{"status":"loading"}')
            return
        target = EMBED if path in EMBED_PATHS else CHAT
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
    print(f"[router] :{PORT} -> chat {CHAT.netloc}, embeddings {EMBED.netloc}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())

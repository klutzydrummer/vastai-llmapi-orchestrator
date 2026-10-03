#!/usr/bin/env python3
"""Records instance stop/destroy calls a worker makes against the Vast API.

Usage: fake_vast_api.py PORT LOGFILE   (one "METHOD PATH AUTH BODY" line per call)
"""
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, LOG = int(sys.argv[1]), sys.argv[2]


class H(BaseHTTPRequestHandler):
    def _record(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n).decode() if n else ""
        with open(LOG, "a") as f:
            f.write(f"{self.command} {self.path} {self.headers.get('Authorization', '')} {body}\n")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"success": true}')

    do_GET = do_PUT = do_DELETE = do_POST = _record

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

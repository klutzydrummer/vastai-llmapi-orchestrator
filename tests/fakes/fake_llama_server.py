#!/usr/bin/env python3
"""Stand-in for llama-server, enough to drive worker/boot.sh end to end.

Behaviour knobs (env):
  FAKE_LLAMA_MODE  ok (default) | crash (exit during load) | novision |
                   textfail | die_after_ready | oom_mmproj (CUDA out of
                   memory loading the projector, as on rental 54231885) |
                   thoughtleak (text reply is only '<|thought|>\n') |
                   roleleak (image reply starts with 'user\n') |
                   thinking (the answer is still in reasoning_content when
                   max_tokens runs out), the replies seen on rental 54245444
  FAKE_LLAMA_LOAD_SECS  seconds of 503 before /health turns 200 (default 2)
  FAKE_EMBED_MODE  ok (default) | zero (all-zero vectors) | crash (the
                   --embedding server exits during load)
  FAKE_EMBED_LOAD_SECS  load time of the --embedding server (default: as above)
  FAKE_EVENTS      file to append "start chat|embed" and "healthy chat|embed"
                   to, so tests can check the order servers came up in
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = os.environ.get("FAKE_LLAMA_MODE", "ok")
EMBED_MODE = os.environ.get("FAKE_EMBED_MODE", "ok")
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
MODE_EARLY = os.environ.get("FAKE_LLAMA_MODE", "ok")
if "--help" in argv:
    print(HELP)
    sys.exit(0)
if "--list-devices" in argv:
    if MODE_EARLY == "hang_devices":
        time.sleep(600)
    print("Available devices:\n  CUDA0: FAKE GPU (24000 MiB, 23000 MiB free)")
    sys.exit(0)


def arg(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


port = int(arg("--port", "8080"))
has_mmproj = "--mmproj" in argv
embedding = "--embedding" in argv
if embedding:
    MODE = "crash" if EMBED_MODE == "crash" else "ok"
    LOAD = float(os.environ.get("FAKE_EMBED_LOAD_SECS", LOAD))
NAME = "embed" if embedding else "chat"
EVENTS = os.environ.get("FAKE_EVENTS")
told = set()


def event(what):
    if EVENTS and what not in told:
        told.add(what)
        with open(EVENTS, "a") as f:
            f.write(f"{what} {NAME}\n")


event("start")
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
            if loaded:
                event("healthy")
            self._json(200 if loaded else 503, {"status": "ok" if loaded else "loading"})
        elif self.path == "/props":
            n_seq = int(arg("-np", "1"))
            self._json(200, {"modalities": {"vision": has_mmproj and MODE != "novision"},
                             "total_slots": n_seq, "model_path": arg("-m"), "build_info": "b0-fake",
                             "default_generation_settings": {"n_ctx": int(arg("-c", "4096")) // n_seq}})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": arg("-a", "model"), "object": "model",
                             "meta": {"n_ctx_train": 4096 if not embedding else 32768,
                                      "n_embd": 256 if not embedding else 128}}]})
        elif self.path == "/slots" and not embedding:
            self._json(200, [{"id": i, "is_processing": False} for i in range(int(arg("-np", "1")))])
        else:
            self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/apply-template" and not embedding:
            self._json(200, {"prompt": "<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n"})
            return
        if embedding:
            if self.path != "/v1/embeddings":
                self._json(501, {"error": {"message": "embedding server: chat not supported"}})
                return
            inp = body.get("input")
            items = inp if isinstance(inp, list) else [inp]
            val = 0.0 if EMBED_MODE == "zero" else 0.5
            self._json(200, {"object": "list", "model": body.get("model"), "data": [
                {"object": "embedding", "index": i, "embedding": [val] * 8} for i in range(len(items))]})
            return
        if self.path == "/v1/embeddings":
            self._json(501, {"error": {"message": "chat server: embeddings not enabled"}})
            return
        if MODE == "textfail":
            self._json(500, {"error": {"message": "boom"}})
            return
        content = body["messages"][-1]["content"]
        is_image = isinstance(content, list)
        if is_image and not has_mmproj:
            self._json(500, {"error": {"message": "image input is not supported"}})
            return
        text = "Red" if is_image else "pong"
        msg = {"role": "assistant", "content": text}
        if MODE == "thoughtleak" and not is_image:
            msg["content"] = "<|thought|>\n"
        elif MODE == "roleleak" and is_image:
            msg["content"] = "user\nRed"
        elif MODE == "thinking":
            msg = {"role": "assistant", "content": "", "reasoning_content": "The user wants one word, so"}
        self._json(200, {"choices": [{"message": msg, "finish_reason": "stop"}]})


srv = ThreadingHTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print("main: server is listening", flush=True)

if MODE == "crash":
    time.sleep(0.5)
    print("error loading model: simulated", flush=True)
    sys.exit(1)
if MODE == "oom_mmproj":
    # What b11371 printed on rental 54231885, the projector loading after the chat model.
    for line in (
            "llama_kv_cache:      CUDA0 KV buffer size =   340.00 MiB",
            "llama_context:      CUDA0 compute buffer size =   522.62 MiB",
            "clip_model_loader: model name:   gemma-4-26B-A4B-it",
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1139.46 MiB on device 0: "
            "cudaMalloc failed: out of memory",
            "alloc_tensor_range: failed to allocate CUDA0 buffer of size 1194806272",
            "clip_init: failed to load model 'mmproj-BF16.gguf': load_tensors: failed to allocate buffer",
            "mtmd_init_from_file: error: Failed to load CLIP model from mmproj-BF16.gguf",
            "srv    load_model: failed to load multimodal model, 'mmproj-BF16.gguf'",
            "main: exiting due to model loading error"):
        print(line, flush=True)
    time.sleep(0.3)
    sys.exit(1)
time.sleep(LOAD)
print("llama_kv_cache:      CUDA0 KV buffer size =    64.00 MiB", flush=True)
print("llama_context:      CUDA0 compute buffer size =    12.50 MiB", flush=True)
print("main: model loaded", flush=True)
if MODE == "die_after_ready":
    time.sleep(6)
    sys.exit(3)
while True:
    time.sleep(3600)

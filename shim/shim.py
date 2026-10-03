#!/usr/bin/env python3
"""OpenAI-compatible front door for a Vast.ai Serverless endpoint.

Point SillyTavern (or anything OpenAI-compatible) at this instead of a fixed
GPU box. Each request is routed through Vast's autoscaler; when no worker is
running, the request waits while one starts, and the connection is kept alive
in the meantime (SSE comments for streaming, leading whitespace for plain JSON)
so clients don't time out during a cold start.

Endpoints:
  POST /v1/chat/completions, /v1/completions   proxied to a worker
  GET  /v1/models                              answered locally, never wakes a GPU
  POST /wake                                   start a worker ahead of use
  GET  /status                                 worker states from the autoscaler
  GET  /health                                 the shim itself

Configuration is by environment variable; see config.example.env.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass

from aiohttp import web

log = logging.getLogger("shim")


def _env_bool(name, default):
    return os.environ.get(name, str(int(default))).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    vast_api_key: str
    endpoint_name: str
    served_model_name: str = "model"
    host: str = "127.0.0.1"
    port: int = 8787
    shim_api_key: str = ""
    request_timeout: float = 1800.0   # max wait for a worker, cold start included
    worker_timeout: float = 900.0     # max time a non-streaming generation may take
    keepalive_interval: float = 10.0
    default_cost: int = 512
    nonstream_keepalive: bool = True
    fast_fail_window: float = 1.5
    max_retries: int = 3              # worker 5xx/429 retries before giving up

    @classmethod
    def from_env(cls):
        key = os.environ.get("VAST_API_KEY", "").strip()
        name = os.environ.get("ENDPOINT_NAME", "").strip()
        if not key or not name:
            raise SystemExit("VAST_API_KEY and ENDPOINT_NAME must be set")
        return cls(
            vast_api_key=key,
            endpoint_name=name,
            served_model_name=os.environ.get("SERVED_MODEL_NAME", "model"),
            host=os.environ.get("SHIM_HOST", "127.0.0.1"),
            port=int(os.environ.get("SHIM_PORT", "8787")),
            shim_api_key=os.environ.get("SHIM_API_KEY", "").strip(),
            request_timeout=float(os.environ.get("REQUEST_TIMEOUT", "1800")),
            worker_timeout=float(os.environ.get("WORKER_TIMEOUT", "900")),
            keepalive_interval=float(os.environ.get("KEEPALIVE_INTERVAL", "10")),
            default_cost=int(os.environ.get("DEFAULT_COST", "512")),
            nonstream_keepalive=_env_bool("NONSTREAM_KEEPALIVE", True),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
        )


class UpstreamError(Exception):
    def __init__(self, status, message, body=None):
        super().__init__(message)
        self.status = status
        self.body = body


def _error_body(status, message):
    return {"error": {"message": message, "type": "orchestrator_error", "code": status}}


class Shim:
    def __init__(self, cfg: Config, client=None, endpoint=None):
        self.cfg = cfg
        self.client = client
        self._endpoint = endpoint
        self._endpoint_lock = asyncio.Lock()
        self.stats = {"requests": 0, "errors": 0, "last_wait_s": None, "last_ok_at": None}

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def start(self, app):
        if self.client is None:
            from vastai import CoroutineServerless
            self.client = CoroutineServerless(api_key=self.cfg.vast_api_key)
            await self.client.__aenter__()

    async def stop(self, app):
        if self.client is not None:
            try:
                await self.client.close()
            except Exception:
                pass

    async def endpoint(self):
        if self._endpoint is not None:
            return self._endpoint
        async with self._endpoint_lock:
            if self._endpoint is None:
                try:
                    self._endpoint = await self.client.get_endpoint(name=self.cfg.endpoint_name)
                except Exception as e:
                    raise UpstreamError(503, f"endpoint {self.cfg.endpoint_name!r} unavailable: {e}")
                log.info("using endpoint %s", self._endpoint)
        return self._endpoint

    # ── helpers ──────────────────────────────────────────────────────────────
    def _authorized(self, request):
        if not self.cfg.shim_api_key:
            return True
        return request.headers.get("Authorization", "") == f"Bearer {self.cfg.shim_api_key}"

    def _cost(self, body):
        for k in ("max_tokens", "max_completion_tokens", "n_predict"):
            v = body.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
        return self.cfg.default_cost

    async def _dispatch(self, route, body, stream):
        """Route to a worker and return the SDK result dict; raise UpstreamError."""
        ep = await self.endpoint()
        attempts = 0
        t0 = time.time()
        while True:
            attempts += 1
            try:
                result = await self.client.queue_endpoint_request(
                    endpoint=ep,
                    worker_route=route,
                    worker_payload=body,
                    cost=self._cost(body),
                    timeout=self.cfg.request_timeout,
                    worker_timeout=self.cfg.worker_timeout,
                    stream=stream,
                    max_retries=self.cfg.max_retries,
                )
            except asyncio.TimeoutError:
                raise UpstreamError(504, f"no worker became ready within {self.cfg.request_timeout:.0f}s")
            except Exception as e:
                # The SDK does not retry a failed /route/ call; one transient
                # autoscaler error shouldn't fail the whole request.
                remaining = self.cfg.request_timeout - (time.time() - t0)
                if "route" in str(e).lower() and attempts < 3 and remaining > 10:
                    log.warning("route failed (%s), retrying", e)
                    await asyncio.sleep(2 * attempts)
                    continue
                raise UpstreamError(502, f"upstream request failed: {e}")
            if not result.get("ok"):
                status = result.get("status") or 502
                resp = result.get("response")
                msg = (resp.get("error") if isinstance(resp, dict) else None) or result.get("text") or "worker error"
                if isinstance(msg, dict):
                    msg = msg.get("message", json.dumps(msg))
                raise UpstreamError(status, str(msg)[:1000], resp if isinstance(resp, dict) else None)
            return result

    # ── handlers ─────────────────────────────────────────────────────────────
    async def health(self, request):
        return web.json_response({"ok": True})

    async def models(self, request):
        if not self._authorized(request):
            return web.json_response(_error_body(401, "unauthorized"), status=401)
        return web.json_response({"object": "list", "data": [{
            "id": self.cfg.served_model_name, "object": "model",
            "created": 0, "owned_by": "vast-serverless"}]})

    async def status(self, request):
        if not self._authorized(request):
            return web.json_response(_error_body(401, "unauthorized"), status=401)
        out = {"endpoint": self.cfg.endpoint_name, "shim": self.stats}
        try:
            workers = await self.client.get_endpoint_workers(await self.endpoint())
            out["workers"] = [asdict(w) for w in workers]
        except Exception as e:
            out["workers_error"] = str(e)
        return web.json_response(out)

    async def wake(self, request):
        if not self._authorized(request):
            return web.json_response(_error_body(401, "unauthorized"), status=401)

        async def _go():
            try:
                await self._dispatch("/v1/completions", {
                    "model": self.cfg.served_model_name, "prompt": "hi", "max_tokens": 1}, False)
                log.info("wake: a worker is ready")
            except Exception as e:
                log.warning("wake failed: %s", e)

        asyncio.create_task(_go())
        return web.json_response({"ok": True, "message": "worker requested"}, status=202)

    async def generate(self, request, route):
        if not self._authorized(request):
            return web.json_response(_error_body(401, "unauthorized"), status=401)
        try:
            body = await request.json()
            assert isinstance(body, dict)
        except Exception:
            return web.json_response(_error_body(400, "body must be a JSON object"), status=400)

        body["model"] = self.cfg.served_model_name
        stream = bool(body.get("stream"))
        self.stats["requests"] += 1
        t0 = time.time()
        task = asyncio.create_task(self._dispatch(route, body, stream))

        # Fast path: fail with a real status code if the error is immediate
        # (bad API key, unknown endpoint) before committing a 200.
        await asyncio.wait({task}, timeout=self.cfg.fast_fail_window)
        if task.done() and task.exception() is not None:
            return self._error_response(task.exception())

        if not stream and not self.cfg.nonstream_keepalive:
            try:
                result = await task
            except Exception as e:
                return self._error_response(e)
            self._ok(t0)
            return web.json_response(result["response"])

        resp = web.StreamResponse(status=200, headers={
            "Content-Type": "text/event-stream" if stream else "application/json",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)
        upstream = None
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=self.cfg.keepalive_interval)
                if not task.done():
                    await resp.write(b": waiting for a GPU worker\n\n" if stream else b" ")
            try:
                result = task.result()
            except Exception as e:
                await self._write_error(resp, e, stream)
                return resp
            wait_s = time.time() - t0
            if stream:
                upstream = result["response"]
                async for event in upstream:
                    await resp.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                await resp.write(b"data: [DONE]\n\n")
            else:
                await resp.write(json.dumps(result["response"]).encode())
            self._ok(t0, wait_s)
            await resp.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            log.info("client went away; cancelling request")
            task.cancel()
            if upstream is not None:
                try:
                    await upstream.aclose()
                except Exception:
                    pass
            raise
        return resp

    def _ok(self, t0, wait_s=None):
        self.stats["last_ok_at"] = int(time.time())
        self.stats["last_wait_s"] = round(wait_s if wait_s is not None else time.time() - t0, 1)

    def _error_response(self, e):
        self.stats["errors"] += 1
        status = e.status if isinstance(e, UpstreamError) else 502
        body = (e.body if isinstance(e, UpstreamError) and e.body else _error_body(status, str(e)))
        log.warning("request failed: %s %s", status, e)
        return web.json_response(body, status=status)

    async def _write_error(self, resp, e, stream):
        self.stats["errors"] += 1
        status = e.status if isinstance(e, UpstreamError) else 502
        log.warning("request failed after headers were sent: %s %s", status, e)
        body = json.dumps(_error_body(status, str(e))).encode()
        if stream:
            await resp.write(b"data: " + body + b"\n\ndata: [DONE]\n\n")
        else:
            await resp.write(body)
        await resp.write_eof()


def make_app(shim: Shim):
    app = web.Application(client_max_size=64 * 1024 * 1024)  # base64 images are large
    app.on_startup.append(shim.start)
    app.on_cleanup.append(shim.stop)
    app.router.add_get("/health", shim.health)
    app.router.add_get("/v1/models", shim.models)
    app.router.add_get("/status", shim.status)
    app.router.add_post("/wake", shim.wake)
    app.router.add_post("/v1/chat/completions", lambda r: shim.generate(r, "/v1/chat/completions"))
    app.router.add_post("/v1/completions", lambda r: shim.generate(r, "/v1/completions"))
    return app


def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.from_env()
    web.run_app(make_app(Shim(cfg)), host=cfg.host, port=cfg.port, access_log=None)


if __name__ == "__main__":
    main()

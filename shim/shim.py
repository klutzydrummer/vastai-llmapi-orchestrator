#!/usr/bin/env python3
"""OpenAI-compatible front door for a Vast.ai Serverless endpoint.

Point SillyTavern (or anything OpenAI-compatible) at this instead of a fixed
GPU box. Each request is routed through Vast's autoscaler; when no worker is
running, the request waits while one starts, and the connection is kept alive
in the meantime (SSE comments for streaming, leading whitespace for plain JSON)
so clients don't time out during a cold start.

Endpoints:
  POST /v1/chat/completions, /v1/completions   proxied to a worker
  POST /v1/embeddings                          proxied to a worker (when
                                               EMBED_MODEL_NAME is set)
  GET  /v1/models                              answered locally, never wakes a GPU
  POST /wake                                   start a worker ahead of use
  GET  /status                                 worker states from the autoscaler
  GET  /health                                 the shim itself

With WARM_HOURS set, the shim also keeps a worker awake during those hours
(see Warm hours in config.example.env).

Configuration is by environment variable; see config.example.env.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from aiohttp import web

log = logging.getLogger("shim")


def _env_bool(name, default):
    return os.environ.get(name, str(int(default))).strip().lower() in ("1", "true", "yes", "on")


def parse_warm_hours(text):
    """"15:00-23:00" or "06:30-08:00,15:00-23:00" -> [(start_min, end_min), ...].
    A range may cross midnight ("22:00-02:00"). Empty -> [] (off)."""
    out = []
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        try:
            a, b = part.split("-")
            mins = []
            for t in (a, b):
                h, m = t.strip().split(":")
                h, m = int(h), int(m)
                if not (0 <= h <= 24 and 0 <= m < 60) or (h == 24 and m):
                    raise ValueError
                mins.append(h * 60 + m)
        except ValueError:
            raise SystemExit(f"WARM_HOURS: can't read {part!r}; use HH:MM-HH:MM, comma-separated")
        if mins[0] == mins[1]:
            raise SystemExit(f"WARM_HOURS: {part!r} is empty")
        out.append((mins[0], mins[1]))
    return out


def in_warm_hours(windows, now):
    """True when `now` (a datetime) falls inside any window."""
    m = now.hour * 60 + now.minute
    return any((a <= m < b) if a < b else (m >= a or m < b) for a, b in windows)


@dataclass
class Config:
    vast_api_key: str
    endpoint_name: str
    served_model_name: str = "model"
    embed_model_name: str = ""        # the worker's embedding model; empty = no /v1/embeddings
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
    warm_hours: list = field(default_factory=list)   # [(start_min, end_min)]; empty = off
    warm_tz: str = ""                 # IANA zone for warm_hours; empty = the machine's local time
    warm_ping_s: float = 300.0        # how often to ping during warm hours; under inactivity_timeout

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
            embed_model_name=os.environ.get("EMBED_MODEL_NAME", "").strip(),
            host=os.environ.get("SHIM_HOST", "127.0.0.1"),
            port=int(os.environ.get("SHIM_PORT", "8787")),
            shim_api_key=os.environ.get("SHIM_API_KEY", "").strip(),
            request_timeout=float(os.environ.get("REQUEST_TIMEOUT", "1800")),
            worker_timeout=float(os.environ.get("WORKER_TIMEOUT", "900")),
            keepalive_interval=float(os.environ.get("KEEPALIVE_INTERVAL", "10")),
            default_cost=int(os.environ.get("DEFAULT_COST", "512")),
            nonstream_keepalive=_env_bool("NONSTREAM_KEEPALIVE", True),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
            warm_hours=parse_warm_hours(os.environ.get("WARM_HOURS", "")),
            warm_tz=os.environ.get("WARM_TZ", "").strip(),
            warm_ping_s=float(os.environ.get("WARM_PING_S", "300")),
        ).checked()

    def checked(self):
        if self.warm_tz:
            try:
                ZoneInfo(self.warm_tz)
            except Exception:
                raise SystemExit(f"WARM_TZ: unknown time zone {self.warm_tz!r} (use an IANA name like America/Chicago)")
        if not 0 < self.warm_ping_s < 900:
            raise SystemExit("WARM_PING_S must be between 0 and 900 seconds (under the endpoint's inactivity_timeout)")
        return self


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
        self.warm = {"hours": os.environ.get("WARM_HOURS", "").strip() or None, "active": False,
                     "pings": 0, "last_ping_at": None, "last_ping": None}
        self._warm_task = self._warm_ping = None
        self._tz = ZoneInfo(cfg.warm_tz) if cfg.warm_tz else None

    def now(self):
        return datetime.now(self._tz) if self._tz else datetime.now().astimezone()

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def start(self, app):
        if self.client is None:
            from vastai import CoroutineServerless
            self.client = CoroutineServerless(api_key=self.cfg.vast_api_key)
            await self.client.__aenter__()
        if self.cfg.warm_hours:
            if not self.cfg.warm_tz:
                log.warning("WARM_TZ is not set; warm hours use this machine's clock (now %s). In Docker "
                            "that is usually UTC", self.now().strftime("%H:%M %Z"))
            self._warm_task = asyncio.create_task(self._warm_loop())

    async def stop(self, app):
        for t in (self._warm_task, self._warm_ping):
            if t is not None:
                t.cancel()
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

    # ── warm hours ───────────────────────────────────────────────────────────
    async def _ping(self):
        """One minimal request through the autoscaler: starts a worker if none
        runs, and counts as activity so a running one isn't released."""
        try:
            await self._dispatch("/v1/completions", {
                "model": self.cfg.served_model_name, "prompt": "hi", "max_tokens": 1}, False)
            return "ok"
        except Exception as e:
            log.warning("warm ping failed: %s", e)
            return f"failed: {e}"[:300]

    async def _warm_loop(self):
        """During warm hours, ping every warm_ping_s. A ping still waiting on a
        cold start is never doubled up; outside the hours nothing is sent and
        the worker idles out as usual."""
        while True:
            active = in_warm_hours(self.cfg.warm_hours, self.now())
            if active != self.warm["active"]:
                log.info("warm hours %s (%s)", "started" if active else "ended", self.warm["hours"])
                self.warm["active"] = active
            if self._warm_ping is not None and self._warm_ping.done():
                self.warm["last_ping"] = self._warm_ping.result()
                self._warm_ping = None
            if active and self._warm_ping is None:
                self.warm["last_ping_at"] = int(time.time())
                self.warm["pings"] += 1
                self._warm_ping = asyncio.create_task(self._ping())
            await asyncio.sleep(self.cfg.warm_ping_s if active else min(60.0, self.cfg.warm_ping_s))

    # ── helpers ──────────────────────────────────────────────────────────────
    def _authorized(self, request):
        if not self.cfg.shim_api_key:
            return True
        return request.headers.get("Authorization", "") == f"Bearer {self.cfg.shim_api_key}"

    def _cost(self, body):
        if "input" in body and "messages" not in body and "prompt" not in body:
            # Embeddings: about 4 characters per token, as the worker counts it.
            inp = body["input"]
            items = inp if isinstance(inp, list) else [inp]
            return max(1, int(sum(len(x) if isinstance(x, str) else len(str(x)) for x in items) / 4))
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
        ids = [self.cfg.served_model_name] + ([self.cfg.embed_model_name] if self.cfg.embed_model_name else [])
        return web.json_response({"object": "list", "data": [{
            "id": i, "object": "model", "created": 0, "owned_by": "vast-serverless"} for i in ids]})

    async def status(self, request):
        if not self._authorized(request):
            return web.json_response(_error_body(401, "unauthorized"), status=401)
        out = {"endpoint": self.cfg.endpoint_name, "shim": self.stats, "warm": self.warm}
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

        if route == "/v1/embeddings":
            if not self.cfg.embed_model_name:
                return web.json_response(_error_body(404, "no embedding model configured (EMBED_MODEL_NAME)"),
                                         status=404)
            body["model"] = self.cfg.embed_model_name
            body.pop("stream", None)
        else:
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
    app.router.add_post("/v1/embeddings", lambda r: shim.generate(r, "/v1/embeddings"))
    return app


def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.from_env()
    web.run_app(make_app(Shim(cfg)), host=cfg.host, port=cfg.port, access_log=None)


if __name__ == "__main__":
    main()

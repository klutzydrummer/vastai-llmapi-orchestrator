#!/usr/bin/env python3
"""Shim tests against a fake Vast autoscaler + fake PyWorker, using the real
vastai SDK client. Run: python3 tests/test_shim.py (needs vastai, aiohttp)."""

import asyncio
import json
import os
import sys
import time

from aiohttp import ClientSession, web

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shim"))
import shim as shim_mod  # noqa: E402
from vastai import CoroutineServerless  # noqa: E402
from vastai.serverless.client.endpoint import Endpoint  # noqa: E402

FAKE_PORT, SHIM_PORT = 18601, 18602
FAKE = f"http://127.0.0.1:{FAKE_PORT}"


class Fake:
    """Autoscaler (/route/) and worker (/v1/...) in one server."""

    def __init__(self):
        self.ready_at = 0.0          # route returns no worker until this time
        self.worker_status = 200
        self.received = []

    async def route(self, request):
        body = await request.json()
        if time.time() < self.ready_at:
            return web.json_response({"request_idx": 7})
        return web.json_response({"request_idx": 7, "url": FAKE, "signature": "x", "reqnum": 1})

    async def worker(self, request):
        body = await request.json()
        assert "auth_data" in body and "payload" in body, body
        payload = body["payload"]
        self.received.append(payload)
        if self.worker_status != 200:
            return web.json_response({"error": {"message": "worker exploded"}}, status=self.worker_status)
        if payload.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for word in ("Hello", " there"):
                chunk = {"choices": [{"delta": {"content": word}, "index": 0}]}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": "Hello there"}}]})


async def make_shim(**overrides):
    cfg = shim_mod.Config(vast_api_key="k", endpoint_name="ep", served_model_name="waifu",
                          port=SHIM_PORT, keepalive_interval=0.5, request_timeout=8,
                          fast_fail_window=0.3, max_retries=2)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    client = CoroutineServerless(api_key="k", autoscaler_url=FAKE, max_poll_interval=0.5)

    async def no_ssl():
        return None
    client.get_ssl_context = no_ssl
    await client.__aenter__()
    ep = Endpoint(client, name="ep", id=1, api_key="k")
    s = shim_mod.Shim(cfg, client=client, endpoint=ep)
    runner = web.AppRunner(shim_mod.make_app(s))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", SHIM_PORT).start()
    return s, runner


PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"PASS  {name}")


async def main():
    fake = Fake()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post("/route/", fake.route)
    app.router.add_post("/v1/chat/completions", fake.worker)
    app.router.add_post("/v1/completions", fake.worker)
    frunner = web.AppRunner(app)
    await frunner.setup()
    await web.TCPSite(frunner, "127.0.0.1", FAKE_PORT).start()

    base = f"http://127.0.0.1:{SHIM_PORT}"
    s, runner = await make_shim()
    async with ClientSession() as http:
        # /v1/models never touches the autoscaler
        async with http.get(base + "/v1/models") as r:
            d = await r.json()
            assert r.status == 200 and d["data"][0]["id"] == "waifu"
        ok("/v1/models answered locally")

        # non-streaming during a cold start: whitespace keepalive, valid JSON at the end
        fake.ready_at = time.time() + 2.5
        t0 = time.time()
        async with http.post(base + "/v1/chat/completions", json={
                "model": "whatever", "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}]}) as r:
            raw = await r.read()
        assert r.status == 200, raw
        assert raw.startswith(b" "), raw[:20]
        assert json.loads(raw)["choices"][0]["message"]["content"] == "Hello there"
        assert time.time() - t0 >= 2.0
        assert fake.received[-1]["model"] == "waifu"
        ok("non-streaming waits through a cold start with keepalives")

        # streaming during a cold start: SSE comments, then chunks, then [DONE]
        fake.ready_at = time.time() + 1.5
        async with http.post(base + "/v1/chat/completions", json={
                "stream": True, "messages": [{"role": "user", "content": "hi"}]}) as r:
            text = (await r.read()).decode()
        assert r.headers["Content-Type"].startswith("text/event-stream")
        assert ": waiting for a GPU worker" in text, text
        events = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: {")]
        assert "".join(e["choices"][0]["delta"]["content"] for e in events) == "Hello there"
        assert text.rstrip().endswith("data: [DONE]")
        ok("streaming waits through a cold start and relays chunks")

        # image payloads pass through untouched (and big bodies are accepted)
        img = "data:image/png;base64," + "A" * 3_000_000
        async with http.post(base + "/v1/chat/completions", json={"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img}}, {"type": "text", "text": "?"}]}]}) as r:
            await r.read()
        assert fake.received[-1]["messages"][0]["content"][0]["image_url"]["url"] == img
        ok("image payload forwarded intact")

        # worker error returned fast -> real status code, not a 200
        fake.ready_at = 0
        fake.worker_status = 400
        async with http.post(base + "/v1/chat/completions", json={"messages": []}) as r:
            d = await r.json()
            assert r.status == 400, (r.status, d)
            assert "worker exploded" in json.dumps(d)
        ok("immediate worker error keeps its status code")

        # worker 5xx is retried a bounded number of times, then surfaced
        fake.worker_status = 500
        n0 = len(fake.received)
        t0 = time.time()
        async with http.post(base + "/v1/chat/completions", json={"messages": []}) as r:
            raw = await r.read()
        assert len(fake.received) - n0 <= 3, len(fake.received) - n0
        assert time.time() - t0 < 8
        assert b"worker exploded" in raw
        ok("5xx retries are bounded")
        fake.worker_status = 200

        # no worker ever becomes ready -> timeout error, streamed as an SSE error event
        fake.ready_at = time.time() + 999
        async with http.post(base + "/v1/chat/completions", json={
                "stream": True, "messages": []}) as r:
            text = (await r.read()).decode()
        assert "no worker became ready" in text and text.rstrip().endswith("data: [DONE]"), text
        ok("cold-start timeout reported to the client")
        fake.ready_at = 0

    await runner.cleanup()

    # auth
    s, runner = await make_shim(shim_api_key="secret")
    async with ClientSession() as http:
        async with http.get(base + "/v1/models") as r:
            assert r.status == 401
        async with http.get(base + "/v1/models", headers={"Authorization": "Bearer secret"}) as r:
            assert r.status == 200
    ok("SHIM_API_KEY enforced")
    await runner.cleanup()
    await frunner.cleanup()
    print(f"---- {len(PASSED)} passed")


if __name__ == "__main__":
    asyncio.run(main())

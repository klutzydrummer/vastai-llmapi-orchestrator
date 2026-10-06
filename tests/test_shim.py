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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
import pyworker_worker  # noqa: E402
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
        self.costs = []              # cost of every /route/ call
        self.paths = []              # path of every request the worker got
        self.workers = []            # get_endpoint_workers answer

    async def route(self, request):
        body = await request.json()
        self.costs.append(body.get("cost"))
        if time.time() < self.ready_at:
            return web.json_response({"request_idx": 7})
        return web.json_response({"request_idx": 7, "url": FAKE, "signature": "x", "reqnum": 1})

    async def endpoint_workers(self, request):
        return web.json_response(self.workers)

    async def worker(self, request):
        body = await request.json()
        assert "auth_data" in body and "payload" in body, body
        payload = body["payload"]
        self.received.append(payload)
        self.paths.append(request.path)
        if self.worker_status != 200:
            return web.json_response({"error": {"message": "worker exploded"}}, status=self.worker_status)
        if request.path == "/orch/info":
            return web.json_response({"chat": {"slots": 2, "ctx_per_slot": 16384, "ctx_total": 32768}})
        if request.path == "/v1/embeddings":
            n = len(payload["input"]) if isinstance(payload["input"], list) else 1
            return web.json_response({"object": "list", "model": payload["model"], "data": [
                {"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(n)]})
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
    app.router.add_post("/v1/embeddings", fake.worker)
    app.router.add_post("/orch/info", fake.worker)
    app.router.add_post("/get_endpoint_workers/", fake.endpoint_workers)
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

        # no embedding model configured: /v1/embeddings is refused without waking a GPU
        n0 = len(fake.received)
        async with http.post(base + "/v1/embeddings", json={"input": "hi"}) as r:
            assert r.status == 404 and len(fake.received) == n0, r.status
        ok("/v1/embeddings refused when no embedding model is set")

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

    # embeddings, with an embedding model configured
    s, runner = await make_shim(embed_model_name="qwen3-embed")
    async with ClientSession() as http:
        async with http.get(base + "/v1/models") as r:
            ids = [m["id"] for m in (await r.json())["data"]]
        assert ids == ["waifu", "qwen3-embed"], ids
        async with http.post(base + "/v1/embeddings", json={
                "model": "text-embedding-3-small", "input": ["a", "b"], "stream": True}) as r:
            raw = await r.read()
        assert r.status == 200, raw
        d = json.loads(raw)
        assert len(d["data"]) == 2 and d["data"][1]["embedding"] == [0.1, 0.2], d
        assert fake.received[-1]["model"] == "qwen3-embed" and "stream" not in fake.received[-1]
        assert s._cost({"input": "x" * 400}) == 100
        # /wake uses the embedding model, leaving the chat model's slots alone
        n0 = len(fake.received)
        async with http.post(base + "/wake") as r:
            assert r.status == 202
        for _ in range(20):
            if len(fake.received) > n0:
                break
            await asyncio.sleep(0.1)
        assert fake.received[n0:] == [{"model": "qwen3-embed", "input": "hi"}], fake.received[n0:]
    ok("/v1/embeddings reaches the worker with the embedding model's name")
    await runner.cleanup()

    # Release rule: every kind of client work goes through the router with a
    # cost above zero, and the worker reports the same number as load.
    table = [
        {"input": "x" * 400}, {"input": ["ab", "cd", "e" * 4000]}, {"input": ""}, {"input": [1, 2]},
        {"messages": [{"role": "user", "content": "hi"}]},                     # no limit named
        {"messages": [], "max_tokens": 300}, {"messages": [], "max_tokens": 0},
        {"messages": [], "max_tokens": -1}, {"messages": [], "max_tokens": True},
        {"messages": [], "max_completion_tokens": 77}, {"prompt": "hi", "n_predict": 9},
        {"prompt": "hi", "max_tokens": "200"}, {"prompt": "hi", "max_tokens": 0.5},
    ]
    for body in table:
        w, c = pyworker_worker.request_workload(body), shim_mod.request_cost(body)
        assert w >= 1 and c >= 1 and int(w) == c, (body, w, c)
    assert shim_mod.request_cost({"messages": []}) == pyworker_worker.DEFAULT_WORKLOAD == shim_mod.DEFAULT_COST
    assert pyworker_worker.request_workload({"messages": [], "max_tokens": 64}) == 64
    ok("shim routing cost and worker load agree, and are never zero")

    s, runner = await make_shim(embed_model_name="qwen3-embed")
    async with ClientSession() as http:
        fake.costs.clear()
        fake.paths.clear()
        async with http.post(base + "/v1/embeddings", json={"input": "a short query"}) as r:
            await r.read()
        async with http.post(base + "/v1/chat/completions", json={
                "stream": True, "messages": [{"role": "user", "content": "hi"}]}) as r:
            await r.read()
        async with http.post(base + "/v1/completions", json={"prompt": "hi"}) as r:
            await r.read()
        assert fake.paths == ["/v1/embeddings", "/v1/chat/completions", "/v1/completions"], fake.paths
        assert fake.costs == [3, 512, 512], fake.costs
    ok("embedding, chat and completion requests all reach Vast's router with a load")
    await runner.cleanup()

    # status page: never sends anything to a worker; values come from the
    # worker right after client work, while one is listed as running
    cache = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"shim-info-{os.getpid()}.json")
    s, runner = await make_shim(info_cache=cache, info_min_interval=0.2)
    async with ClientSession() as http:
        async with http.get(base + "/") as r:
            html = await r.text()
        assert r.status == 200 and "Copy all" in html and "info.json" in html
        async with http.get(base + "/v1/info") as r:   # for a proxy that only passes /v1/*
            assert r.status == 200 and await r.text() == html
        fake.workers = []
        fake.costs.clear()
        fake.paths.clear()
        async with http.get(base + "/info.json") as r:
            d = await r.json()
        assert d["live"] is None and not d["live_fresh"] and not fake.costs, d
        assert d["models"] == {"chat": "waifu", "embedding": None} and d["api_key_required"] is False
        ok("status page with no worker running asks no worker")
        fake.workers = [{"id": 5, "status": "running"}]
        for _ in range(5):
            async with http.get(base + "/info.json") as r:
                d = await r.json()
        assert not fake.costs and d["live"] is None and d["workers"][0]["status"] == "running", (fake.costs, d)
        ok("status page never sends a request to a running worker (it would count as activity)")
        async with http.post(base + "/v1/chat/completions", json={"messages": [], "max_tokens": 5}) as r:
            await r.read()
        for _ in range(30):
            if "/orch/info" in fake.paths:
                break
            await asyncio.sleep(0.1)
        assert fake.paths == ["/v1/chat/completions", "/orch/info"] and fake.costs == [5, 1], (fake.paths, fake.costs)
        for _ in range(30):
            if s._info is not None:
                break
            await asyncio.sleep(0.05)
        async with http.get(base + "/info.json") as r:
            d = await r.json()
        assert d["live_fresh"] and d["live"]["chat"]["ctx_per_slot"] == 16384 and os.path.exists(cache), d
        ok("worker info is refreshed right after client work")
        # A finished request while no worker is listed running: no info
        # request, since a routed one would start a worker.
        fake.workers = []
        fake.costs.clear()
        fake.paths.clear()
        await asyncio.sleep(0.3)
        async with http.post(base + "/v1/chat/completions", json={"messages": [], "max_tokens": 5}) as r:
            await r.read()
        await asyncio.sleep(0.5)
        assert fake.paths == ["/v1/chat/completions"] and len(fake.costs) == 1, (fake.paths, fake.costs)
        ok("info refresh skipped when no worker is listed running")
    await runner.cleanup()
    s, runner = await make_shim(info_cache=cache)
    fake.workers = []
    async with ClientSession() as http:
        async with http.get(base + "/info.json") as r:
            d = await r.json()
        assert not d["live_fresh"] and d["live"]["chat"]["slots"] == 2 and d["live_at"], d
    ok("status page shows the last answer, with its age, once the worker is gone")
    os.remove(cache)
    await runner.cleanup()

    # auth
    s, runner = await make_shim(shim_api_key="secret")
    async with ClientSession() as http:
        async with http.get(base + "/v1/models") as r:
            assert r.status == 401
        async with http.get(base + "/v1/models", headers={"Authorization": "Bearer secret"}) as r:
            assert r.status == 200
        async with http.get(base + "/info.json") as r:
            assert r.status == 401
        async with http.get(base + "/info") as r:
            assert r.status == 200   # the page has no data of its own; it asks for the key
        async with http.get(base + "/v1/info.json") as r:
            assert r.status == 401
        async with http.get(base + "/v1/info.json", headers={"Authorization": "Bearer secret"}) as r:
            assert r.status == 200 and "models" in await r.json()
    ok("SHIM_API_KEY enforced")
    await runner.cleanup()
    await frunner.cleanup()
    print(f"---- {len(PASSED)} passed")


if __name__ == "__main__":
    asyncio.run(main())

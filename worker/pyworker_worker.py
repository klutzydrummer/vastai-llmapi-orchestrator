"""PyWorker entry point: Vast's llama worker plus /v1/embeddings and /orch/info.

boot.sh copies this file to $WORKSPACE_DIR/vast-pyworker/worker.py, inside a
checkout of vast-ai/pyworker at the pinned PYWORKER_REF. That ref's
start_server.sh runs `python3 -m worker` when worker.py exists at the checkout
root, before falling back to workers/$BACKEND/worker.py.

Everything except the extra routes and the workload function is
workers/openai/core.py's run() at that ref, using its own request parser and
benchmark. The stock worker counts a chat request without max_tokens as no
load at all; request_workload() below doesn't. Requests on all routes go to
the model server on 127.0.0.1:18000, worker/router.py in front of the llama-servers;
/orch/info is answered by the router itself (slots, context per slot, the
memory plan) for the shim's status page.

Release rule (README, "When a worker is released"): Vast's autoscaler
releases a worker once the endpoint has been idle for inactivity_timeout. The
only activity it can see from us is requests routed to workers and the load
the workers report for them. So every client request (chat, completion, embedding) is a
handler here with a workload above zero, worked out by request_workload(),
the same function the shim uses for the routing cost. Streams count until
their last chunk: the SDK keeps a request in its working set until the
response is finished. /orch/info is the exception and is asked only right
after client work (shim.py), never on its own.
"""

import os

DEFAULT_WORKLOAD = 512   # a generation request that names no limit; = shim DEFAULT_COST


def request_workload(data):
    """Load one request puts on the worker, in the benchmark's units (tokens).
    Embeddings: about 4 characters per input token. Generations: the token
    limit the request names, else DEFAULT_WORKLOAD. Never below 1, so no
    request reaches the autoscaler as no load at all."""
    if not isinstance(data, dict):
        return float(DEFAULT_WORKLOAD)
    if "input" in data and "messages" not in data and "prompt" not in data:
        inp = data["input"]
        items = inp if isinstance(inp, list) else [inp]
        return max(1.0, sum(len(x) if isinstance(x, str) else len(str(x)) for x in items) / 4)
    for k in ("max_tokens", "max_completion_tokens", "n_predict"):
        v = data.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 1:
            return float(v)
    return float(DEFAULT_WORKLOAD)


def main():
    from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig
    from workers.openai.core import (
        MODEL_SERVER_PORT,
        MODEL_SERVER_URL,
        _env_lines,
        completions_benchmark_generator,
        request_parser,
    )

    handlers = [
        HandlerConfig(
            route="/v1/completions",
            workload_calculator=request_workload,
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
            benchmark_config=BenchmarkConfig(
                generator=completions_benchmark_generator, concurrency=10, runs=3
            ),
        ),
        HandlerConfig(
            route="/v1/chat/completions",
            workload_calculator=request_workload,
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
        ),
    ]
    if os.environ.get("EMBED_SERVED_NAME"):
        handlers.append(HandlerConfig(
            route="/v1/embeddings",
            workload_calculator=request_workload,
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
        ))
    handlers.append(HandlerConfig(
        route="/orch/info",
        workload_calculator=lambda data: 1.0,
        allow_parallel_requests=True,
        request_parser=request_parser,
        max_queue_time=30.0,
    ))
    print(f"orch pyworker: routes {[h.route for h in handlers]}", flush=True)
    Worker(WorkerConfig(
        model_server_url=MODEL_SERVER_URL,
        model_server_port=MODEL_SERVER_PORT,
        model_log_file=os.environ["MODEL_LOG"],
        model_healthcheck_url=os.environ.get("MODEL_HEALTH_ENDPOINT", "/health"),
        handlers=handlers,
        log_action_config=LogActionConfig(
            on_load=_env_lines("MODEL_LOAD_LOG_MSG", []),
            on_error=_env_lines("MODEL_ERROR_LOG_MSGS", []),
            on_info=_env_lines("MODEL_INFO_LOG_MSGS", ['"message":"Download']),
        ),
    )).run()


if __name__ == "__main__":
    main()

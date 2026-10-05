"""PyWorker entry point: Vast's llama worker plus /v1/embeddings and /orch/info.

boot.sh copies this file to $WORKSPACE_DIR/vast-pyworker/worker.py, inside a
checkout of vast-ai/pyworker at the pinned PYWORKER_REF. That ref's
start_server.sh runs `python3 -m worker` when worker.py exists at the checkout
root, before falling back to workers/$BACKEND/worker.py.

Everything except the extra route is workers/openai/core.py's run() at that
ref, using its own request parser and benchmark, so the chat routes behave
exactly like the stock llama worker. Requests on all routes go to the model
server on 127.0.0.1:18000, worker/router.py in front of the llama-servers;
/orch/info is answered by the router itself (slots, context per slot, the
memory plan) for the shim's status page.
"""

import os

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig
from workers.openai.core import (
    MODEL_SERVER_PORT,
    MODEL_SERVER_URL,
    _env_lines,
    completions_benchmark_generator,
    request_parser,
)


def embedding_workload(data):
    """Rough token count of the input (about 4 characters per token)."""
    inp = data.get("input", "")
    items = inp if isinstance(inp, list) else [inp]
    return max(1.0, sum(len(x) if isinstance(x, str) else len(str(x)) for x in items) / 4)


def main():
    handlers = [
        HandlerConfig(
            route="/v1/completions",
            workload_calculator=lambda data: data.get("max_tokens", 0),
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
            benchmark_config=BenchmarkConfig(
                generator=completions_benchmark_generator, concurrency=10, runs=3
            ),
        ),
        HandlerConfig(
            route="/v1/chat/completions",
            workload_calculator=lambda data: data.get("max_tokens", 0),
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
        ),
    ]
    if os.environ.get("EMBED_SERVED_NAME"):
        handlers.append(HandlerConfig(
            route="/v1/embeddings",
            workload_calculator=embedding_workload,
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

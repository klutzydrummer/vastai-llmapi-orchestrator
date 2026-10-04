# One image for everything that runs at home: the shim, the watchdog and the
# deploy commands. See compose.yaml for how they're run.
FROM python:3.12-slim AS base

# git: deploy.py resolves branch/tag refs with `git ls-remote`.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY shim/requirements.txt shim/requirements.txt
RUN pip install --no-cache-dir -r shim/requirements.txt

COPY shim/ shim/
COPY deploy/ deploy/
COPY worker/ worker/
COPY docker/entrypoint.sh /usr/local/bin/orch

RUN useradd --uid 1000 --create-home orch && mkdir -p /data /config && chown orch /data
USER orch

# deploy/state.json lives on the /data volume so it survives image rebuilds
# and is shared by the watchdog and one-off deploy commands.
ENV ORCH_STATE_PATH=/data/state.json \
    ORCH_CONFIG=/config/config.toml \
    SHIM_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1
EXPOSE 8787
ENTRYPOINT ["orch"]
CMD ["shim"]

# The offline test suite plus the tools it needs (docker compose run --rm test).
FROM base AS test
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl openssl procps util-linux \
    && rm -rf /var/lib/apt/lists/*
COPY tests/ tests/
# py_compile writes __pycache__ next to the sources.
RUN chown -R orch /app
USER orch
ENTRYPOINT []
CMD ["bash", "tests/run_all.sh"]

# Last stage = what `docker build .` and compose build by default.
FROM base AS runtime

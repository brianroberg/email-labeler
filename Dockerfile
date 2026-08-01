FROM python:3.14-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY *.py ./
COPY config.toml ./

# Release identity (decision D11): the build stamps the git SHA so the daemon
# can log it at startup. Placed after the code COPY so a SHA change never busts
# the dependency layer; the default keeps un-arg'd builds working (the build is
# driven by the external agent-stack compose file).
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

HEALTHCHECK --interval=120s --timeout=5s --retries=3 \
    CMD test -f /tmp/healthcheck && \
        test $(($(date +%s) - $(stat -c %Y /tmp/healthcheck))) -lt 180

CMD ["uv", "run", "python", "daemon.py"]

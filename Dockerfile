# syntax=docker/dockerfile:1.7
# ARIE container images. Two targets share this file:
#   api — FastAPI + PDF worker pool (Python 3.14, uv workspace)
#   web — React workspace built by Vite, served by nginx, proxying /v1 and /mcp to api

ARG PYTHON_VERSION=3.14
ARG NODE_VERSION=24
ARG UV_VERSION=0.13.6
ARG NGINX_VERSION=1.29

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------
# API: resolve locked dependencies first so source edits reuse the cached layer.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS api-build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Workspace manifests only: third-party dependencies change far less often than source.
COPY pyproject.toml uv.lock ./
COPY core/pyproject.toml core/
COPY api/pyproject.toml api/
COPY worker/pyproject.toml worker/
COPY sdk/pyproject.toml sdk/
COPY plugins/ /tmp/plugins/
RUN find /tmp/plugins -name pyproject.toml | while read -r f; do \
        d="plugins/${f#/tmp/plugins/}"; mkdir -p "$(dirname "$d")"; cp "$f" "$d"; \
    done && rm -rf /tmp/plugins
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace

# Workspace members install editable: the API resolves its project root from source paths.
COPY core/ core/
COPY api/ api/
COPY worker/ worker/
COPY sdk/ sdk/
COPY plugins/ plugins/
COPY prompts/ prompts/
COPY arie.yaml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:${PYTHON_VERSION}-slim AS api
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH \
    ARIE_CONFIG=/app/data/arie.yaml \
    FASTEMBED_CACHE_PATH=/app/data/models \
    HF_HOME=/app/data/models/hf
RUN groupadd --system --gid 10001 arie \
    && useradd --system --uid 10001 --gid arie --home-dir /app --no-create-home arie
WORKDIR /app
COPY --from=api-build --chown=arie:arie /app /app
COPY --chmod=0755 docker/api-entrypoint.sh /usr/local/bin/arie-entrypoint
# The data directory becomes the named volume; pre-create it so the volume inherits ownership.
RUN install -d -o arie -g arie /app/data /app/data/papers /app/data/models \
    && chmod 0644 /app/arie.yaml
USER arie
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4).status != 200)"]
ENTRYPOINT ["arie-entrypoint"]
CMD ["uvicorn", "arie_api.routes:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--proxy-headers", "--forwarded-allow-ips", "*"]

# ---------------------------------------------------------------------------
# Web: install from the lockfile, build static assets, serve with nginx.
# ---------------------------------------------------------------------------
FROM node:${NODE_VERSION}-alpine AS web-build
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY web/ ./
# Empty by default: the browser calls same-origin /v1, which nginx proxies to the API.
ARG VITE_API_URL=
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM nginx:${NGINX_VERSION}-alpine AS web
COPY docker/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=web-build /web/dist /usr/share/nginx/html
ENV ARIE_API_UPSTREAM=api:8000
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1/healthz"]

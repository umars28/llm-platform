# Two stages: dependencies resolve once and are cached, the app layer rebuilds
# on every code change. A single-stage build reinstalls the world for a one-line
# edit, which is the difference between a 10-second and a 3-minute CI loop.
FROM python:3.12-slim AS deps

# A released version rather than :latest, which moves by design -- the same
# objection Terraform raises here against image_tag = "latest". Pinned by tag
# and not by digest on purpose: uv publishes a multi-arch index, and the only
# digests the registry hands back per-arch would build on CI's amd64 and fail
# on an arm64 laptop.
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml ./
RUN uv venv /opt/venv \
 && VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
      fastapi uvicorn[standard] httpx pyyaml prometheus-client redis

FROM python:3.12-slim AS runtime

# Patch the base image's OS packages. python:3.12-slim ships whatever Debian
# had when the tag was built, and Trivy failed this build on a CRITICAL in
# perl-base and a HIGH in libpcre2 that both had fixes available upstream.
# This trades byte-for-byte reproducibility for being patched, which is the
# right way round for an image holding provider credentials. Neither this nor
# the Python install above is pinned, so the image is not reproducible by
# digest -- see the README for what that costs.
RUN apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*

# A non-root user, and a read-only-friendly layout. Running as root in a
# container is the default and is worth the thirty seconds it takes to undo.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin gateway
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GATEWAY_CONFIG=/etc/llm-gateway/gateway.yaml

WORKDIR /app
COPY src/ ./src/
COPY config/gateway.yaml /etc/llm-gateway/gateway.yaml
ENV PYTHONPATH=/app/src

USER 10001
EXPOSE 8080

# The probe hits liveness, not readiness: a container that is alive but not
# ready should be left alone to recover, not killed and restarted.
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status==200 else 1)"

ENTRYPOINT ["python", "-m", "uvicorn", "llm_gateway.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

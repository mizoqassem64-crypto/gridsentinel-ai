# GridSentinel API V2 - container image (template)
# ------------------------------------------------------------
# Build and run on a Linux/glibc host. Containers cannot run on this
# Android/Termux device (no docker/podman daemon, no root, no proot layer).
#
#   docker build -f deploy/Dockerfile -t gridsentinel-api .
#   docker run -d --read-only --tmpfs /tmp \
#       -e GRIDSENTINEL_API_KEY="$(cat /etc/gridsentinel/api.env.key)" \
#       -p 127.0.0.1:8000:8000 gridsentinel-api
#
# The image is self-contained (stdlib HTTP API + pinned ML wheels), runs as
# an unprivileged user, never bakes the API key, and treats models/ as
# read-only. Bind is loopback; put a TLS reverse proxy in front for remote
# access. The API key is required at runtime - the container fails closed
# (exits) when GRIDSENTINEL_API_KEY is absent.

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRIDSENTINEL_HOST=127.0.0.1 \
    GRIDSENTINEL_PORT=8000

# curl is used only by the documented HEALTHCHECK probe.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/gridsentinel

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

COPY . .

RUN useradd --system --create-home --shell /usr/sbin/nologin gridsentinel \
    && chown -R gridsentinel:gridsentinel /srv/gridsentinel \
    && chmod 0755 /srv/gridsentinel/deploy/run_api.sh \
        /srv/gridsentinel/deploy/healthcheck.sh

USER gridsentinel
ENV HOME=/tmp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["/srv/gridsentinel/deploy/healthcheck.sh"]

# GRIDSENTINEL_API_KEY must be supplied at runtime (docker run -e / compose
# env_file). The launcher exits non-zero when it is missing (fail closed).
ENTRYPOINT ["/srv/gridsentinel/deploy/run_api.sh"]
#!/bin/sh
# Host/container liveness probe for the GridSentinel inference API.
#
# Exits 0 only when GET /health returns HTTP 200. Intended for use as a
# Docker HEALTHCHECK, a systemd ExecHealthCheck, or a cron/watch loop.
# POSIX sh for portability across Termux (Android), glibc Linux, and
# container images.
set -eu

HOST="${GRIDSENTINEL_HOST:-127.0.0.1}"
PORT="${GRIDSENTINEL_PORT:-8000}"

url="http://${HOST}:${PORT}/health"
status="$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null || true)"

if [ "$status" = "200" ]; then
    exit 0
fi

echo "healthcheck: ${url} returned non-200 HTTP ($status)" >&2
exit 1
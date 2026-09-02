# GridSentinel AI V2 - Deployment (Phase 6B)

Operational runbooks for the standard-library inference API
(`backend.server`). The API is single-process and stateless apart from the
read-only `models/` bundle and the process-local rate limiter.

## Invariants this deployment must preserve

1. Model artifacts are never statically served. The API exposes only
   `GET /health` and `POST /v1/assess`.
2. `trusted_source` is always `False` at the server boundary. Clients can
   never set it (rejected with 422).
3. `models/` is read-only at runtime. The engine's manifest verification is
   fail-closed: an altered/unverifiable bundle makes inference fail with 500,
   never silently degrade.
4. The API key is environment-only. It is never logged, echoed, or shipped in
   the repo. Rotate by restarting the process with a new
   `GRIDSENTINEL_API_KEY`.
5. The ML artifacts (`models/`, `datasets/`, `ai/ml/`) are never modified at
   runtime.

## Configuration

Environment-only. No config file is read.

| Variable | Default | Meaning |
|---|---|---|
| `GRIDSENTINEL_API_KEY` | *(required)* | Key checked against the `X-API-Key` header. Unconfigured `POST /v1/assess` fails closed with 503. |
| `GRIDSENTINEL_HOST` | `127.0.0.1` | Bind address. Keep loopback unless a TLS reverse proxy is in front. |
| `GRIDSENTINEL_PORT` | `8000` | Bind port. |
| `GRIDSENTINEL_MAX_BODY_BYTES` | `65536` | Maximum accepted request body. |
| `GRIDSENTINEL_RATE_LIMIT_REQUESTS` | `60` | Fixed-window request budget per client IP (in-process, per worker). |
| `GRIDSENTINEL_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window. |
| `GRIDSENTINEL_LOG_LEVEL` | `INFO` | Structured JSON log level (stderr). |

The rate limiter is intentionally in-process and per-worker; it is not a
substitute for a shared limiter/API gateway in multi-worker or multi-node
deployments.

## Option A - Termux / Android (single edge device)

Unprivileged, non-root sandbox. Bind stays on loopback by default; a Wi-Fi
LAN client requires `--host 192.168.x.x` plus a TLS tunnel (`ssh -L`, WireGuard)
since the server does no TLS itself.

```sh
# From the repository root:
GRIDSENTINEL_API_KEY="$(cat /path/to/api.key)" deploy/run_api.sh
# health:
deploy/healthcheck.sh
```

Supervision (optional): install `termux-services` for runit-style restart
(`sv start gridsentinel-api`), or run via `nohup`. Note Android may pause
background processes; a foreground Termux session or `termux-services` keeps
the service alive. Logs go to stderr - redirect to a file and rotate with
`logrotate` or an external collector.

## Option B - Linux / glibc server

Requires a Python 3.14 interpreter and the ML wheels
(`torch==2.11.0`, `numpy==2.4.4`, `pandas==3.0.5`, `scikit-learn==1.9.0`) for
the target architecture (validate wheel availability on the host). The
Termux Android wheels do not transfer; install fresh from `requirements.txt`.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
GRIDSENTINEL_API_KEY="$(cat /etc/gridsentinel/api.env.key)" deploy/run_api.sh \
    --host 127.0.0.1 --port 8000
```

Recommended production wiring:

- Run as an unprivileged user with `deploy/gridsentinel-api.service`
  (systemd): `EnvironmentFile=/etc/gridsentinel/api.env` (mode 0600),
  `Restart=on-failure`, `NoNewPrivileges=true`, `ProtectSystem=strict`.
- Put nginx/caddy as a TLS reverse proxy in front of `127.0.0.1:8000`;
  terminate TLS there and forward nothing but the two endpoints.
- `systemctl stop` sends SIGTERM; the server drains in-flight requests,
  returns exit code 0, and the unit stops.

## Option C - Container (Linux host only)

Containers cannot run on this Android/Termux device (no docker/podman
daemon, no root, no proot layer). Build and run on a Linux host using
`deploy/Dockerfile` / `docker compose -f deploy/docker-compose.yml`.

Key properties: `python:3.14-slim` base; non-root user; read-only fs
(`tmpfs /tmp`); only `127.0.0.1:8000` published (or no host port when a
proxy sidecar is used); `GRIDSENTINEL_API_KEY` injected at runtime - never
baked into the image; `HEALTHCHECK` via `deploy/healthcheck.sh`.

```sh
docker build -f deploy/Dockerfile -t gridsentinel-api .
docker run -d --read-only --tmpfs /tmp \
    -e GRIDSENTINEL_API_KEY="$(cat /etc/gridsentinel/api.env.key)" \
    -p 127.0.0.1:8000:8000 gridsentinel-api
```

## Verified behavior (Phase 6B)

- Cold start to `GET /health` is sub-second because `backend/schemas.py` /
  `backend/server.py` resolve `torch/numpy` lazily on the first request.
- SIGTERM stops the accept loop, drains in-flight handler threads, and exits
  with code 0 (systemd stop / `docker stop` / `kill` / `terminate()`).
- SIGINT (Ctrl+C) keeps the interactive KeyboardInterrupt path.

## Known operational limitations

- `server_close()` joins in-flight handler threads without a timeout; a
  deliberately stalled client can therefore prolong shutdown. No socket
  timeouts are set on request handlers.
- Per-process rate limiting only (resets on restart; no shared state).
- No built-in TLS (loopback bind + reverse proxy required for any remote
  access).
- No native log rotation (rotate externally).

## Deployment verification

Run `backend/tests/test_deployment_contract.py` (repository root), e.g.:

```sh
PYTHONPATH="$(pwd)" python3 backend/tests/test_deployment_contract.py
```

Then confirm: `git status --short` is clean (artifacts untouched); `curl
127.0.0.1:8000/health` returns `{"status":"ok",...}`; an authenticated
`POST /v1/assess` returns 200; a wrong API key returns 401.
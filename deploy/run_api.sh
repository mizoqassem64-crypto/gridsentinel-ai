#!/bin/sh
# GridSentinel inference API launcher (Phase 6B).
#
# Resolves the repository root, enforces required configuration, and execs
# the standard-library server so the process receives signals directly
# (PID 1 in a container, direct child of a supervisor, or a foreground
# Termux process). All server arguments are passed through unchanged.
#
# Usage:
#     GRIDSENTINEL_API_KEY=<key> deploy/run_api.sh [--host H] [--port P]
#
# Configuration is environment-only. The API key is never written to logs,
# files, or the process argument vector. POSIX sh for portability across
# Termux (Android), glibc Linux, and container images.
set -eu

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SELF_DIR")"

if [ -z "${GRIDSENTINEL_API_KEY:-}" ]; then
    echo "run_api.sh: GRIDSENTINEL_API_KEY is required. Set it in the" >&2
    echo "process environment or in the supervisor's EnvironmentFile." >&2
    exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
exec python3 -m backend.server "$@"
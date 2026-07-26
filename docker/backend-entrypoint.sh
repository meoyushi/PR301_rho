#!/usr/bin/env bash
# Materialise the .env the Gemini client expects (rho.llm.gemini reads
# GEMINI_API_KEY from a .env file in the working dir) from the runtime env var,
# so the key is provided at `docker run` time and never baked into the image.
set -euo pipefail

if [ "${extraction_backend:-gemini}" = "gemini" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "ERROR: extraction_backend=gemini but GEMINI_API_KEY is not set." >&2
  echo "Pass it at run time, e.g. -e GEMINI_API_KEY=... or --env-file .env" >&2
  exit 1
fi

# Write .env only if a key was supplied. A bare key or a JSON array of keys are
# both accepted by load_api_keys; we pass the value through verbatim.
if [ -n "${GEMINI_API_KEY:-}" ] && [ ! -f /app/.env ]; then
  printf 'GEMINI_API_KEY=%s\n' "${GEMINI_API_KEY}" > /app/.env
fi

exec "$@"

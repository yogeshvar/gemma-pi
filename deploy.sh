#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  source "${ROOT}/.env"
  set +a
fi
: "${PI_HOST:?Set PI_HOST in .env or environment (e.g. PI_HOST=pi@raspberrypi.local)}"
DEST="${PI_DEST:-~/gemma-pi}"
rsync -avz --delete \
  --exclude '.venv' --exclude 'venv' --exclude '__pycache__' --exclude '.git' \
  "${ROOT}/" "${PI_HOST}:${DEST}/"
echo "Synced to ${PI_HOST}:${DEST}"

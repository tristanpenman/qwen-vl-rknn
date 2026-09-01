#!/usr/bin/env bash
set -euo pipefail

echo "==> Building GPU image"
docker compose build python-gpu

echo "==> Starting GPU container"
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose up --remove-orphans -d python-gpu

echo "==> Starting shell"
docker compose exec python-gpu /usr/local/bin/docker-entrypoint.sh bash

#!/usr/bin/env bash
set -euo pipefail

echo "==> Building images"
docker compose build python

echo "==> Starting background containers"
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose up --remove-orphans -d python

echo "==> Starting shell"
docker compose exec python /usr/local/bin/docker-entrypoint.sh bash

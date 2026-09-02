#!/usr/bin/env bash
set -euo pipefail

echo "==> Building GPU image"
docker compose build python-gpu

echo "==> Starting GPU container"
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose up --remove-orphans -d python-gpu

if [ "$#" -eq 0 ]; then
  echo "==> Starting shell"
else
  echo "==> Running command"
fi

docker compose exec python-gpu /usr/local/bin/docker-entrypoint.sh "$@"

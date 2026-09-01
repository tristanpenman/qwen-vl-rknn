#!/usr/bin/env bash
set -euo pipefail

# Create a user and group matching the host's UID/GID, then drop privileges so
# files written to bind-mounted volumes retain the host user's ownership.
USER_ID="${HOST_UID:-1000}"
GROUP_ID="${HOST_GID:-1000}"
USERNAME=docker

if getent group "$GROUP_ID" >/dev/null; then
  GROUP_NAME="$(getent group "$GROUP_ID" | cut -d: -f1)"
else
  GROUP_NAME="$USERNAME"
  groupadd -g "$GROUP_ID" "$GROUP_NAME"
fi

if getent passwd "$USER_ID" >/dev/null; then
  USERNAME="$(getent passwd "$USER_ID" | cut -d: -f1)"
else
  useradd -u "$USER_ID" -g "$GROUP_ID" -m -s /bin/bash "$USERNAME" 2>/dev/null
fi

export HOME="/home/$USERNAME"

if [ "$#" -eq 0 ]; then
  set -- bash
fi

exec gosu "$USER_ID:$GROUP_ID" "$@"

#!/usr/bin/env bash
# Usage: notify.sh <title> <body> [priority: min|low|default|high|urgent]
#
# Reads NTFY_TOPIC from .env (gitignored). Set it there; never commit the real topic.
# Falls back to the default topic if .env is absent.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO_DIR/.env" ] && source "$REPO_DIR/.env"
NTFY_TOPIC="${NTFY_TOPIC:-ctrnn-gpu-ts9f2k}"

TITLE="${1:-GPU notification}"
BODY="${2:-}"
PRIORITY="${3:-default}"

curl -s \
  -H "Title: $TITLE" \
  -H "Priority: $PRIORITY" \
  -H "Tags: robot,computer" \
  -d "$BODY" \
  "https://ntfy.sh/$NTFY_TOPIC" > /dev/null

echo "[notify] sent: $TITLE"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDS="$ROOT/.pids"

stop_one () {
  local name="$1" file="$PIDS/$1.pid"
  if [[ -f "$file" ]]; then
    local pid; pid="$(cat "$file" || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "• Stopping $name (pid $pid)"
      kill "$pid" 2>/dev/null || true
      for i in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.15; done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  else
    echo "• $name: pid file not found"
  fi
}

stop_one backend
stop_one frontend
echo "✅ Stopped."

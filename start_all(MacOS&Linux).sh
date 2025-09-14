#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACK="$ROOT/CarCondition"
FRONT="$ROOT/FrontEnd/car-condition-frontend"
LOGS="$ROOT/logs"
PIDS="$ROOT/.pids"
mkdir -p "$LOGS" "$PIDS"

have(){ command -v "$1" >/dev/null 2>&1; }
open_url(){
  local url="$1"
  if have xdg-open; then xdg-open "$url" >/dev/null 2>&1 || true
  elif have open; then open "$url" >/dev/null 2>&1 || true
  fi
}

# --- .env for frontend ---
if [[ ! -f "$FRONT/.env" ]]; then
  echo "NUXT_PUBLIC_API_BASE=http://localhost:8000" > "$FRONT/.env"
else
  grep -q '^NUXT_PUBLIC_API_BASE=' "$FRONT/.env" || \
    echo "NUXT_PUBLIC_API_BASE=http://localhost:8000" >> "$FRONT/.env"
fi

# --- .env for backend ---
if [[ ! -f "$BACK/.env" ]]; then
  echo "• Creating backend .env file"
  cat > "$BACK/.env" << 'EOF'
MODEL_PATH=models/model.pth
DEVICE=cpu
# если нет интернета или не хочешь докачивать resnet weights:
RESNET_WEIGHTS=NONE
EOF
fi

# --- Python / venv ---
PYBIN="python3"; have python3 || PYBIN="python"
echo "• Using $($PYBIN --version 2>&1)"
if [[ ! -f "$BACK/.venv/bin/python" ]]; then
  echo "• Creating virtualenv in CarCondition/.venv"
  "$PYBIN" -m venv "$BACK/.venv"
fi
VENV_PY="$BACK/.venv/bin/python"
VENV_PIP="$BACK/.venv/bin/pip"

echo "• Installing backend deps"
"$VENV_PY" -m pip install --upgrade pip >/dev/null
"$VENV_PIP" install -r "$BACK/requirements.txt"

# --- Backend run (detached) ---
echo "• Starting FastAPI on :8000"
nohup "$VENV_PY" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload \
  >"$LOGS/backend.log" 2>&1 &
echo $! > "$PIDS/backend.pid"

# --- Frontend deps ---
cd "$FRONT"
if have pnpm; then
  echo "• Installing frontend deps with pnpm"
  pnpm install
  FRONT_RUN="pnpm dev"
else
  echo "• pnpm not found -> using npm"
  npm install
  FRONT_RUN="npm run dev"
fi

# --- Frontend run (detached) ---
echo "• Starting Nuxt on :3000"
nohup bash -lc "$FRONT_RUN" >"$LOGS/frontend.log" 2>&1 &
echo $! > "$PIDS/frontend.pid"

echo
echo "✅ Both services started."
echo "  Backend:  http://localhost:8000   (logs: $LOGS/backend.log)"
echo "  Frontend: http://localhost:3000   (logs: $LOGS/frontend.log)"
echo "  To stop:  ./stop_all.sh"
echo
open_url "http://localhost:3000"

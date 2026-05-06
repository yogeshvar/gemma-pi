#!/usr/bin/env bash
# One-shot dev setup: Python venv + deps + Ollama install (Pi/Linux + Mac/Homebrew) + model pull.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DRY_RUN=0
SKIP_OLLAMA=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-ollama) SKIP_OLLAMA=1 ;;
    -h|--help)
      echo "Usage: ./bootstrap.sh [--dry-run] [--skip-ollama]"
      echo "  --dry-run      Print actions only"
      echo "  --skip-ollama  Skip Ollama install / pull (venv + pip only)"
      exit 0
      ;;
  esac
done

run() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env"
  set +a
fi

MODEL="${PI_ASSISTANT_OLLAMA_MODEL:-gemma3:1b}"
HOST="${PI_ASSISTANT_OLLAMA_HOST:-http://127.0.0.1:11434}"

echo "==> Python venv + pip"
if [[ ! -d "$ROOT/venv" ]]; then
  run python3 -m venv "$ROOT/venv"
else
  echo "    venv/ already exists"
fi
run "$ROOT/venv/bin/python" -m pip install -U pip
run "$ROOT/venv/bin/pip" install -r "$ROOT/requirements.txt"

if [[ "$SKIP_OLLAMA" == 1 ]]; then
  echo "==> Skipping Ollama (--skip-ollama)"
else
  echo "==> Ollama"
  if ! command -v ollama >/dev/null 2>&1; then
    OS="$(uname -s)"
    if [[ "$OS" == "Darwin" ]]; then
      if command -v brew >/dev/null 2>&1; then
        run brew install ollama
      else
        echo "ERROR: Homebrew not found. Install from https://brew.sh then: brew install ollama"
        exit 1
      fi
    elif [[ "$OS" == "Linux" ]]; then
      echo "    Installing Ollama via https://ollama.com/install.sh (may prompt for sudo)"
      if [[ "$DRY_RUN" == 1 ]]; then
        echo '[dry-run] curl -fsSL https://ollama.com/install.sh | sh'
      else
        curl -fsSL https://ollama.com/install.sh | sh
      fi
    else
      echo "ERROR: Unsupported OS: $OS — install Ollama from https://ollama.com"
      exit 1
    fi
  else
    echo "    ollama already on PATH"
  fi

  echo "==> Checking Ollama API at $HOST"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo '[dry-run] curl API check'
  elif curl -fsS "$HOST/api/tags" >/dev/null 2>&1; then
    echo "    daemon reachable"
  else
    echo "WARN: Could not reach $HOST/api/tags — start the daemon:"
    echo "      macOS: open Ollama app, or run: ollama serve"
    echo "      Linux: sudo systemctl enable --now ollama   (or: ollama serve)"
  fi

  echo "==> Pull model $MODEL (idempotent)"
  run ollama pull "$MODEL"
fi

echo ""
echo "==> Next (not automated)"
echo "    • Whisper: build whisper.cpp; set PI_ASSISTANT_WHISPER_CLI / PI_ASSISTANT_WHISPER_MODEL"
echo "    • Piper: install binary + voice; set PI_ASSISTANT_PIPER_BIN / PI_ASSISTANT_PIPER_VOICE"
echo "    • Run: source venv/bin/activate && PI_ASSISTANT_FULLSCREEN=false python main.py"
echo "    • Memory DB: PI_ASSISTANT_MEMORY_DB_PATH (default ~/.pi-assistant/memory.db; portable: data/memory.db)"
echo "    • Prompts: edit prompts/*.md or PI_ASSISTANT_PROMPT_DIR"

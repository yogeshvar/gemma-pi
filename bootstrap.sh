#!/usr/bin/env bash
# One-shot dev setup: Python venv + pip + optional llama-server API check.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DRY_RUN=0
SKIP_LLM_SERVER=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-llm-server) SKIP_LLM_SERVER=1 ;;
    --skip-ollama)
      SKIP_LLM_SERVER=1
      echo "WARN: --skip-ollama is deprecated; use --skip-llm-server" >&2
      ;;
    -h|--help)
      echo "Usage: ./bootstrap.sh [--dry-run] [--skip-llm-server]"
      echo "  --dry-run           Print actions only"
      echo "  --skip-llm-server   Skip llama-server reachability check (venv + pip only)"
      echo "  --skip-ollama       Deprecated alias for --skip-llm-server"
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

MODEL="${PI_ASSISTANT_LLAMACPP_MODEL:-${PI_ASSISTANT_OLLAMA_MODEL:-gemma3-1b-it}}"
BASE="${PI_ASSISTANT_LLAMACPP_BASE_URL:-http://127.0.0.1:8080/v1}"
BASE="${BASE%/}"

echo "==> Python venv + pip"
if [[ ! -d "$ROOT/venv" ]]; then
  run python3 -m venv "$ROOT/venv"
else
  echo "    venv/ already exists"
fi
run "$ROOT/venv/bin/python" -m pip install -U pip
run "$ROOT/venv/bin/pip" install -r "$ROOT/requirements.txt"

if [[ "$SKIP_LLM_SERVER" == 1 ]]; then
  echo "==> Skipping llama-server check (--skip-llm-server)"
else
  echo "==> llama-server (llama.cpp)"
  echo "    Build: https://github.com/ggml-org/llama.cpp — enable the server target (e.g. cmake -DLLAMA_BUILD_SERVER=ON)."
  echo "    GGUF weights: use a Hugging Face *-GGUF repo or convert; match chat template / Jinja for your model."
  echo "    Example: llama-server -m \"\$HOME/models/your-model.gguf\" -c 4096 -t 4 --host 127.0.0.1 --port 8080"
  echo "    Tool calling (web_search): https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md"
  echo "    This app expects PI_ASSISTANT_LLAMACPP_BASE_URL (default $BASE) including the /v1 suffix."
  echo "    Model id in requests: PI_ASSISTANT_LLAMACPP_MODEL (default in .env.example: $MODEL)."
  echo "==> Checking llama-server OpenAI API at $BASE/models"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo '[dry-run] curl API check'
  elif curl -fsS "$BASE/models" >/dev/null 2>&1; then
    echo "    server reachable"
  else
    echo "WARN: Could not reach $BASE/models — start llama-server first, then set:"
    echo "      PI_ASSISTANT_LLAMACPP_BASE_URL=http://127.0.0.1:8080/v1"
    echo "      PI_ASSISTANT_LLAMACPP_MODEL=$MODEL"
  fi
fi

echo ""
echo "==> Next (not automated)"
echo "    • LLM: keep llama-server running; align PI_ASSISTANT_LLAMACPP_MODEL with your -m GGUF"
echo "    • Whisper: build whisper.cpp; set PI_ASSISTANT_WHISPER_CLI / PI_ASSISTANT_WHISPER_MODEL"
echo "    • Piper: install binary + voice; set PI_ASSISTANT_PIPER_BIN / PI_ASSISTANT_PIPER_VOICE"
echo "    • Run: source venv/bin/activate && PI_ASSISTANT_FULLSCREEN=false python main.py"
echo "    • Memory DB: PI_ASSISTANT_MEMORY_DB_PATH (default ~/.pi-assistant/memory.db; portable: data/memory.db)"
echo "    • Prompts: edit prompts/*.md or PI_ASSISTANT_PROMPT_DIR"

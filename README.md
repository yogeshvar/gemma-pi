# Pi Assistant (gemma-pi)

Offline Raspberry Pi 5 voice assistant with a Pygame face, local Whisper + **llama.cpp** (`llama-server`) + Piper pipeline, and SQLite memory.

## Clone and bootstrap

```bash
git clone <your-remote-url> gemma-pi
cd gemma-pi
./bootstrap.sh
```

`bootstrap.sh` creates **`venv/`**, installs **Python dependencies**, and (unless you pass **`--skip-llm-server`**) probes **`llama-server`** at `PI_ASSISTANT_LLAMACPP_BASE_URL` (default `http://127.0.0.1:8080/v1`). It does **not** compile llama.cpp or download GGUF weights; see **Prerequisites** below. Use **`./bootstrap.sh --dry-run`** to print steps only. The flag **`--skip-ollama`** is a deprecated alias for **`--skip-llm-server`**.

Then:

```bash
source venv/bin/activate
# optional: copy env and edit paths
cp .env.example .env
python main.py              # UI (fullscreen on Pi; use PI_ASSISTANT_FULLSCREEN=false on Mac)
python main.py --cli       # terminal loop, no Pygame
```

- **UI**: tap to talk, **Enter** from idle, **Esc** / **Q** quit, **Ctrl+C** in the terminal also exits cleanly (restores the display after fullscreen). **f** (idle) forgets last exchange. Status text sits on a **readable panel** with animations while **listening** / **thinking**. Optional **voice fillers** (short spoken ack + “thinking” lines) play over the speaker while Whisper and the LLM run (`PI_ASSISTANT_VOICE_FILLERS_ENABLED`, default on). On errors the face turns **rose**: **tap** or **Enter** to try again, **R** to dismiss to idle.
- **CLI**: empty line = listen; `f` = forget; `q` = quit.

## Configuration (`.env`)

All app keys use the prefix **`PI_ASSISTANT_`** (see [`config.py`](config.py)). Copy [`.env.example`](.env.example) to `.env`.

| Variable | Purpose |
|----------|---------|
| `PI_ASSISTANT_LLAMACPP_BASE_URL` | llama-server OpenAI base URL including **`/v1`** (default `http://127.0.0.1:8080/v1`) |
| `PI_ASSISTANT_LLAMACPP_MODEL` | `model` field in JSON requests — match your server/GGUF setup (default `gemma3-1b-it`) |
| `PI_ASSISTANT_OLLAMA_HOST` / `PI_ASSISTANT_OLLAMA_MODEL` | **Deprecated** — accepted as aliases for the llama-server URL and model id; use **`…8080/v1`**, not the old Ollama `:11434` base. |
| `PI_ASSISTANT_WEB_SEARCH_ENABLED` | `true` / `false` — register OpenAI-style **`web_search`** tool with llama-server (needs network; see below) |
| `PI_ASSISTANT_WEB_SEARCH_PROVIDER` | `ddgs` (default, no API key), `brave`, or `tavily` |
| `PI_ASSISTANT_BRAVE_API_KEY` / `PI_ASSISTANT_TAVILY_API_KEY` | Provider keys when using Brave or Tavily |
| `PI_ASSISTANT_WEB_SEARCH_MAX_RESULTS` / `MAX_CHARS` / `TIMEOUT_SECONDS` / `MAX_TOOL_ROUNDS` | Search snippet size and tool-loop cap |
| `PI_ASSISTANT_PROMPT_DIR` | Folder of `*.md` merged into the system message (default `./prompts`) |
| `PI_ASSISTANT_SYSTEM_PROMPT` | Extra system text **appended after** merged prompt files and optional web prompts |
| `PI_ASSISTANT_MEMORY_DB_PATH` | SQLite file for conversations (default `~/.pi-assistant/memory.db`) |
| `PI_ASSISTANT_WHISPER_CLI` / `PI_ASSISTANT_WHISPER_MODEL` | whisper.cpp binary + model |
| `PI_ASSISTANT_PIPER_BIN` / `PI_ASSISTANT_PIPER_VOICE` | Piper binary + ONNX voice |
| `PI_ASSISTANT_FULLSCREEN` | `true` / `false` for Pygame |
| `PI_ASSISTANT_INPUT_DEVICE` / `PI_ASSISTANT_OUTPUT_DEVICE` | sounddevice index **or** name substring (e.g. `pipewire`); unset = PortAudio default |
| `PI_ASSISTANT_VOICE_FILLERS_ENABLED` | `true` / `false` — short spoken lines while waiting on STT/LLM (default `true`) |

Deploy from a laptop: set `PI_HOST` in `.env`, run [`./deploy.sh`](deploy.sh).

### Audio (PipeWire, Bluetooth)

On **Raspberry Pi OS** with **PipeWire**, PortAudio often lists devices such as `pipewire`, `pulse`, and `default`. Pinning **`PI_ASSISTANT_INPUT_DEVICE=pipewire`** and **`PI_ASSISTANT_OUTPUT_DEVICE=pipewire`** routes capture and playback through PipeWire so Bluetooth or USB changes are handled by your session defaults instead of fragile raw indices.

At startup the app **verifies** devices and logs their names. If you see “no default device” or `-1`, run from a **desktop terminal** (not a bare SSH session without audio), ensure PipeWire is running, and check:

```bash
pactl get-default-source
pactl get-default-sink
```

**Bluetooth headsets** often work better as **output only**; use a USB or onboard **mic** for input if capture is silent or unreliable.

## Web search (optional, online)

When **`PI_ASSISTANT_WEB_SEARCH_ENABLED=true`**, Pi sends an OpenAI-style **`web_search`** tool definition to **llama-server** so the model can search the web for time-sensitive or factual questions. This **requires network access** on the device running the app (unlike the default offline stack).

- **Models / server:** Tool calling depends on the **GGUF**, **chat template**, and **llama.cpp** build. If the server rejects tools (HTTP 400) or the model cannot use them, the app **falls back to normal chat** (no live web) and logs a warning. See the upstream [function calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md) notes and pick a tool-capable setup (e.g. Llama 3.x, Mistral, Qwen 2.5-class models with matching templates).
- **Providers:** Default **`ddgs`** uses the `duckduckgo-search` package (no API key). For production, consider **Brave** or **Tavily** with keys in `.env`.
- **Prompts:** Markdown under [`prompts/web/`](prompts/web/) is merged into the **system** message only when web search is enabled (grounding, privacy, voice UX). Restart the app after changing `.env` so the merged system prompt matches the flag.

## Prompts (always sent to Gemma)

Markdown files under [`prompts/`](prompts/) are read in **lexicographic order**, joined with `---` separators, and sent as the **system** message to **llama-server**. Files named `README.md` or starting with **`_`** are skipped. If the directory is missing or empty, a small built-in fallback is used.

Edit [`prompts/01_identity.md`](prompts/01_identity.md) and [`prompts/02_style.md`](prompts/02_style.md), or point `PI_ASSISTANT_PROMPT_DIR` at another folder.

## Tests

```bash
source venv/bin/activate   # or .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v
```

## Memory (SQLite “memory file”)

Conversation memory lives in a **single SQLite database file** (WAL mode), managed by [`core/memory.py`](core/memory.py): sessions, turns, last *N* turns as chat context, session break after idle, and **forget last exchange** (two turns).

- **Default path**: `~/.pi-assistant/memory.db` (created automatically).
- **Portable clone**: set `PI_ASSISTANT_MEMORY_DB_PATH=data/memory.db` so state stays inside the repo directory. Database files under `data/` are **gitignored** (`data/*.db`).

Backup: copy the `.db` file while the app is stopped (or rely on SQLite WAL + normal shutdown).

## Caveman

- **Runtime (Gemma)**: Terse style hints live in **[`prompts/02_style.md`](prompts/02_style.md)** (inspired by [Caveman](https://github.com/JuliusBrussee/caveman); not a runtime dependency).
- **Cursor / editor**: Install the upstream skill for your own workflow, for example:

  ```bash
  npx skills add JuliusBrussee/caveman -a cursor
  ```

  Or use their installer: `curl -fsSL https://raw.githubusercontent.com/JuliusBrussee/caveman/main/install.sh | bash`

## systemd (optional)

```bash
mkdir -p ~/.config/systemd/user
cp systemd/pi-assistant.service ~/.config/systemd/user/
# edit paths if your checkout is not ~/gemma-pi
systemctl --user daemon-reload
systemctl --user enable --now pi-assistant.service
```

Ensure `venv` exists and `DISPLAY=:0` for the graphical session.

## Prerequisites (not covered by bootstrap)

**llama.cpp** (`llama-server`): build from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with the HTTP server enabled, obtain **GGUF** weights for your chosen instruct model, and run the server (example: `llama-server -m ~/models/your-model.gguf -c 4096 -t 4 --port 8080`). Keep **`PI_ASSISTANT_LLAMACPP_BASE_URL`** and **`PI_ASSISTANT_LLAMACPP_MODEL`** aligned with that process.

**Whisper** (whisper.cpp), **Piper**, and **PipeWire** are still manual installs on the Pi; set env paths or defaults in `config.py`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

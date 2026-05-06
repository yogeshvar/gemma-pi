# Pi Assistant (gemma-pi)

Offline Raspberry Pi 5 voice assistant with a Pygame face, local Whisper + Ollama + Piper pipeline, and SQLite memory.

## Clone and bootstrap

```bash
git clone <your-remote-url> gemma-pi
cd gemma-pi
./bootstrap.sh
```

`bootstrap.sh` creates **`venv/`**, installs **Python dependencies**, and (unless you pass `--skip-ollama`) ensures **Ollama** is installed (**Linux**: official install script; **macOS**: `brew install ollama` when Homebrew exists) and runs **`ollama pull`** for `PI_ASSISTANT_OLLAMA_MODEL` (default `gemma3:1b`). Use **`./bootstrap.sh --dry-run`** to print steps only.

Then:

```bash
source venv/bin/activate
# optional: copy env and edit paths
cp .env.example .env
python main.py              # UI (fullscreen on Pi; use PI_ASSISTANT_FULLSCREEN=false on Mac)
python main.py --cli       # terminal loop, no Pygame
```

- **UI**: tap to talk, **Enter** from idle, **Esc** / **Q** quit, **f** (idle) forgets last exchange.
- **CLI**: empty line = listen; `f` = forget; `q` = quit.

## Configuration (`.env`)

All app keys use the prefix **`PI_ASSISTANT_`** (see [`config.py`](config.py)). Copy [`.env.example`](.env.example) to `.env`.

| Variable | Purpose |
|----------|---------|
| `PI_ASSISTANT_OLLAMA_HOST` | Ollama base URL (default `http://127.0.0.1:11434`) |
| `PI_ASSISTANT_OLLAMA_MODEL` | Model tag (default `gemma3:1b`) |
| `PI_ASSISTANT_PROMPT_DIR` | Folder of `*.md` merged into the system message (default `./prompts`) |
| `PI_ASSISTANT_SYSTEM_PROMPT` | Extra system text **appended after** merged prompt files |
| `PI_ASSISTANT_MEMORY_DB_PATH` | SQLite file for conversations (default `~/.pi-assistant/memory.db`) |
| `PI_ASSISTANT_WHISPER_CLI` / `PI_ASSISTANT_WHISPER_MODEL` | whisper.cpp binary + model |
| `PI_ASSISTANT_PIPER_BIN` / `PI_ASSISTANT_PIPER_VOICE` | Piper binary + ONNX voice |
| `PI_ASSISTANT_FULLSCREEN` | `true` / `false` for Pygame |

Deploy from a laptop: set `PI_HOST` in `.env`, run [`./deploy.sh`](deploy.sh).

## Prompts (always sent to Gemma)

Markdown files under [`prompts/`](prompts/) are read in **lexicographic order**, joined with `---` separators, and sent as the **system** message to Ollama. Files named `README.md` or starting with **`_`** are skipped. If the directory is missing or empty, a small built-in fallback is used.

Edit [`prompts/01_identity.md`](prompts/01_identity.md) and [`prompts/02_style.md`](prompts/02_style.md), or point `PI_ASSISTANT_PROMPT_DIR` at another folder.

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

**Whisper** (whisper.cpp), **Piper**, and **PipeWire** are still manual installs on the Pi; set env paths or defaults in `config.py`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

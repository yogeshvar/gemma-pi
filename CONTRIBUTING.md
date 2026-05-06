# Contributing

1. Clone the repo and run `./bootstrap.sh` (add `--dry-run` to preview, `--skip-ollama` if you manage Ollama yourself).
2. Activate the venv: `source venv/bin/activate`.
3. Adjust `.env` from `.env.example` if paths differ on your machine.
4. Run `python main.py` (UI) or `python main.py --cli` (headless loop).

System behavior for the LLM is driven by markdown files under `prompts/` (merged in order). Optional extra text can be appended via `PI_ASSISTANT_SYSTEM_PROMPT`.

For terse **editor** assistance while working in Cursor, see the Caveman section in `README.md`.

Repository-specific Copilot instructions

1) Build / install / run (project-specific)

- Recommended local setup (uses "uv" helper as used in start.sh and .cursor rules):
  - python3 -m pip install --upgrade uv
  - uv venv .venv --python 3.13 --clear
  - Use bash: source .venv/bin/activate   (on Windows with bash/MSYS: source .venv/Scripts/activate)
  - uv add -r requirements.txt
  - uv sync
  - To download the model snapshot (keeps LLM weights local):
    - uv run app/download_model.py

- Run the API server (two options):
  - From repository root (recommended for dev):
    - uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  - Or from inside app/ (start.sh assumes this):
    - cd app && python main.py
  - Health/docs: http://localhost:8000/ and http://localhost:8000/docs

- Docker (notes from start.sh):
  - docker build --tag kushsergej-llm:latest --file app/Dockerfile app/
  - Example runtime was commented in start.sh (volume-mount model_snapshot and expose 8000)

- Tests & linting
  - No test runner (pytest/unittest) or linter (flake8/mypy/ruff) configured in repository currently.
  - If adding tests, use pytest and provide a single-test run command like: pytest path/to/test_file.py::test_name

2) High-level architecture (big picture)

- app/
  - main.py: FastAPI app + lifecycle that loads a Transformers model (AutoTokenizer, AutoModelForCausalLM) from a local snapshot (app/model_snapshot).
  - Model is loaded using local_files_only=True and selects device via torch.cuda.is_available(). The API exposes / (health) and /generate.
  - app/model_snapshot/: packaged model files (README here documents the Granite model used).
  - app/mcp-server/: MCP server implementation (mcp_server.py) exposing typed tools and prompts for tool-assisted tasks.

- scripts/: utilities and training/finetune scripts (QLoRA_fine_tune.py, embeddings.py, etc.). These are not wired into CI.

- .cursor/: AI assistant settings and MCP server definitions. .cursor/rules/* defines agent behavior and python-specific rules; .cursor/mcp.json contains MCP server configs (e.g., "exchange" server that runs the app/mcp-server/ converter via uv).

3) Key repository conventions and patterns

- "uv" workflow: project uses the "uv" helper (see start.sh and .cursor rules). Typical workflow: install/upgrade uv, create venv with uv venv, add requirements via uv add -r requirements.txt, then use uv run <script>. The .cursor python rules explicitly reference this flow.

- Model-loading and placement:
  - Models are expected in app/model_snapshot. Environment variable MODEL_PATH can override (main.py reads MODEL_PATH, default 'app/model_snapshot').
  - main.py uses local_files_only=True to avoid remote downloads during load; keep model files present locally when running.

- Device handling and generation:
  - main.py detects CUDA and uses torch.float16 when on GPU. Generation is performed through a Transformers pipeline (task='text-generation'). If adding other inference backends, follow existing pattern (device detection, dtype selection).

- Python typing style (enforced by .cursor rules):
  - Prefer `X | None` over `Optional[X]`.
  - Avoid `Any` unless strictly necessary.
  - Ensure functions include full parameter and return type annotations.

- MCP & .cursor integration:
  - .cursor/mcp.json contains MCP server entries (playwright, exchange, etc.). Copilot sessions should respect these entries: the "exchange" MCP can be started via the command in that file (it points to a uv executable invocation that runs app/mcp-server/mcp_server.py).
  - .cursor/rules/general.mdc requests that assistant replies begin by naming the model used, act as a senior DevOps engineer (concise), and always ask for approval before making changes. Respect these when automating tasks or making edits.

4) Where to look next / useful files

- app/main.py — primary web/API entrypoint and inference logic.
- app/mcp-server/mcp_server.py — example MCP tool and prompt definitions.
- app/model_snapshot/README.md — model metadata and usage examples for the bundled Granite model.
- start.sh — canonical sequence for setting up the environment (shows uv commands and Docker notes).
- .cursor/rules/*.mdc and .cursor/mcp.json — existing AI assistant rules and MCP server configs; follow them to preserve expected assistant behavior.

Questions about MCP servers

- This repo already defines MCP servers in .cursor/mcp.json (playwright, exchange, context7). If you want, configure an MCP server for local Playwright testing or a remote MCP endpoint — say whether to add a Playwright MCP server setup for end-to-end web checks.

Summary

- Created repository-specific Copilot instructions at .github/copilot-instructions.md covering: exact setup/run commands, architecture overview, and repo-specific conventions (uv workflow, typing rules, MCP setup, model placement).

If any area needs more detail (examples for adding tests, recommended linters, or explicit Docker run examples), say which and adjustments will be made.
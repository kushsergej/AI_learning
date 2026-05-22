# Repository-specific Copilot instructions

Short summary
- Project: FastAPI service that loads a local Transformers causal LM snapshot and exposes / and /generate.
- Key helpers: "uv" workflow (start.sh), local model snapshot at app/model_snapshot, small MCP server at app/mcp-server.
- Edit policy: propose edits and request approval before committing.

1) Build / install / run (project-specific)
- Recommended local setup (uses "uv" helper as used in start.sh):
  - python3 -m pip install --upgrade uv
  - uv venv .venv --python 3.13 --clear
  - Activate venv:
    - On macOS/Linux: source .venv/bin/activate
    - On Windows (PowerShell/CMD under MSYS/MSYS2/bash): source .venv/Scripts/activate
  - uv add -r requirements.txt
  - uv sync
  - Download the local model snapshot (keeps LLM weights local): uv run app/download_model.py

- Run the API server (development):
  - From repository root (recommended):
    - uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  - Or from app/ (start.sh uses this):
    - cd app && python main.py
  - Health/docs: http://localhost:8000/ and http://localhost:8000/docs

- Run the MCP server (example):
  - From repo root (dev):
    - cd app && python mcp-server/mcp_server.py
  - Or use the uv helper if preferred: uv run app/mcp-server/mcp_server.py

- Docker (notes from start.sh):
  - Build image:
    - docker build --tag kushsergej-llm:latest --file app/Dockerfile app/
  - Suggested run (Unix shells):
    - docker run -d --rm -v "$(pwd)/app/model_snapshot:/app/model_snapshot" -p 8000:8000 --name llm_backend kushsergej-llm:latest
  - Suggested run (PowerShell):
    - docker run -d --rm -v "${PWD}\app\model_snapshot:/app/model_snapshot" -p 8000:8000 --name llm_backend kushsergej-llm:latest
  - The start.sh contains commented examples; adapt for Windows path quoting or use WSL/Git Bash.

- Tests & linting:
  - No test runner or linter are enforced in CI currently.
  - Single-test run example (pytest):
    - pytest path/to/test_file.py::test_name
  - If adding linting, recommended quick commands:
    - ruff check path/to/module.py
    - pytest -q path/to/test_file.py::test_name

2) High-level architecture (big picture)
- app/
  - main.py: FastAPI app that loads AutoTokenizer and AutoModelForCausalLM from a local snapshot (app/model_snapshot). Uses an async Lifespan context manager to initialize the model at startup. Exposes:
    - GET / — healthcheck
    - POST /generate — accepts JSON {message, temperature?, max_tokens?} and returns generated text
  - mcp-server/: FastMCP-based MCP server (mcp_server.py) exposing typed tools and prompts (currency-conversion example).
  - model_snapshot/: packaged model files. main.py uses local_files_only=True — keep model files present locally or set MODEL_PATH.
- scripts/: utilities and training/finetune scripts (QLoRA_fine_tune.py, embeddings.py, etc.). Not wired into CI.
- start.sh: canonical developer flow using uv helper: create venv, install deps, download model, optional Docker steps.

3) Key conventions and repository-specific patterns
- "uv" workflow:
  - Use uv for venv creation, dependency installation, and running helper scripts. See start.sh for canonical steps.
- Model placement and loading:
  - Default model path: app/model_snapshot. Override via MODEL_PATH environment variable (export MODEL_PATH=/path/to/model).
  - main.py loads with local_files_only=True — avoid auto-downloads during startup.
- Device & dtype handling:
  - main.py detects CUDA (torch.cuda.is_available()) and uses torch.float16 when on GPU. Keep dtype/device code when modifying inference.
- Generation defaults:
  - Uses transformers pipeline(task='text-generation') with do_sample=True, top_p=0.9 by default. Requests pass temperature and max_new_tokens.
- Typing & style:
  - Functions tend to use `X | None` types; prefer explicit parameter/return annotations and avoid broad `Any` unless necessary.
- MCP patterns:
  - MCP server uses FastMCP; tools are decorated with @mcp.tool() and prompts with @mcp.prompt(). Follow these examples when adding tools.
- Windows considerations:
  - start.sh expects a Unix-like shell. For native Windows, use WSL/Git Bash or adjust commands for PowerShell.

4) Files and AI assistant configs
- Existing assistant configs found: none of .cursor/, CLAUDE.md, AGENTS.md, .windsurfrules, CONVENTIONS.md or similar appear in the repo root.
- app/mcp-server/ contains FastMCP usage examples — reference these for tool patterns.

5) Where to look next
- app/main.py — model-loading & API endpoints
- app/mcp-server/mcp_server.py — MCP tools and prompt examples
- start.sh — canonical developer setup flow
- app/model_snapshot/README.md — metadata for the bundled model snapshot

Suggested improvements (small, non-breaking)
- Add a short Docker run example for PowerShell into start.sh.
- Add a minimal pytest test and ruff config to enable quick CI feedback.
- Add a brief note in README pointing to MODEL_PATH and expected disk footprint for model_snapshot.

---

This update is a minor, documentation-only change.  

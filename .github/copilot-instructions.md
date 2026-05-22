# Repository-specific Copilot instructions

Short summary
- Project: FastAPI service that loads a local Transformers causal LM snapshot and exposes / and /generate.
- Key helpers: "uv" workflow (start.sh), local model snapshot at app/model_snapshot, simple MCP server at app/mcp-server.
- Edit policy: Please approve before applying changes to this file.

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

- Docker (notes from start.sh):
  - Build image:
    - docker build --tag kushsergej-llm:latest --file app/Dockerfile app/
  - Suggested run (mount model_snapshot, expose 8000):
    - docker run -d --rm -v "$(pwd)/app/model_snapshot:/app/model_snapshot" -p 8000:8000 --name llm_backend kushsergej-llm:latest
  - The start.sh contains commented examples; adapt for Windows path quoting.

- Tests & linting:
  - No test runner or linter are currently enforced in CI.
  - If adding tests, use pytest; run a single test with:
    - pytest path/to/test_file.py::test_name
  - Recommended one-off lint/test commands (if you add tooling):
    - ruff check path/to/module.py
    - pytest -q path/to/test_file.py::test_name

2) High-level architecture (big picture)
- app/
  - main.py: FastAPI app. Loads AutoTokenizer and AutoModelForCausalLM from a local snapshot (app/model_snapshot). Uses a Lifespan context manager to initialize the model at startup. Exposes:
    - GET / — healthcheck
    - POST /generate — accepts JSON {message, temperature?, max_tokens?} and returns generated text
  - mcp-server/: a small FastMCP-based MCP server (mcp_server.py) exposing typed tools/prompts (e.g., currency conversion example).
  - model_snapshot/: packaged model files. main.py loads with local_files_only=True — model files must be present locally.
- scripts/: utilities and training/finetune scripts (QLoRA_fine_tune.py, embeddings.py, etc.). Not wired into CI.
- start.sh: canonical developer flow using uv helper: create venv, install deps, download model, optionally build/run Docker.

3) Key conventions and repository-specific patterns
- "uv" workflow:
  - This repo uses the "uv" helper for venv management and running small scripts. Follow start.sh for canonical commands.
- Model placement and loading:
  - Models live in app/model_snapshot by default. Use MODEL_PATH environment variable to override.
  - main.py uses local_files_only=True; ensure model files exist locally before running.
- Device & dtype handling:
  - main.py detects CUDA (torch.cuda.is_available()) and uses torch.float16 when on GPU.
  - Generation uses Transformers pipeline(task='text-generation') with do_sample=True and top_p tuning. Keep heavy inference considerations in mind when modifying generation logic.
- Typing & style:
  - Prefer `X | None` over `Optional[X]`. Avoid broad `Any` unless necessary. Functions should include parameter and return type annotations.
- MCP & assistant rules:
  - An MCP server is present at app/mcp-server/mcp_server.py; it uses FastMCP.
  - Existing assistant behavior/config may reference .cursor rules (see notes below). If adding MCP servers, follow existing FastMCP patterns in that module.
- Windows specifics:
  - start.sh and some examples assume a Unix-like shell; running on native Windows may require adapting commands or using Git Bash / WSL / MSYS.

4) Where to look next
- app/main.py — primary entrypoint and model-loading lifecycle
- app/mcp-server/mcp_server.py — example MCP tools/prompts
- app/model_snapshot/README.md — model metadata and guidance for the bundled snapshot
- start.sh — step-by-step setup (uses uv helper)
- pyproject.toml / requirements.txt — dependency hints

5) Files and AI assistant configs to incorporate
- .github/copilot-instructions.md — (this file) keep up-to-date
- .cursor/* — this repo previously referenced .cursor rules in docs; if present, incorporate .cursor/mcp.json and .cursor/rules/*.mdc into assistant config
- app/mcp-server/ — the FastMCP examples should be referenced by Copilot for tool patterns

Optional notes for maintainers (short)
- Consider adding a minimal pytest test and a linter (ruff) to provide quick CI feedback.
- Add a short Docker run example in start.sh for Windows users (PowerShell variant).

End of proposed file.

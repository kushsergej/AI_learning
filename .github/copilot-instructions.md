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
    - GET / — healthcheck endpoint
    - POST /generate — accepts JSON {message, temperature?, max_tokens?} and returns generated text
  - mcp-server/: FastMCP-based MCP server (mcp_server.py) exposing typed tools and prompts (currency-conversion example as reference).
  - model_snapshot/: packaged model files (IBM Granite 3.3 2B Instruct by default). main.py uses local_files_only=True — keep model files present locally or set MODEL_PATH.
  - Dockerfile: containerizes the API server with model snapshot volumes.
- scripts/: standalone utilities not wired into CI:
  - QLoRA_fine_tune.py: parameter-efficient fine-tuning example using PEFT.
  - embeddings.py: sentence embeddings utilities.
  - huggingface_play.py, vLLM.py, sigmoid_plot.py: exploratory scripts.
- start.sh: canonical developer flow using uv helper: create venv, install deps, download model, Docker build/run examples.

3) Key conventions and repository-specific patterns
- "uv" workflow:
  - Use uv for venv creation, dependency installation, and running helper scripts. See start.sh for canonical steps.
- Model placement and loading:
  - Default model path: app/model_snapshot. Override via MODEL_PATH environment variable (export MODEL_PATH=/path/to/model).
  - download_model.py uses snapshot_download to fetch IBM Granite 3.3 2B Instruct from HuggingFace Hub. Excludes .pt/.bin weights if safetensors exist.
  - main.py loads with local_files_only=True — avoid auto-downloads during startup.
- Device & dtype handling:
  - main.py detects CUDA (torch.cuda.is_available()) and uses torch.float16 when on GPU, else defaults to float32. Keep dtype/device code when modifying inference.
- Generation defaults:
  - Uses transformers pipeline(task='text-generation') with do_sample=True, top_p=0.9 by default. Request defaults: temperature=0.2, max_tokens=256.
- Typing & style:
  - Functions use explicit parameter/return annotations. Prefer `X | None` union types; avoid broad `Any` unless necessary.
  - Logging uses logger (configured with INFO level at module startup).
- MCP patterns:
  - MCP server uses FastMCP; tools are decorated with @mcp.tool() with detailed docstrings. Prompts use @mcp.prompt(). Reference mcp_server.py for currency-conversion tool pattern.
- Windows considerations:
  - start.sh expects a Unix-like shell. For native Windows, use WSL/Git Bash or adjust commands for PowerShell.
- Error handling:
  - API endpoints return JSONResponse with status codes. Generation errors are caught and logged. Model loading failures prevent startup.

4) Where to look next
- app/main.py — model-loading, FastAPI endpoints, lifespan setup
- app/mcp-server/mcp_server.py — MCP tools and prompt patterns
- app/download_model.py — model snapshot download and placement
- start.sh — canonical developer setup flow
- pyproject.toml — dependency declarations

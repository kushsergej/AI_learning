# Repository-specific Copilot instructions

Short summary
- Project: FastAPI service that loads a local Transformers causal LM snapshot and exposes / and /generate.
- Key helpers: "uv" workflow (start.sh), local model snapshot at app/model_snapshot, small MCP server at app/mcp-server.
- Edit policy: propose edits and request approval before committing.

1) Build / install / run (project-specific)
- Recommended local setup (uses "uv" helper as used in start.sh):
  - **Prerequisites**: Python 3.13+ required. If not available, install or use `pyenv` / `asdf`.
  - python3 -m pip install --upgrade uv
  - uv venv .venv --python 3.13 --clear
  - Activate venv:
    - On macOS/Linux: source .venv/bin/activate
    - On Windows (PowerShell/CMD under MSYS/MSYS2/bash): source .venv/Scripts/activate
  - uv sync (installs from pyproject.toml + uv.lock)
  - Download the local model snapshot (keeps LLM weights local): uv run app/download_model.py
    - Downloads IBM Granite 3.3 2B Instruct (~5GB) to app/model_snapshot/; uses .safetensors, excludes .pt/.bin
    - Set MODEL_PATH env var to override snapshot location (export MODEL_PATH=/custom/path)

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

- Run the MCP client (example):
  - Connects to MCP server to call tools (currency conversion as reference):
    - uv run app/mcp-client/mcp_client.py
  - MCPClient class handles server connection, tool listing, and query processing.

- Docker (notes from start.sh):
  - **Note**: Uses app_requirements.txt (lightweight set) instead of full requirements.txt for image size.
  - Build image:
    - docker build --tag kushsergej-llm:latest --file app/Dockerfile app/
  - Suggested run (Unix shells):
    - docker run -d --rm -v "$(pwd)/app/model_snapshot:/app/model_snapshot" -p 8000:8000 --name llm_backend kushsergej-llm:latest
  - Suggested run (PowerShell):
    - docker run -d --rm -v "${PWD}\app\model_snapshot:/app/model_snapshot" -p 8000:8000 --name llm_backend kushsergej-llm:latest
  - The start.sh contains commented examples; adapt for Windows path quoting or use WSL/Git Bash.
  - MODEL_PATH env var can be set in docker run: docker run -e MODEL_PATH=/custom/path ...

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
  - mcp-server/: FastMCP-based MCP server (mcp_server.py) exposing typed tools and prompts. Currency-conversion tool (google_converter) serves as pattern reference for adding new tools.
  - mcp-client/: MCPClient class (mcp_client.py) that connects to a running MCP server via stdio, lists available tools, and processes queries using Claude.
  - model_snapshot/: packaged model files (IBM Granite 3.3 2B Instruct by default). main.py uses local_files_only=True — keep model files present locally or set MODEL_PATH.
  - Dockerfile: containerizes the API server with model snapshot volumes. Uses app_requirements.txt (optimized subset) for image build.
- scripts/: standalone utilities not wired into CI:
  - QLoRA_fine_tune.py: parameter-efficient fine-tuning example using PEFT.
  - embeddings.py: sentence embeddings utilities.
  - huggingface_play.py, vLLM.py, sigmoid_plot.py: exploratory scripts.
- start.sh: canonical developer flow using uv helper: create venv, install deps, download model, Docker build/run examples.
- .env & environment variables:
  - MODEL_PATH (optional): override model snapshot location. Defaults to 'app/model_snapshot'.
  - TZ (optional): timezone (set to 'Europe/Warsaw' in Docker by default, adjust as needed).
  - Other keys (HUGGINGFACE_TOKEN, etc.) can be added to .env and loaded via dotenv if needed.

3) Key conventions and repository-specific patterns
- Dependency management:
  - uv workflow: Use uv for venv creation, dependency installation, and running scripts. See start.sh for canonical steps.
  - pyproject.toml defines project metadata and dependencies; uv.lock locks versions.
  - requirements.txt: duplicate of pyproject.toml dependencies (for compatibility with pip workflows).
  - app_requirements.txt: lightweight subset for Docker builds (excludes heavy dev/exploratory deps).
- Model placement and loading:
  - Default model path: app/model_snapshot. Override via MODEL_PATH environment variable (export MODEL_PATH=/path/to/model).
  - download_model.py uses snapshot_download to fetch IBM Granite 3.3 2B Instruct from HuggingFace Hub. Excludes .pt/.bin weights if safetensors exist (reduces storage ~50%).
  - main.py loads with local_files_only=True — avoid auto-downloads during startup. Raise error if model not found.
- Device & dtype handling:
  - main.py detects CUDA (torch.cuda.is_available()) and uses torch.float16 when on GPU, else defaults to float32. Keep dtype/device code when modifying inference.
  - asyncio.to_thread() wraps pipeline inference to avoid blocking the event loop.
- Generation defaults:
  - Uses transformers pipeline(task='text-generation') with do_sample=True, top_p=0.9 by default. Request defaults: temperature=0.2, max_tokens=256.
  - Adjust temperature and max_tokens at request time via /generate POST body for flexibility.
- Typing & style:
  - Functions use explicit parameter/return annotations. Prefer `X | None` union types; avoid broad `Any` unless necessary.
  - Logging uses logger (configured with INFO level at module startup; timestamps and log level included).
- MCP patterns:
  - MCP server uses FastMCP; tools are decorated with @mcp.tool() with detailed docstrings. Prompts use @mcp.prompt(). Reference mcp_server.py for currency-conversion tool pattern.
  - MCP client uses AsyncExitStack and stdio_client for server connection; call session.initialize() before listing tools or calling methods.
- HTTP middleware & logging:
  - Request/response logging middleware logs all HTTP calls (method, URL, status) at INFO level for debugging.
  - Error responses use JSONResponse with appropriate status codes (200 for success, 500 for errors).
- Windows considerations:
  - start.sh expects a Unix-like shell. For native Windows, use WSL/Git Bash or adapt commands for PowerShell.
  - Docker run volume syntax differs: Unix uses $(pwd); PowerShell uses ${PWD}. Set MSYS_NO_PATHCONV=1 in Git Bash to avoid path conversion.
- Error handling:
  - API endpoints catch exceptions and return JSONResponse with error details. Model loading failures log and prevent startup (fail-fast).
  - Tools (MCP & API) return error dicts {'error': str(e)} on failure; callers check for 'error' key to handle gracefully.

4) Where to look next
- app/main.py — FastAPI setup, model loading with lifespan, endpoints (/healthcheck, /generate)
- app/download_model.py — model snapshot download & configuration
- app/mcp-server/mcp_server.py — MCP tool patterns (currency-conversion as reference)
- app/mcp-client/mcp_client.py — MCP client connection & tool invocation
- app/Dockerfile — Docker build configuration, environment setup
- app/app_requirements.txt — lightweight dependencies for Docker
- scripts/ — exploratory/utility scripts (fine-tuning, embeddings, playground code)
- start.sh — canonical developer setup (venv, deps, model download, Docker build)
- pyproject.toml — project metadata & dependencies
- .env (not tracked) — local environment overrides (MODEL_PATH, secrets, etc.)

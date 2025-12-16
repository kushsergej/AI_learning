#!/bin/bash
set -e

python3 -m pip install --upgrade uv
uv venv .venv --python 3.13 --clear
source .venv/Scripts/activate

uv add -r requirements.txt


# decouple LLM and code (save LLM weights locally)
uv run app/download_model.py


# docker build \
#     -t kushsergej-llm:v1 \
#     -f app/Dockerfile \
#     app/
# docker image ls

# docker rm -f llm_backend 2>/dev/null || true
# MSYS_NO_PATHCONV=1 docker run -d \
#     -v $(pwd)/app/model_snapshot:/app/model_snapshot \
#     -p 8000:8000 \
#     --name llm_backend \
#     kushsergej-llm:v1
# docker container ls





# ------------------------------------- #
# curl --request GET http://localhost:8000/docs
# curl --request POST --header 'Content-Type: application/json' --data '{"message": "Who is the Rome Pope now?"}' http://localhost:8000/generate
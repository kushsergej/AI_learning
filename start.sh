#!/bin/bash
set -e

python3 -m pip install --upgrade uv
uv venv .venv --python 3.13 --clear
source .venv/Scripts/activate

uv add -r requirements.txt

# decouple LLM and code (save LLM weights locally)
uv run app/download_model.py
docker build --progress=plain -t fastapi-llm -f app/Dockerfile app/
docker image ls
# docker rm -f myapp 2>/dev/null || true
# docker run -d -p 8000:8000 --name myapp fastapi-llm






# ------------------------------------- #
#  curl --silent --request POST --header 'Content-Type: application/json' --data '{"llm_prompt": "Who is the Rome Pope now?"}' http://0.0.0.0:8000/llm

# MCP https://www.revolut.com/currency-converter/convert-pln-to-eur-exchange-rate?amount=164540
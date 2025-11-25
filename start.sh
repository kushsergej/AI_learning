#!/bin/bash

python3 -m pip install --upgrade uv
uv venv .venv --python 3.13 --clear
source .venv/Scripts/activate

uv add -r requirements.txt

# app provision
cd app/
uv run download_model.py
DOCKER_BUILDKIT=1 docker build --progress=plain -t fastapi-llm -f Dockerfile .
docker rm -f myapp 2>/dev/null || true
docker run -d -p 8000:8000 --name myapp fastapi-llm






# ------------------------------------- #
#  curl --silent --request POST --header 'Content-Type: application/json' --data '{"llm_prompt": "Who is the Rome Pope now?"}' http://0.0.0.0:8000/llm

# MCP https://www.revolut.com/currency-converter/convert-pln-to-eur-exchange-rate?amount=164480
# MCP https://www.revolut.com/api/exchange/quote?amount=16448000&country=GB&fromCurrency=PLN&isRecipientAmount=false&toCurrency=EUR
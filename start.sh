#!/bin/bash

python3 -m pip install --upgrade uv
uv venv .venv --python 3.13 --clear
source .venv/Scripts/activate

uv add -r requirements.txt

# uv run main.py

# uv run scripts/embeddings.py &
# uv run scripts/huggingface_play.py &
# wait
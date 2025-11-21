#!/bin/bash

python3 -m pip install --upgrade uv
uv venv .venv --python 3.13 --clear
source .venv/Scripts/activate

uv add -r requirements.txt

# cd app/
# uv run app/download_model.py &
# uv run app/main.py &
# wait
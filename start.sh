#!/bin/bash

python -m pip install --upgrade uv
uv venv .venv --python 3.13 --clear
source .venv/Scripts/activate

uv add ipykernel dotenv openai numpy matplotlib
# uv run main.py
#!/bin/bash

python3 -m pip install --upgrade uv
uv venv .venv --python 3.13 --clear
source .venv/Scripts/activate

uv add -r requirements.txt

uv run app/main.py
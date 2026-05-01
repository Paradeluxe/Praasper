#!/usr/bin/env bash
export PATH="/tmp/ffmpeg-7.0.2-amd64-static:$PATH"
cd /mnt/d/hermes_playground/Praasper
source .venv/bin/activate
python "test scripts/grid_new_praasper.py" "$@"

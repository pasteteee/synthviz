#!/usr/bin/env bash
# System + Python dependencies for SynthViz (Debian/Ubuntu).
set -e
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
$SUDO apt-get update
$SUDO apt-get install -y ffmpeg fonts-dejavu-core
# LSP limiter (optional, the engine falls back to its own limiter)
$SUDO apt-get install -y lsp-plugins-vst3 || echo "lsp-plugins-vst3 not available, using the built-in limiter"
pip install -r requirements.txt
echo "optional: bash third_party/surge/install.sh  (Surge XT presets, ~20-30 min build)"
echo "done: export PYTHONPATH=. && python release.py --n 1 --audio-only"

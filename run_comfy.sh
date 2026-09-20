#!/usr/bin/env bash
# Local baseline launcher; upstream attribution is retained in LICENSE.
set -euo pipefail
SP_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${COMFYUI_ROOT:=${SP_REPO_ROOT}/../ComfyUI}"
: "${COMFYUI_PYTHON:=python}"
export MINIMAX_SP_LOGDIR="${MINIMAX_SP_LOGDIR:-${SP_REPO_ROOT}/results}"
export MINIMAX_SP_STREAM="${MINIMAX_SP_STREAM:-1}"
export MINIMAX_SP_EXCHANGE="${MINIMAX_SP_EXCHANGE:-all_to_all}"
export MINIMAX_SP_PROFILE="${MINIMAX_SP_PROFILE:-1}"
export MINIMAX_SP_VERIFY="${MINIMAX_SP_VERIFY:-1}"
export PYTHONUNBUFFERED=1
mkdir -p -- "$MINIMAX_SP_LOGDIR"
cd -- "$COMFYUI_ROOT"
exec "$COMFYUI_PYTHON" -B main.py \
  --cuda-device 0,1 --port 8188 --fp16-unet \
  --disable-comfy-compiler --disable-cuda-graphs \
  --use-pytorch-cross-attention --disable-auto-launch --preview-method none \
  --verbose INFO "$MINIMAX_SP_LOGDIR/comfyui_sp.log" "$@"

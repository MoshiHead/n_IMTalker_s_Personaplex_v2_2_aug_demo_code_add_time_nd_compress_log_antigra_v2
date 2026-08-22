#!/usr/bin/env bash
set -euo pipefail

# Production entrypoint: AH AudioPace with the current 2-hour static-head LoRA.
ROOT="${SPEECH2AVATAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMTALKER_DIR="${IMTALKER_DIR:-$ROOT/IMTalker}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

exec env \
  IMTALKER_DIR="$IMTALKER_DIR" \
  VENV_DIR="$VENV_DIR" \
  PORT="${PORT:-8999}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  VOICE_PROMPT="${VOICE_PROMPT:-VARM3.pt}" \
  A_CFG_SCALE="${A_CFG_SCALE:-1.13}" \
  NFE="${NFE:-5}" \
  LORA_GENERATOR_PATH="${LORA_GENERATOR_PATH:-$ROOT/checkpoints/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt}" \
  ENABLE_SEARCH="${ENABLE_SEARCH:-${ENABLE_RAG:-0}}" \
  WEB_SEARCH_ENABLED="${WEB_SEARCH_ENABLED:-}" \
  WEB_SEARCH_API_KEY="${WEB_SEARCH_API_KEY:-}" \
  WEB_SEARCH_PROVIDER="${WEB_SEARCH_PROVIDER:-tavily}" \
  ROUTER_THRESHOLD="${ROUTER_THRESHOLD:-0.40}" \
  ROUTER_RULES="${ROUTER_RULES:-1}" \
  bash "$IMTALKER_DIR/start_typeah_audio_pace_varm3.sh"

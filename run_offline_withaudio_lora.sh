#!/usr/bin/env bash
set -euo pipefail

ROOT="${SPEECH2AVATAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMTALKER_DIR="${IMTALKER_DIR:-$ROOT/IMTalker}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
VENV="${VENV:-/workspace/preprocess_5090}"

if [[ -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
fi

cd "$IMTALKER_DIR"
export PYTHONPATH="$IMTALKER_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u generator/preview_lora_checkpoint.py \
  --ref_path "${REF_PATH:-$IMTALKER_DIR/assets/3robert.jpeg}" \
  --aud_path "${AUDIO_PATH:-$IMTALKER_DIR/assets/audio_3.wav}" \
  --renderer_path "${RENDERER_PATH:-$IMTALKER_DIR/checkpoints/renderer.ckpt}" \
  --generator_path "${GENERATOR_PATH:-$IMTALKER_DIR/checkpoints/generator.ckpt}" \
  --wav2vec_model_path "${WAV2VEC_MODEL_PATH:-$IMTALKER_DIR/checkpoints/wav2vec2-base-960h}" \
  --lora_path "${LORA_PATH:-$CHECKPOINT_DIR/lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt}" \
  --out_dir "${OUT_DIR:-$ROOT/outputs/offline_withaudio_lora}" \
  --name "${NAME:-audio3}" \
  --lora_rank "${LORA_RANK:-64}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --a_cfg_scale "${A_CFG_SCALE:-1.0}" \
  --nfe "${NFE:-5}" \
  --seed "${SEED:-25}" \
  --crop

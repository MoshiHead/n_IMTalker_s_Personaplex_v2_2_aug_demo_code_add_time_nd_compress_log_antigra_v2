# Live Working Setup: Type A

Date: 2026-06-30

This is the current working live PersonaPlex + IMTalker setup. Refer to it as **Type A**.

## What Works

Type A fixes the main live failure:

```text
User speaks into the mic -> PersonaPlex hidden state changes -> avatar mouth moves
```

The fix is not to drive the avatar from PersonaPlex hidden states all the time. Instead, avatar motion is gated by **PersonaPlex assistant output audio RMS**:

```text
assistant reply audio active:
  PersonaPlex Helium hidden -> adapter -> IMTalker

assistant reply audio silent, including while the user is talking:
  silence Helium seed + zero audio -> adapter -> IMTalker
```

This means the avatar listens silently while the user talks and only lip-syncs when PersonaPlex is actually replying.

## Main Files In This Repo

Required live server files:

```text
run_live.sh
scripts/download_live_assets.sh
IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py
IMTalker/ws_av_binary_codec.py
IMTalker/liveTry.py
IMTalker/static/index_v3_binary_fullscreen.html
IMTalker/generator/FM.py
IMTalker/generator/wav2vec2.py
IMTalker/generator/helium_w2v_frontend_adapter.py
IMTalker/generator/unitalk_wav2vec_adapter.py
IMTalker/generator/train_lora.py
IMTalker/generator/preview_lora_checkpoint.py
```

Useful training/evaluation files currently included:

```text
IMTalker/tools/train_unitalk_all_layers_ddp.py
IMTalker/tools/replay_live_helium_compare_unitalk_chunk2s.py
IMTalker/tools/prepare_unitalk_2s_dataset.py
IMTalker/tools/run_causal6_sweep.sh
PERSONAPLEX_LOOKAHEAD_RMS_TRAINING.md
```

## External Assets

Large files are intentionally not committed to Git. They are downloaded by:

```bash
bash scripts/download_live_assets.sh
```

Verified Hugging Face locations:

```text
Base IMTalker:
  repo: cbsjtu01/IMTalker
  files:
    renderer.ckpt
    generator.ckpt
    wav2vec2-base-960h/*

PersonaPlex bnb4:
  repo: brianmatzelle/personaplex-7b-v1-bnb-4bit
  file:
    model_bnb_4bit.pt

PersonaPlex Mimi/tokenizer:
  repo: nvidia/personaplex-7b-v1
  files:
    tokenizer-e351c8d8-checkpoint125.safetensors
    tokenizer_spm_32k_3.model

Current Type A adapter:
  dataset: niloy629/hdtf_preprocess
  path:
    personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt

Silence Helium seed:
  dataset: niloy629/hdtf_preprocess
  path:
    personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt

Static-head with-audio LoRA:
  dataset: niloy629/hdtf_preprocess
  path:
    lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt

Optional cached blink motion:
  dataset: niloy629/hdtf_preprocess
  path:
    lora/3robert_audio3_ditto_static_motion.pt

Older frontend adapter baseline:
  dataset: niloy629/hdtf_preprocess
  path:
    personaplex_helium_w2v_frontend_adapter/checkpoints/phase2_best_wav2vec_final_loss.pt

Older UniTalk adapter baseline:
  repo: asifcsai/UniTalk
  path:
    adapters/moshi_to_adapter/adapter_phase2_latest_ep4.pt
```

No additional checkpoint upload was required for Type A because the needed large files are already present in `niloy629/hdtf_preprocess` or their source model repos.

## Fresh Pod Setup

```bash
apt-get update && apt-get install -y python3.11 python3.11-venv ffmpeg git htop tmux

cd /workspace
git clone https://github.com/nash-raf/speech2avatar.git
cd /workspace/speech2avatar

python3.11 -m venv /workspace/preprocess_5090
source /workspace/preprocess_5090/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install "setuptools==80.9.0"
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r IMTalker/requirement.txt
pip install "huggingface_hub[cli]" hf_transfer tensorboard
pip install "sphn>=0.2.0,<0.3.0" einops sentencepiece aiohttp av aiortc bitsandbytes
```

Authenticate with Hugging Face if needed:

```bash
export HF_TOKEN=...
hf auth login --token "$HF_TOKEN"
```

Download assets:

```bash
bash scripts/download_live_assets.sh
```

Install PersonaPlex/Moshi package without dependency replacement:

```bash
source /workspace/preprocess_5090/bin/activate
cd /workspace/speech2avatar/checkpoints/personaplex_bnb4
pip install -e moshi/ --no-deps
cd /workspace/speech2avatar
```

## Run Type A

```bash
cd /workspace/speech2avatar
bash run_live.sh
```

Open the pod proxy for port `8998`.

Default Type A values:

```text
image: IMTalker/assets/3robert.jpeg
cfg: 1.15
nfe: 5
chunk: 0.96s, 12 PersonaPlex hidden tokens, 24 IMTalker frames
adapter: unitalk_last_layer
adapter window: 8s lookahead
future context: 0.48s / 6 hidden tokens
assistant RMS gate threshold: 0.006
assistant RMS gate hold: 1 chunk
LoRA: with-audio rank-64 static-head LoRA
blink: disabled by default
```

## Important Runtime Controls

Change CFG:

```bash
A_CFG_SCALE=1.5 bash run_live.sh
```

Change NFE:

```bash
NFE=6 bash run_live.sh
```

Use source_5:

```bash
REF_PATH=/workspace/speech2avatar/IMTalker/assets/source_5.png bash run_live.sh
```

Tune assistant-output gate:

```bash
ASSISTANT_SPEECH_RMS_THRESHOLD=0.006 \
ASSISTANT_SPEECH_HOLD_CHUNKS=1 \
bash run_live.sh
```

Disable the gate only for debugging:

```bash
DISABLE_ASSISTANT_OUTPUT_GATE=1 bash run_live.sh
```

Expected result with the gate disabled: the mouth may move when the user talks, because PersonaPlex hidden states change while listening.

Enable cached blink composite:

```bash
ENABLE_BLINK=1 bash run_live.sh
```

## Healthy Logs

Look for:

```text
[liveTryHeliumFM] ready
[liveTryPlasticity] installed PersonaPlex graphed hidden capture
[liveTryHeliumFM_ws_binary] serving ... index_v3_binary_fullscreen.html
Uvicorn running on http://0.0.0.0:8998
```

When assistant output starts after silence, a Type A server prints:

```text
Transition from silence to assistant speech
```

## Current Known-Good Pod Example

This was the pod used to verify Type A:

```bash
ssh root@149.36.0.134 -p 46332 -i ~/.ssh/id_ed25519
```

Verified public URL at the time:

```text
https://q22wcuhhzq7x9w-8998.proxy.runpod.net/
```

Remote log:

```bash
tail -f /workspace/IMTalker/logs/live_rms_lora_3robert_cfg1p15_gate.log
```

## Why This Should Stay

The gate is not a cosmetic hack. It fixes the signal boundary:

```text
User speech / listening state should not animate the avatar mouth.
Assistant output speech should animate the avatar mouth.
```

Training can make the adapter smoother, but the assistant-output gate is the deterministic safety layer that prevents user speech from being interpreted as avatar speech.

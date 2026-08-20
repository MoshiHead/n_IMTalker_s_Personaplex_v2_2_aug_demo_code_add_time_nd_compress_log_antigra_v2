# Current Live Server: AH AudioPace

This is the current PersonaPlex + IMTalker production path. It uses the AH
AudioPace server: AJ's separate audio/video WebSockets plus paced assistant
audio writes so a renderer stall cannot flush queued audio as a speed-up burst.

## Exact 8999 Configuration

The last verified `8999` run used:

```text
server:       IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py
base model:   cbsjtu01/IMTalker generator.ckpt + renderer.ckpt
LoRA:         ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt
adapter:      personaplex_lookahead096_future048_rms50_adapter.pt
voice:        VARM3.pt
CFG:          1.13
NFE:          5
precision:    FP32 renderer, TF32 enabled
transport:    Opus, separate conversation/audio and JPEG video WebSockets
```

The model timing is a single contract. Do not change one value in isolation:

```text
PersonaPlex hidden rate: 12.5 Hz
hidden steps/chunk:      12
audio chunk:             0.96 s
adapter future context:  6 hidden steps = 0.48 s
IMTalker output:         24 frames at 25 fps
prebuffer:               1 chunk
```

The assistant-output RMS gate affects only the hidden states sent to IMTalker.
It does not change PersonaPlex reply generation. Do not replace missing
pose/gaze/camera conditions with explicit zero tensors.

## Checkpoints

All runtime weights are external to Git. `scripts/download_live_assets.sh`
retrieves them after Hugging Face login.

```text
Base generator and renderer:
  cbsjtu01/IMTalker
  generator.ckpt
  renderer.ckpt
  wav2vec2-base-960h/*

PersonaPlex 4-bit model:
  brianmatzelle/personaplex-7b-v1-bnb-4bit

PersonaPlex Mimi/tokenizer/voices:
  nvidia/personaplex-7b-v1

Live adapter and silence seed:
  niloy629/hdtf_preprocess (dataset)
  personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt
  personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt

Current 8999 LoRA:
  niloy629/hdtf_preprocess (dataset)
  live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt
  SHA-256: f34de1bd748857bc802102da578046bb10bd5f664460607d9c785ff35922e52f
  Size: 632,593,445 bytes
```

## Setup

Use Python 3.11 and a CUDA-capable PyTorch build appropriate for the machine.
The commands below are the x86_64 CUDA-pod reference setup. Do not install a
desktop CUDA wheel on Jetson.

```bash
git clone https://github.com/nash-raf/speech2avatar.git
cd speech2avatar

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install setuptools==80.9.0

# On an x86_64 CUDA 12.8 host:
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r IMTalker/requirement.txt
pip install 'huggingface_hub[cli]' hf_transfer tensorboard \
  'sphn>=0.2.0,<0.3.0' einops sentencepiece aiohttp av aiortc bitsandbytes

export HF_TOKEN=YOUR_READ_TOKEN
hf auth login --token "$HF_TOKEN"
bash scripts/download_live_assets.sh

pip install -e checkpoints/personaplex_bnb4/moshi --no-deps
```

## Run the 8999 Equivalent

```bash
cd speech2avatar
source .venv/bin/activate

PORT=8999 \
CUDA_VISIBLE_DEVICES=0 \
VENV_DIR="$PWD/.venv" \
bash run_live.sh
```

`run_live.sh` defaults to the current two-hour LoRA and AH AudioPace. Useful
overrides:

```bash
# Use another public port or GPU.
PORT=8998 CUDA_VISIBLE_DEVICES=1 VENV_DIR="$PWD/.venv" bash run_live.sh

# Compare against the base generator without loading LoRA.
DISABLE_LORA=1 PORT=8998 VENV_DIR="$PWD/.venv" \
  bash IMTalker/start_typeah_audio_pace_varm3.sh
```

Verify startup locally:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8999/
ss -ltnp | grep ':8999'
```

Healthy startup prints `lora loaded`, `installed PersonaPlex graphed hidden
capture`, and a Uvicorn listener. The page uses the browser workers under
`IMTalker/static/assets/`; keep those files beside the HTML.

## Jetson Orin Note

The code can be cloned on Jetson, but this exact deployment has not been
validated there. Jetson is ARM64 and needs NVIDIA's JetPack-matched PyTorch,
TorchVision, TorchAudio, and a compatible `bitsandbytes` build. The 7B
PersonaPlex model, Mimi codec, FP32 renderer, and browser server also need
enough shared RAM and GPU memory. Treat the x86_64 instructions above as the
reference configuration; validate model loading and latency on the target
Orin before considering it production-ready.

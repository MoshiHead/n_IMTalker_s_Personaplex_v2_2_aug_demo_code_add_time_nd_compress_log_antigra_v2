# PersonaPlex + IMTalker Live Winner Handoff

Last verified: 2026-07-16

This is the canonical handoff for the current live PersonaPlex + IMTalker system. It supersedes `winner10.md`, `live_working.md`, and the old Type A through Type AI launch notes when the request is to run the latest working server.

## Decision

- **AH AudioPace is the recommended winner.**
- **AJ NetworkIso is the fallback and A/B comparison baseline.**
- **Use `VARM3.pt` for every new run.** Do not silently substitute NATM0, NATF0, or another voice.
- Keep the IMTalker timing contract at `0.96 s`: 12 PersonaPlex hidden steps, 24 IMTalker frames, and 6 future adapter steps.
- Do not change CFG, NFE, chunk size, renderer precision, gate settings, transport, prompt, or checkpoints while diagnosing an unrelated problem.

## Naming

### AJ: Network isolation

AJ keeps assistant Opus audio on `/ws/conversation` and sends JPEG video through a separate `/ws/video?session_id=...` socket. This prevents video traffic from blocking assistant audio on one WebSocket.

### AH: Audio pacing, based on AJ

AH includes all AJ behavior and adds one targeted change in `_audio_sender`: after a GPU or renderer stall, queued audio packets are still sent at roughly the media cadence instead of being flushed as a catch-up burst. This fixed the intermittent voice speed-up.

AH does not change PersonaPlex generation, IMTalker generation, the renderer, the adapter, the RMS gate, or the browser UI.

## Canonical Local Files

Repository:

```text
/home/user/D/speech2avatar
```

Required winner files:

```text
IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py
IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AJNetworkIso.py
IMTalker/start_winner_live.sh
IMTalker/start_typeah_audio_pace_varm3.sh
IMTalker/start_typeaj_network_iso_varm3.sh
IMTalker/ws_av_binary_codec.py
IMTalker/static/index_v3_binary_fullscreen_aj_nodrop.html
IMTalker/static/assets/audio-processor-aj-nodrop.js
IMTalker/static/assets/decoderWorker.min.js
IMTalker/static/assets/decoderWorker.min.wasm
IMTalker/static/assets/encoderWorker.min-DpsJ02BN.js
IMTalker/liveTry.py
IMTalker/generator/FM.py
IMTalker/generator/FMT.py
IMTalker/generator/wav2vec2.py
IMTalker/generator/helium_w2v_frontend_adapter.py
IMTalker/generator/unitalk_wav2vec_adapter.py
IMTalker/generator/train_lora.py
IMTalker/renderer/
IMTalker/assets/3robert.jpeg
IMTalker/prompts/RB_Robert_System_Prompt_full.txt
scripts/download_live_assets.sh
```

Exact files copied from the verified pod have these SHA-256 hashes:

```text
0b432c0c094c2986817588cdfd5cc03c96e91f1a32789c74037071c82db0f9bd  AHAudioPace.py
db4c72845819be4c09b04dad8963ebe9b2ddd95e193ea5a308b6599535180b1a  AJNetworkIso.py
c090b6a5a076743055f1dd34301662405a28d5cb1636556e9de4c895ddffe4d3  ws_av_binary_codec.py
8e399dfc85d6bf1ba6b045311fc6bf28010e1d8aa2a5a1970abc3cf84d32aa72  index_v3_binary_fullscreen_aj_nodrop.html
ffa1a4efb7704a4058f56f72db94d43ed91f8ebc06e5f48873d69354382473cb2  audio-processor-aj-nodrop.js
```

## Runtime Architecture

```text
Browser microphone
  -> Opus input transport
  -> PersonaPlex Mimi input codec at 24 kHz / 80 ms cadence
  -> PersonaPlex bnb4 conversational generation

PersonaPlex assistant audio
  -> persistent Opus output stream on /ws/conversation
  -> browser AudioWorklet with no emergency queue deletion

PersonaPlex assistant hidden states at 12.5 Hz
  -> assistant-output RMS gate
  -> 0.96 s lookahead adapter window (12 hidden steps, 6 future steps)
  -> predicted Wav2Vec-compatible hidden states
  -> 25 fps IMTalker conditioning
  -> frozen static-head LoRA generator, 24 frames per chunk
  -> FP32 renderer
  -> JPEG video on the separate /ws/video socket
```

The gate only controls which hidden state drives IMTalker. It does not alter PersonaPlex turn-taking or reply generation.

## Required Model Assets

The launch script accepts overrides, but automatically checks the verified pod layout and the fresh-clone layout produced by `scripts/download_live_assets.sh`.

```text
Base generator:
  cbsjtu01/IMTalker: generator.ckpt

Renderer:
  cbsjtu01/IMTalker: renderer.ckpt

Wav2Vec model:
  cbsjtu01/IMTalker: wav2vec2-base-960h/*

PersonaPlex 4-bit model and Moshi package:
  brianmatzelle/personaplex-7b-v1-bnb-4bit

PersonaPlex tokenizer, Mimi weights, and voices.tgz:
  nvidia/personaplex-7b-v1

Live lookahead/RMS adapter:
  niloy629/hdtf_preprocess
  personaplex_lookahead_rms_adapter/checkpoints/
  personaplex_lookahead096_future048_rms50_adapter.pt

Silence Helium seed:
  niloy629/hdtf_preprocess
  personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt

Static-head with-audio LoRA:
  niloy629/hdtf_preprocess
  lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt

Default voice:
  nvidia/personaplex-7b-v1: voices.tgz -> voices/VARM3.pt
```

## Resume An Existing Pod Safely

Current verified RunPod connection at the time of this handoff:

```bash
ssh -o StrictHostKeyChecking=no -p 43020 -i ~/.ssh/id_ed25519 root@149.36.0.134
```

First inspect. Never kill a server merely because its port is the default:

```bash
ps -eo pid,args | grep -E 'AHAudioPace|AJNetworkIso' | grep -v grep
ss -ltnp | grep -E ':(8998|8999|9000)\b' || true
nvidia-smi
df -h /workspace
```

The 2026-07-16 process snapshot was:

```text
8998: AJ, NATM0, CFG 1.15
8999: AH, VARM3, CFG 1.13
9000: AH, NATF0, CFG 1.13
```

That snapshot is not the new voice policy. All future starts use VARM3 unless the user explicitly requests another voice.

## Start AH

AH defaults to port 8998, GPU 0, VARM3, CFG 1.13, and NFE 5. Override the GPU and port to avoid an existing process.

```bash
mkdir -p /workspace/IMTalker/logs
nohup env \
  IMTALKER_DIR=/workspace/IMTalker \
  VENV_DIR=/workspace/preprocess_5090 \
  CUDA_VISIBLE_DEVICES=0 \
  PORT=8998 \
  VOICE_PROMPT=VARM3.pt \
  bash /workspace/IMTalker/start_typeah_audio_pace_varm3.sh \
  > /workspace/IMTalker/logs/typeah_audio_pace_8998_varm3.log 2>&1 &
```

## Start AJ

AJ defaults to port 8999, GPU 1, VARM3, CFG 1.15, and NFE 5.

```bash
mkdir -p /workspace/IMTalker/logs
nohup env \
  IMTALKER_DIR=/workspace/IMTalker \
  VENV_DIR=/workspace/preprocess_5090 \
  CUDA_VISIBLE_DEVICES=1 \
  PORT=8999 \
  VOICE_PROMPT=VARM3.pt \
  bash /workspace/IMTalker/start_typeaj_network_iso_varm3.sh \
  > /workspace/IMTalker/logs/typeaj_network_iso_8999_varm3.log 2>&1 &
```

Do not add `&` after a `nohup` command that is already wrapped by another process supervisor.

## Fresh Pod Setup

At the time of this handoff, the winner bundle exists in the local `speech2avatar` working tree but is not yet committed or pushed. A future chat must first check whether the GitHub revision contains `IMTalker/start_winner_live.sh` and `AHAudioPace.py`. If it does not, either commit/push the scoped winner files or copy the required files listed under **Canonical Local Files** from `/home/user/D/speech2avatar` to the same relative paths on the pod. Do not assume a plain clone already contains uncommitted files.

```bash
apt-get update
apt-get install -y python3.11 python3.11-venv ffmpeg git htop tmux

cd /workspace
git clone https://github.com/nash-raf/speech2avatar.git
cd /workspace/speech2avatar

test -f IMTalker/start_winner_live.sh
test -f IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py

python3.11 -m venv /workspace/preprocess_5090
source /workspace/preprocess_5090/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install 'setuptools==80.9.0'
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r IMTalker/requirement.txt
pip install 'huggingface_hub[cli]' hf_transfer tensorboard
pip install 'sphn>=0.2.0,<0.3.0' einops sentencepiece aiohttp av aiortc bitsandbytes

export HF_TOKEN=YOUR_TOKEN
hf auth login --token "$HF_TOKEN"
bash scripts/download_live_assets.sh

cd /workspace/speech2avatar/checkpoints/personaplex_bnb4
pip install -e moshi/ --no-deps
```

Run AH directly from the clone:

```bash
mkdir -p /workspace/speech2avatar/IMTalker/logs
nohup env \
  CUDA_VISIBLE_DEVICES=0 \
  PORT=8998 \
  VOICE_PROMPT=VARM3.pt \
  bash /workspace/speech2avatar/IMTalker/start_typeah_audio_pace_varm3.sh \
  > /workspace/speech2avatar/IMTalker/logs/typeah_8998_varm3.log 2>&1 &
```

The launcher discovers assets in either `/workspace/hf_assets` and `/workspace/personaplex_bnb4`, or the fresh-clone `speech2avatar/checkpoints` layout.

## Validation

Replace the port and log path as needed:

```bash
tail -n 100 -f /workspace/IMTalker/logs/typeah_audio_pace_8998_varm3.log
curl -sS -o /tmp/imtalker_live.html -w '%{http_code} %{size_download}\n' \
  http://127.0.0.1:8998/
ss -ltnp | grep ':8998\b'
```

Healthy startup must show the model, adapter, prompt, voice, and HTTP server loading without a traceback. Browser validation must also confirm:

1. Start Conversation progresses beyond the initial frame counter.
2. Assistant speech is audible and intelligible.
3. The avatar remains quiet while the user speaks.
4. Audio does not accelerate after a render stall.
5. Audio and video continue for a multi-minute conversation.

Public RunPod URLs follow this pattern when the pod exposes the port:

```text
https://<pod-proxy-id>-8998.proxy.runpod.net/
https://<pod-proxy-id>-8999.proxy.runpod.net/
```

## Prompt Cache And Session Reset

`IMTALKER_PROMPT_STATE_CACHE=1` enables the in-process prompt prefill cache. The full prompt still comes from:

```text
IMTalker/prompts/RB_Robert_System_Prompt_full.txt
```

The cache avoids recomputing the same prompt prefix for later sessions in the same server process. It is not a replacement for the prompt text and does not intentionally preserve a prior user's conversation. A new WebSocket conversation must create fresh conversational state from the cached prompt prefix.

## Non-Negotiable Timing Contract

Do not independently change any one of these values:

```text
PersonaPlex hidden rate:       12.5 Hz
hidden steps per chunk:        12
audio/wav2vec chunk:           0.96 s
IMTalker output frames:        24 at 25 fps
adapter future context:        6 hidden steps = 0.48 s
prebuffer:                     1 chunk
renderer precision:            FP32
assistant RMS gate:            0.006, hold 1 chunk
output audio:                  persistent Opus stream
```

A 2-second chunk experiment is not the winner. Standard Wav2Vec batch inference is not this live path. Do not reintroduce static zero pose/camera/gaze tensors, sliding-window experiments, random-noise warm starts, native PersonaPlex UI, or cached-engine experiments unless the user explicitly asks for an A/B test.

## Common Failures

### Stuck at 0/12 or partial frame count

Check the traceback, PersonaPlex package import, voice path, WebSocket endpoints, browser worker assets, CUDA memory, and whether the selected GPU is already occupied. Do not assume the adapter is the cause.

### Voice speeds up after a stall

Confirm the process is `AHAudioPace.py`, not AJ or an older AG/AE server. AH's distinguishing behavior is wall-clock spacing between queued audio sends.

### Audio cuts

Do not clear or drain assistant audio queues to catch up. Confirm the no-drop AudioWorklet is loaded and inspect producer/consumer timing before changing codecs.

### Mouth moves while the user speaks

Confirm the assistant-output RMS gate is enabled and the real silence Helium seed exists. The gate belongs only on the IMTalker conditioning path.

### Lip motion goes out of distribution

First verify the exact adapter checkpoint, 12/24/6 timing contract, RMS gate, and static-head LoRA. Missing pose/gaze/camera is expected because the generator was trained with condition dropout; do not replace absent conditions with explicit zero trajectories.

## Future-Chat Rules

1. Read this file completely before changing or starting the live system.
2. Treat AH as the winner and AJ as the control.
3. Use VARM3 by default.
4. Inspect ports, PIDs, GPUs, and disk before launching.
5. Preserve existing servers unless the user explicitly says to stop or replace one.
6. Change one variable per experiment and create a new named file for structural experiments.
7. Verify actual command lines and checksums instead of trusting labels such as A, AG, AI, AJ, or AH.
8. Never paste Hugging Face tokens into documentation, Git, logs, or chat responses.
9. Before relying on a fresh Git clone, verify that the local winner bundle has been committed and pushed; otherwise sync it from the canonical local tree.

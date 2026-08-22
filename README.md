# speech2avatar

This repository contains the working PersonaPlex + IMTalker live avatar stack.

The intended runtime is a GPU pod under `/workspace`, but the paths are configurable.

## What Is Inside

- `IMTalker/` - the working IMTalker fork used for live PersonaPlex hidden-state driving, static-head LoRA, and blink motion-map compositing.
- `personaplex/` - the PersonaPlex/Moshi code needed by the live server.
- `scripts/download_live_assets.sh` - downloads the external checkpoints needed for live inference.
- `run_live.sh` - starts the live websocket server.
- `run_offline_withaudio_lora.sh` - renders original IMTalker and the
  rank-64 with-audio LoRA side by side from an image and WAV file.
- `live.md` - full setup and run notes.

Large model files are intentionally not committed to Git. They are downloaded from Hugging Face.

Read [live_working.md](live_working.md) for the current known-good **Type A**
live server, including the assistant-output RMS gate that prevents the avatar
from lip-syncing to user speech.

Read [live.md](live.md) for the broader setup and path notes.

For the current 8-second look-ahead adapter, all-layer training, RMS-weighted
fine-tuning, dataset structure, and released checkpoint, read
[PERSONAPLEX_LOOKAHEAD_RMS_TRAINING.md](PERSONAPLEX_LOOKAHEAD_RMS_TRAINING.md).

## Offline With-Audio LoRA

After downloading the checkpoints, run:

```bash
REF_PATH=/path/to/source.png \
AUDIO_PATH=/path/to/audio.wav \
./run_offline_withaudio_lora.sh
```

The checkpoint applies LoRA to the FMT attention, MLP and AdaLN layers,
the FMT decoder, and the audio, pose, gaze, and camera projections. Wav2Vec
and the renderer remain frozen. The checkpoint is downloaded from Hugging
Face by `scripts/download_live_assets.sh`; it is not committed to Git.

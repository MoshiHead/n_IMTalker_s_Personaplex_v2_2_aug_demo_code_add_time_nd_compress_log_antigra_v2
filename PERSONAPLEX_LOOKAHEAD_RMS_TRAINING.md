# PersonaPlex Look-Ahead RMS Adapter

This document is the reproducible handoff for the current PersonaPlex
Helium-to-IMTalker adapter.

## Final Live Architecture

```text
PersonaPlex hidden state [T,4096] at 12.5 Hz
-> rolling 100-token / 8-second deque
-> UniTalkWav2VecAdapter at 50 Hz
-> use layer 12 only
-> emit 48 Wav2Vec frames / .96 seconds
-> retain 24 Wav2Vec frames / .48 seconds as future context
-> interpolate emitted features to 24 video frames at 25 fps
-> IMTalker audio projection
-> IMTalker FM generator with persistent state
-> renderer
```

The live deque advances by 12 Helium tokens (`.96s`) per update. The adapter
keeps six Helium tokens (`.48s`) to the right of the emitted region as future
context. The resulting FM call is exactly 24 frames, matching IMTalker's
native `.96s` clip size and avoiding partial-chunk padding.

The live server delays the corresponding PersonaPlex reply audio by the same
six steps so audio and video remain synchronized.

## Released Checkpoint

Hugging Face dataset repository:

```text
niloy629/hdtf_preprocess
```

Files:

```text
personaplex_lookahead_rms_adapter/
  checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt
  stats/silence_helium_mean.pt
  README.md
```

Checkpoint SHA256:

```text
c7df331ab3b5f815988f3361f82d7fc16ecacaeeaa482bbbd86c42a7e27ff1b8
```

Silence seed SHA256:

```text
20a6d6eb58608d6d202bac46958e595e243635fdeeb8f04eb1afbe2ac7f2f16d
```

The checkpoint is adapter-only. It does not contain the IMTalker generator,
renderer, Wav2Vec model, PersonaPlex model, or optimizer state.

## Source Dataset

The original paired adapter dataset is stored in:

```text
dataset repo: niloy629/hdtf_preprocess
folder: personaplex_helium_w2v_frontend_adapter/
```

Restore the split archive from:

```text
personaplex_helium_w2v_frontend_adapter/dataset/
```

```bash
cat personaplex_helium_w2v_adapter_dataset.tar.zst.part-* \
  > personaplex_helium_w2v_adapter_dataset.tar.zst

tar --zstd -xf personaplex_helium_w2v_adapter_dataset.tar.zst \
  -C /workspace
```

This creates:

```text
/workspace/personaplex_frontend_adapter_dataset
```

The dataset contains 6,890 synchronized eight-second examples:

```text
helium/
  *_helium.pt             # PersonaPlex hidden states, about [100,4096], 12.5 Hz
reply_wav_24k/
  *.wav                   # PersonaPlex reply audio, mono 24 kHz
w2v_frontend_50hz/
  *.pt                    # frozen Wav2Vec projected-frontend targets
w2v_final_50hz/
  *.pt                    # frozen Wav2Vec final hidden targets
stats/
  *.pt                    # target normalization statistics
```

The dataset was produced from PersonaPlex replies to LibriSpeech prompts. The
saved Helium tensor comes from PersonaPlex/Moshi's transformer output during
autoregressive reply generation. The matching WAV is the generated assistant
reply, not the LibriSpeech input audio.

For all-layer training, generate this additional directory:

```text
w2v_all_layers_50hz/
  *_w2v_all_layers.pt     # approximately [399,12,768]
```

Command:

```bash
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$shard python \
    IMTalker/tools/precompute_w2v_all_layers.py \
    --dataset_root /workspace/personaplex_frontend_adapter_dataset \
    --wav2vec_ckpt /workspace/IMTalker/checkpoints/wav2vec2-base-960h \
    --shard_idx "$shard" \
    --num_shards 4 \
    --device cuda &
done
wait
```

## Adapter Architecture

Implementation:

```text
IMTalker/generator/unitalk_wav2vec_adapter.py
```

`UniTalkWav2VecAdapter`:

```text
input: [B,T,4096]
LayerNorm(4096)
Linear(4096 -> 768)
grouped positional Conv1d:
  kernel=128
  groups=16
12 TransformerEncoderLayer blocks:
  hidden=768
  heads=12
  FFN=3072
  pre-norm
output: [B,T,12,768]
```

Training supervises all 12 output layers against the frozen Wav2Vec hidden
layers. Live inference sends only output layer 12 to IMTalker.

The Helium sequence is interpolated from 12.5 Hz to the teacher sequence
length before entering the adapter.

The current code also supports streaming-architecture experiments:

```text
right_context_frames=-1   # original bidirectional Transformer
right_context_frames=0    # causal Transformer, no future attention
right_context_frames=N    # limited future context of N 50 Hz frames
recurrent_layers=N        # optional GRU residual block before Transformer
```

These variants are launched by `IMTalker/tools/run_causal6_sweep.sh`. They are
kept as experiments; the current working Type A server still uses the
look-ahead RMS checkpoint listed above.

## Base Look-Ahead Training

Training script:

```text
IMTalker/tools/train_unitalk_all_layers_ddp.py
```

Loss is evaluated only on the emitted `.96s` region:

```text
tail_loss_frames=48
future_context_frames=24
loss = MSE + 0.1 * cosine_loss
```

The adapter still receives the complete eight-second sequence. The final 24
teacher frames are context only and are excluded from loss.

The working run used four GPUs with two samples per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --nproc_per_node=4 \
  IMTalker/tools/train_unitalk_all_layers_ddp.py \
  --dataset_root /workspace/personaplex_frontend_adapter_dataset \
  --save_dir /workspace/exps/unitalk_all12_lookahead096_future048_gb8_e30 \
  --epochs 30 \
  --batch_size 2 \
  --num_workers 2 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --lambda_mse 1.0 \
  --lambda_cos 0.1 \
  --tail_loss_frames 48 \
  --future_context_frames 24 \
  --checkpoint_epochs 5,10,20,30 \
  --seed 1234 \
  --precision bf16
```

Effective global batch size was 8. The epoch-30 loss was:

```text
total: 0.033703
MSE:   0.020617
cos:   0.130867
```

## RMS-Weighted Fine-Tuning

RMS is a training label only. It is not passed to the adapter at inference and
there is no hard RMS gate.

Create aligned RMS masks:

```bash
python IMTalker/tools/precompute_rms_masks.py \
  --dataset_root /workspace/personaplex_frontend_adapter_dataset \
  --threshold_db -50
```

The script:

1. Computes reply-audio RMS every 20 ms at 50 Hz.
2. Converts RMS to dBFS.
3. Labels frames below `-50 dBFS` as silence.
4. Aligns labels to each Wav2Vec teacher sequence.

About 20% of frames were labeled silence. Loss weights were:

```text
speech frame:  1
silence frame: 4
```

The weighted objective is:

```text
weight = 1 + 3 * silence
loss = weighted_MSE + 0.1 * weighted_cosine
```

Silent frames are trained toward their real Wav2Vec silence targets, not zero.

Fine-tuning command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --nproc_per_node=4 \
  IMTalker/tools/train_unitalk_all_layers_ddp.py \
  --dataset_root /workspace/personaplex_frontend_adapter_dataset \
  --save_dir /workspace/exps/unitalk_lookahead096_future048_rms50_gb8_e10 \
  --epochs 10 \
  --batch_size 2 \
  --num_workers 2 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --lambda_mse 1.0 \
  --lambda_cos 0.1 \
  --tail_loss_frames 48 \
  --future_context_frames 24 \
  --rms_dir /workspace/personaplex_frontend_adapter_dataset/rms_50hz \
  --silence_weight 3.0 \
  --resume /workspace/exps/unitalk_all12_lookahead096_future048_gb8_e30/checkpoints/epoch_000030.pt \
  --checkpoint_epochs 5,10 \
  --seed 2234 \
  --precision bf16
```

Final weighted epoch-10 loss:

```text
total: 0.024284
MSE:   0.014634
cos:   0.096501
```

On the saved live session, RMS weighting reduced latent motion during silence
by about 17%, while speech motion changed by about 2%.

## Startup Silence Seed

An all-zero Helium deque is out of distribution and produced poor initial
frames. The live server now initializes its deque with the mean of 119,590
RMS-labeled PersonaPlex silence tokens:

```text
stats/silence_helium_mean.pt
```

Pass it with:

```text
--silence_helium_path /path/to/silence_helium_mean.pt
```

## Live Replay Validation

Replay a saved live Helium session using the exact live contract:

```bash
python IMTalker/tools/replay_live_helium_compare_unitalk_chunk2s.py \
  --dump_dir /path/to/saved_live_session \
  --mode lookahead \
  --generator_path /workspace/IMTalker/checkpoints/generator.ckpt \
  --renderer_path /workspace/IMTalker/checkpoints/renderer.ckpt \
  --adapter_path /path/to/personaplex_lookahead096_future048_rms50_adapter.pt \
  --adapter_type unitalk_last_layer \
  --ref_path /workspace/IMTalker/assets/3robert.jpeg \
  --out_dir /workspace/replay \
  --nfe 5 \
  --a_cfg_scale 1.15 \
  --fix_noise_seed \
  --rolling_window_tokens 100 \
  --rolling_emit_tokens 12 \
  --rolling_future_tokens 6
```

## Important Failed Experiment

The two-second experiment emitted 50 video frames per FM call. IMTalker FM's
native clip is 24 frames, so it processed the call as:

```text
24 + 24 + 2
```

The final two frames were padded to 24, and the old streaming state retained
the padded future. Motion jumps at two-second boundaries were about 10-12
times normal motion. Do not use arbitrary FM call lengths without repairing
partial-chunk state handling.

The final `.96s` design emits exactly 24 frames. Measured boundary motion was
`0.78x` ordinary within-chunk motion, so the periodic reset was removed.

## Fresh-Clone Live Server

The committed defaults reproduce the current live server:

```bash
git clone https://github.com/nash-raf/speech2avatar.git
cd speech2avatar

# Create/install the Python 3.11 environment as described in live.md.
export HF_TOKEN=...
hf auth login --token "$HF_TOKEN"

bash scripts/download_live_assets.sh
bash run_live.sh
```

`scripts/download_live_assets.sh` downloads:

- Original IMTalker generator, renderer, and Wav2Vec.
- The look-ahead RMS adapter and real-silence Helium seed.
- The with-audio static-head LoRA generator checkpoint.
- PersonaPlex bnb4 code and weights.
- Gated PersonaPlex Mimi and tokenizer assets.

The Hugging Face account must have accepted the
`nvidia/personaplex-7b-v1` license.

Current `run_live.sh` defaults:

```text
image: IMTalker/assets/3robert.jpeg
CFG: 1.15
NFE: 5
adapter mode: lookahead
window: 100 tokens / 8 seconds
advance: 12 tokens / .96 seconds
future context: 6 tokens / .48 seconds
FM output: 24 frames
static generator: rank-64 with-audio LoRA
blink composite: disabled (set ENABLE_BLINK=1 to enable)
port: 8998
```

## Relevant Files

```text
IMTalker/generator/unitalk_wav2vec_adapter.py
IMTalker/tools/precompute_w2v_all_layers.py
IMTalker/tools/train_unitalk_all_layers_ddp.py
IMTalker/tools/precompute_rms_masks.py
IMTalker/tools/prepare_unitalk_2s_dataset.py
IMTalker/tools/replay_live_helium_compare_unitalk_chunk2s.py
IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py
run_live.sh
scripts/download_live_assets.sh
```

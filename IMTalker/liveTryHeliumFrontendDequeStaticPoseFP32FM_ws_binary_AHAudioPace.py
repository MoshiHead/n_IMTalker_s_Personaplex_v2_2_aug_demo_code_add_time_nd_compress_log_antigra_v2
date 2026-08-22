"""PersonaPlex + IMTalker AJ server with split audio/video WebSockets.

Raw PersonaPlex audio is never stored in or cleared with the video queue.
Assistant Opus audio stays on /ws/conversation, while JPEG video frames are
sent through /ws/video?session_id=... to avoid websocket head-of-line blocking.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import json
import os
import queue

import ws_av_binary_codec as _wsbin
import sys
import threading
import time
import traceback
import types
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sphn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.FM import FMGenerator
from generator.train_lora import apply_lora_to_model
from generator.helium_w2v_frontend_adapter import HeliumToWav2VecFrontendAdapter
from generator.unitalk_wav2vec_adapter import UniTalkLastLayerLiveAdapter
from generator.options.base_options import BaseOptions
from generator.wav2vec2 import Wav2VecModel
if os.environ.get("IMTALKER_CACHED_ENGINE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    from liveTry_cached import MoshiOnlyEngine
else:
    from liveTry import MoshiOnlyEngine
from renderer.models import IMTRenderer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 24_000          # Mimi sample rate (24kHz)
VIDEO_FPS = 25              # IMTalker frame rate
MIMI_FRAME_SIZE = 1_920     # samples per Mimi frame (80ms @ 24kHz)
MAIN_CODEBOOKS = 8          # codebooks used for Helium input embeddings
PREBUFFER_CHUNKS = 0        # produce this many chunks before sender starts pacing
WAV2VEC_SR = 16_000


class PlasticityProjectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_ln = nn.LayerNorm(4096)
        self.net = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_ln(x))


class PlasticityUpsampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(768, 768, kernel_size=4, stride=4)

    def forward(self, low: torch.Tensor, target_len: int) -> torch.Tensor:
        y = self.up(low.transpose(1, 2).contiguous())
        if y.shape[-1] != int(target_len):
            y = F.interpolate(y, size=int(target_len), mode="linear", align_corners=False)
        return y.transpose(1, 2).contiguous()


class PlasticityCausalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.15,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(768)
        self.ff = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 768),
        )
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop1(attn)
        h = self.norm2(x)
        x = x + self.drop2(self.ff(h))
        return x


class PlasticityCausalTransformer(nn.Module):
    def __init__(self, max_len: int = 2048) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([PlasticityCausalBlock() for _ in range(8)])
        self.norm = nn.LayerNorm(768)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.causal_mask[: x.shape[1], : x.shape[1]].to(device=x.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class StudioNativeLiveAdapter(nn.Module):
    """Frontend fp32 adapter live wrapper.

    Training contract:
      raw 12.5Hz Helium -> Wav2Vec2 projected frontend [T50, 768]
      live contract:
      projected frontend -> frozen Wav2Vec2 encoder -> final hidden -> IMTalker audio_projection.
    """

    def __init__(self, wav2vec_model_path: str, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.model = HeliumToWav2VecFrontendAdapter(num_layers=int(num_layers), dropout=float(dropout))
        self.wav2vec = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).eval().float()
        for param in self.wav2vec.parameters():
            param.requires_grad_(False)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        return self.model.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def forward_single(self, source: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = source.unsqueeze(0).contiguous()
        target_len = int(target_len)
        frontend_len = max(1, target_len * 2)
        frontend50 = self.model(src.float(), target_len=frontend_len).float()
        final50 = self.wav2vec.encode_from_projected_frontend(frontend50).last_hidden_state.float()
        final25 = F.interpolate(
            final50.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)[0].float().contiguous()
        return frontend50[0].float().contiguous(), final50[0].float().contiguous(), final25


class MoshiOnlyEngineWithHidden(MoshiOnlyEngine):
    """Moshi reply engine that also returns the main LM hidden for each generated step.

    Layer[-2] is exposed as a native LMGen output so Moshi can keep CUDA graph
    replay enabled. We do not use Python forward hooks in this path.
    """

    def __init__(
        self,
        *args,
        capture_layer: int = -2,
        thinking_sound_path: str = "",
        search_max_filler_sec: float = 6.0,
        **kwargs,
    ) -> None:
        self.tf_capture_layer = int(capture_layer)
        super().__init__(*args, **kwargs)

        # Forensic finding (conversation_logs_1/2/3): the old fixed 25-chunk
        # (~2.0s) filler cap was shorter than observed real-world retrieval+
        # compression latency -- compression alone was observed taking
        # ~2.5-2.7s, and web-search-augmented turns 2.4-3.7s end to end, so
        # the cap fired first in 5/5 logged search turns and discarded a
        # correctly-computed answer every time. Overriding the class-level
        # default (see _SEARCH_MAX_FILLER_FRAMES below) here with an instance
        # attribute computed from a CLI-configurable seconds value (default
        # 6.0s: comfortably above the worst observed combined latency of
        # ~3.7s, including margin for GPU-contention slowdown).
        self._SEARCH_MAX_FILLER_FRAMES = max(1, round(float(search_max_filler_sec) * TARGET_SR / MIMI_FRAME_SIZE))

        # "Thinking sound": played in place of the model's own audio ONLY while
        # an online search is actually running (see _start_thinking_sound and
        # its two call sites). It is deliberately never played on turns the
        # model answers from its own knowledge, nor while the router is still
        # deciding -- in both of those cases the model is about to speak within
        # a few hundred milliseconds, and a filler clip would delay and mask a
        # reply that needed no waiting at all. Loaded once here via the same
        # resample-to-24kHz-mono helper used for --audio_path.
        self.thinking_sound_pcm: np.ndarray | None = None
        self._thinking_sound_cursor = 0
        if thinking_sound_path:
            if Path(thinking_sound_path).is_file():
                try:
                    self.thinking_sound_pcm = load_audio_24k(thinking_sound_path)
                    print(
                        f"[liveTryPlasticity][search] thinking sound loaded: {thinking_sound_path} "
                        f"({self.thinking_sound_pcm.shape[0] / TARGET_SR:.2f}s)",
                        flush=True,
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    print(
                        f"[liveTryPlasticity][search] failed to load thinking sound "
                        f"{thinking_sound_path!r}: {e!r}\n{tb}",
                        flush=True,
                    )
                    self.conv_logger.error("thinking_sound_load", e, tb)
                    self.thinking_sound_pcm = None
            else:
                print(
                    f"[liveTryPlasticity][search] thinking sound path not found: {thinking_sound_path} "
                    f"-- will stay silent during the search instead",
                    flush=True,
                )

        self._install_graph_hidden_capture()

    def _next_thinking_sound_chunk(self) -> np.ndarray:
        """Next MIMI_FRAME_SIZE samples of the thinking sound, looping
        seamlessly. Advances self._thinking_sound_cursor and
        self._thinking_sound_play_count (incremented every time the clip
        wraps back to its start, i.e. every completed extra play-through)."""
        pcm = self.thinking_sound_pcm
        n = pcm.shape[0]
        out = np.empty(MIMI_FRAME_SIZE, dtype=np.float32)
        pos = 0
        cursor = self._thinking_sound_cursor % n
        while pos < MIMI_FRAME_SIZE:
            take = min(MIMI_FRAME_SIZE - pos, n - cursor)
            out[pos:pos + take] = pcm[cursor:cursor + take]
            pos += take
            new_cursor = (cursor + take) % n
            if new_cursor < cursor or (new_cursor == 0 and take > 0):
                self._thinking_sound_play_count += 1
            cursor = new_cursor
        self._thinking_sound_cursor = cursor
        return out

    def _will_actually_search(self) -> bool:
        """Whether a 'search' decision would really reach the network.

        The thinking sound is gated on this, not merely on the router saying
        'search': with web search disabled or no API key configured, the
        pipeline decides to search, discovers it cannot, and falls straight
        back to the model's own knowledge. Playing a 'searching...' cue for a
        search that never happens would tell the user something untrue."""
        return bool(self.web_search_enabled and self.web_search_api_key)

    def _start_thinking_sound(self, turn_id, transcript: str = "") -> None:
        """Begin the thinking sound. MUST run on the GPU thread -- it mutates
        the playback state that _step() reads every 80ms.

        Idempotent: the rules path starts the sound synchronously the instant
        it commits to searching, while the model-router path signals from the
        background thread, and both can land for the same turn."""
        # Withhold the model's words for the whole search, whether or not the
        # filler clip is available -- these are independent concerns (one is
        # what the user hears, the other is what the model commits to).
        if self.suppress_text_during_search:
            self.suppress_text_until_ref = True
        if self.thinking_sound_pcm is None or self.search_thinking_active:
            return
        self.search_thinking_active = True
        self._thinking_sound_started_at = time.perf_counter()
        self._thinking_sound_play_count = 1
        self.conv_logger.event("thinking_sound_start", transcript=transcript)
        self.conv_logger.narrate_thinking_start(turn_id)

    def _stop_thinking_sound(self, turn_id, reason: str, reason_text: str) -> None:
        """Shared stop logic for both the real-ref-ready and filler-timeout
        paths in _consume_pending -- keeps duration/loop-count
        bookkeeping in one place."""
        if not self.search_thinking_active:
            return
        self.search_thinking_active = False
        duration_s = max(0.0, time.perf_counter() - self._thinking_sound_started_at)
        clip_duration_s = (
            self.thinking_sound_pcm.shape[0] / TARGET_SR if self.thinking_sound_pcm is not None else 0.0
        )
        self.conv_logger.event(
            "thinking_sound_stop", reason=reason, duration_s=round(duration_s, 2),
            play_count=self._thinking_sound_play_count,
        )
        self.conv_logger.narrate_thinking_stop(
            turn_id, reason_text, duration_s, self._thinking_sound_play_count, clip_duration_s
        )
        # Not a pipeline cost -- it runs concurrently with the search -- but it
        # is exactly how much of the wait the user was given something to listen
        # to, which is the number to compare the search latency against.
        self.conv_logger.latency.stage(
            getattr(self, "_latency_turn_id", None), "thinking_sound", duration_s,
            note=f"stopped because {reason}, played {self._thinking_sound_play_count}x",
        )

    def _install_graph_hidden_capture(self) -> None:
        lm_model = self.lm
        lm_gen = self.lm_gen
        if hasattr(lm_gen, "prepare_step_input") and hasattr(lm_gen, "process_transformer_output"):
            @torch.no_grad()
            def personaplex_step_with_hidden(
                self_gen,
                input_tokens: torch.Tensor = None,
                moshi_tokens: torch.Tensor = None,
                text_token: torch.Tensor = None,
                depformer_replace_tokens: torch.Tensor | None = None,
            ):
                prepared = self_gen.prepare_step_input(input_tokens, moshi_tokens, text_token)
                if prepared is None:
                    return None
                input_, provided_, target_, model_input_position, target_position = prepared
                state = self_gen._streaming_state
                transformer_out, text_logits = state.graphed_main(input_)
                output = self_gen.process_transformer_output(
                    transformer_out,
                    text_logits,
                    provided_,
                    target_,
                    model_input_position,
                    target_position,
                )
                return output, transformer_out, transformer_out

            lm_gen._step = types.MethodType(personaplex_step_with_hidden, lm_gen)
            # This override forwards text_token to prepare_step_input, and
            # process_transformer_output honors a provided token instead of
            # sampling one. That is what lets _step() force the model silent
            # while a search is in flight (see _step / suppress_text_until_ref).
            self._step_supports_text_token = True
            lm_gen.streaming_forever(1)
            self._warmup_runtime()
            print("[liveTryPlasticity] installed PersonaPlex graphed hidden capture", flush=True)
            return

        from moshi.models.lm import scatter_with_mask_
        from moshi.modules.transformer import create_sin_embedding
        from moshi.utils.sampling import sample_token

        capture_layer = int(self.tf_capture_layer) % len(lm_model.transformer.layers)

        old_state = getattr(lm_gen, "_streaming_state", None)
        if old_state is not None:
            with contextlib.suppress(Exception):
                old_state.__exit__(None, None, None)
            with contextlib.suppress(Exception):
                lm_gen._stop_streaming()

        def forward_text_with_layer(self_lm, sequence, sum_condition=None, cross_attention_src=None):
            B, K, S = sequence.shape
            assert K == self_lm.num_codebooks, (K, self_lm.num_codebooks)
            input_sequence = sequence
            input_ = None
            for cb_index in range(self_lm.num_audio_codebooks):
                audio_emb = self_lm.emb[cb_index](input_sequence[:, cb_index + self_lm.audio_offset])
                input_ = audio_emb if input_ is None else input_ + audio_emb
            text_emb = self_lm.text_emb(input_sequence[:, 0])
            input_ = text_emb if input_ is None else input_ + text_emb
            if sum_condition is not None:
                input_ = input_ + sum_condition.to(input_)
            if cross_attention_src is not None:
                cross_attention_src = cross_attention_src.to(input_)

            transformer = self_lm.transformer
            _, T, C = input_.shape
            dtype_input = input_.dtype
            state = transformer._streaming_state
            if state is None:
                offsets = torch.zeros(1, dtype=torch.long, device=input_.device)
            else:
                offsets = state.offsets

            x = input_
            if transformer.positional_embedding in {"sin", "sin_rope"}:
                positions = torch.arange(T, device=x.device).view(1, -1, 1)
                positions = positions + offsets.view(-1, 1, 1)
                pos_emb = create_sin_embedding(positions, C, max_period=transformer.max_period, dtype=x.dtype)
                x = x + transformer.positional_scale * pos_emb

            captured = x
            for idx, layer in enumerate(transformer.layers):
                x = layer(x, cross_attention_src=cross_attention_src)
                if idx == capture_layer:
                    captured = x

            if state is not None:
                state.offsets[:] = torch.where(state.exec_mask, state.offsets + T, state.offsets)

            transformer_out = x.to(dtype_input)
            layer_hidden = captured.to(dtype_input)
            if self_lm.out_norm:
                transformer_out = self_lm.out_norm(transformer_out)
            text_logits = self_lm.text_linear(transformer_out)
            text_logits = text_logits[:, None]
            return transformer_out, text_logits, layer_hidden

        @torch.no_grad()
        def step_with_layer(self_gen, input_tokens: torch.Tensor, depformer_replace_tokens: torch.Tensor | None = None):
            state = self_gen._streaming_state
            if state is None:
                raise RuntimeError("You should wrap those calls with a `with lm_gen.streaming(): ...`.")
            lm_model_local = self_gen.lm_model

            assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = input_tokens.shape
            assert B == state.batch_size, f"Got a batch size {B}, expected {state.batch_size}"
            assert S == 1, "Only support being given steps one by one."
            needed_tokens = lm_model_local.num_codebooks - lm_model_local.dep_q - 1
            assert Ki >= needed_tokens, f"We expect {needed_tokens} tokens from the user stream, got {Ki}."
            if Ki > needed_tokens:
                input_tokens = input_tokens[:, :needed_tokens, :]

            CT = state.cache.shape[2]
            delays = self_gen.delays_cuda[lm_model_local.dep_q + 1:]
            write_positions = (state.offsets[:, None, None] + delays[:, None]) % CT
            scatter_with_mask_(state.cache[:, lm_model_local.dep_q + 1:], -1, write_positions, input_tokens, state.exec_mask[:, None, None])

            is_init = state.offsets[:, None, None] <= self_gen.delays_cuda[:, None]
            is_init |= ~state.exec_mask[:, None, None]
            positions = (state.offsets % CT)[:, None, None].expand_as(is_init)
            input_ = state.cache.gather(dim=2, index=positions)
            input_ = torch.where(is_init, state.initial, input_)

            if self_gen.check:
                assert not (input_ == lm_model_local.ungenerated_token_id).any(), (state.offsets, input_)
                assert (input_[:, lm_model_local.audio_offset:] <= lm_model_local.card).all(), input_
                assert (input_[:, :1] <= lm_model_local.text_card).all()

            zero = torch.full((1,), lm_model_local.zero_token_id, dtype=torch.long, device=input_.device)
            if self_gen.cfg_coef != 1.:
                if state.cfg_is_masked_until is not None:
                    limit = self_gen.delays_cuda[:, None] + state.cfg_is_masked_until.view(-1, 1, 1)
                    is_zeroed = state.offsets[:, None, None] <= limit
                    masked = torch.where(is_zeroed & ~is_init, zero, input_)
                    input_ = torch.cat([input_, masked], dim=0)
                else:
                    input_ = input_.repeat(2, 1, 1)
                if self_gen.cfg_is_no_text:
                    input_[B:, :1] = torch.where(~is_init[:, :1], zero, input_[B:, :1])

            transformer_out, text_logits, layer_hidden = state.graphed_main(input_, state.condition_sum, state.condition_cross)
            if self_gen.cfg_coef != 1.:
                logits, logits_null = text_logits.chunk(2)
                if self_gen.cfg_is_no_text:
                    text_logits = logits
                    layer_hidden = layer_hidden[:B]
                else:
                    text_logits = logits_null + (logits - logits_null) * self_gen.cfg_coef
                    layer_hidden = layer_hidden[:B]

            if self_gen.on_text_logits_hook:
                self_gen.on_text_logits_hook(text_logits)
            text_token = sample_token(text_logits.float(), self_gen.use_sampling, self_gen.temp_text, self_gen.top_k_text)
            assert text_token.dim() == 3, text_token.shape
            assert text_token.shape[2] == 1
            assert text_token.shape[1] == 1, "Only one text stream supported."
            text_token = text_token[:, 0, 0]
            if self_gen.on_text_hook is not None:
                self_gen.on_text_hook(text_token)

            if state.graphed_depth is None:
                audio_tokens = None
            else:
                if depformer_replace_tokens is None:
                    audio_tokens = state.graphed_depth(text_token, transformer_out)
                else:
                    assert depformer_replace_tokens.dim() == 3
                    audio_tokens = depformer_replace_tokens.squeeze(-1)
                if self_gen.on_audio_hook is not None:
                    self_gen.on_audio_hook(audio_tokens)

            state.offsets = torch.where(state.exec_mask, state.offsets + 1, state.offsets)
            state.offset_cpu += 1
            positions = (state.offsets % CT)[:, None, None]
            scatter_with_mask_(state.cache[:, :1], -1, positions, text_token[:, None, None], state.exec_mask[:, None, None])
            if audio_tokens is not None:
                audio_tokens = audio_tokens[:, :, None]
                scatter_with_mask_(state.cache[:, 1: lm_model_local.dep_q + 1, :], -1, positions.expand_as(audio_tokens), audio_tokens, state.exec_mask[:, None, None])

            if not self_gen.support_out_of_sync and state.offset_cpu <= self_gen.max_delay:
                return None
            gen_delays_cuda = self_gen.delays_cuda[: lm_model_local.dep_q + 1]
            index = (state.offsets[:, None, None] - self_gen.max_delay + gen_delays_cuda[:, None]) % CT
            out = state.cache.gather(dim=2, index=index)
            mask = (state.offsets <= self_gen.max_delay) | ~state.exec_mask
            out[mask, :, :] = lm_model_local.ungenerated_token_id
            return out, transformer_out, layer_hidden

        lm_model.forward_text = types.MethodType(forward_text_with_layer, lm_model)
        lm_gen._step = types.MethodType(step_with_layer, lm_gen)
        # step_with_layer takes only input_tokens, so text cannot be forced on
        # this fallback path; search-window suppression degrades to a no-op.
        self._step_supports_text_token = False
        lm_gen.streaming_forever(1)
        self._warmup_runtime()
        print(f"[liveTryPlasticity] installed graphed layer capture layer={self.tf_capture_layer}", flush=True)

    # -- Turn-detection constants (frame = one MIMI_FRAME_SIZE / 80ms chunk,
    # same granularity as _step() itself) --
    _VAD_SILENCE_FRAMES_REQUIRED = 12   # ~960ms of silence ends an utterance
    # Class-level fallback only -- __init__ always overrides this with an
    # instance attribute derived from the --search_max_filler_sec CLI flag (see
    # __init__ below). Kept here so the attribute still exists with a sane
    # value for any code path that might reference it before __init__ runs.
    _SEARCH_MAX_FILLER_FRAMES = 25          # ~2s filler cap before a fallback <ref>
    # RMS level at which the model's own output counts as "it has started
    # speaking". Mirrors the default --assistant_speech_rms_threshold used by
    # the avatar gate. Used for the time-to-first-word metric.
    _SPEECH_RMS_THRESHOLD = 0.006

    # Conservative default: _install_graph_hidden_capture() sets the real value
    # for whichever step override it installs. Only the PersonaPlex graphed
    # path forwards a text_token, so search-window text suppression is a no-op
    # on the fallback path rather than a TypeError.
    _step_supports_text_token = False

    def reset_session(self) -> None:
        # A session that ends (or is restarted by a reconnect) with a turn still
        # in flight would otherwise never have its latency block written -- the
        # block is normally emitted when the NEXT question arrives, and for the
        # last turn of a conversation there is no next question. Flush it here,
        # while the record and the response bounds are still intact. Guarded
        # with getattr: reset_session also runs during __init__ (via
        # _warmup_runtime), before any of this state exists.
        if getattr(self, "_latency_turn_id", None) is not None:
            try:
                import search_helpers

                tail = search_helpers.strip_injected_tags(
                    self.audio_text[self._turn_start_audio_text_len:]
                )
            except Exception:
                tail = ""
            self._finish_turn_latency(
                tail, outcome="session ended before the user asked anything else"
            )

        super().reset_session()
        # Per-conversation STT/turn-detection state. No-ops harmlessly when
        # STT isn't configured (self.stt_lm_gen stays None).
        self.stt_token_buffer: list = []
        self.stt_in_utterance = False
        self.stt_silence_frame_count = 0
        self.stt_last_vad_end = False
        self.search_turn_epoch = 0
        self.search_ref_committed_this_turn = False
        self.search_awaiting_ref = False
        # Set True if an uncaught exception ever escapes the per-chunk STT/
        # search hook in _step() (see try/except around _stt_step /
        # _consume_pending below). Before that guard existed, such an
        # exception propagated out of _step() and killed the whole GPU
        # producer thread silently -- confirmed by conversation_logs_4, whose
        # log ends mid-turn with no error line and no further component_status
        # entries (the thread died with nothing left to log to). Once set,
        # search is skipped for the rest of THIS session but the underlying
        # avatar conversation keeps working.
        self.search_hard_disabled = False
        # Per-turn stage-timing accumulator, see _start_turn / _log_timing_summary.
        self._turn_timing_start: float | None = None
        self._turn_timing_stages: dict[str, float] = {}
        # -- Dedicated end-to-end latency log (latency_logger.py) ------------
        # `_turn_speech_end_perf` is the perf_counter reading of the instant the
        # user's utterance ENDED. It is t=0 for every number in
        # latency_<session>.log, and is deliberately taken in _stt_step (where
        # the VAD actually fires) rather than in _start_turn, so the
        # speech-to-text decode that runs in between is measured as a pipeline
        # cost instead of vanishing before the clock starts.
        self._turn_speech_end_perf: float | None = None
        # Stage timings/counts measured in _stt_step, before the turn record exists.
        self._pending_stt_stages: dict[str, float] = {}
        self._pending_stt_counts: dict[str, int] = {}
        # Which turn the per-chunk generation costs below are being charged to,
        # and when the model last emitted a text token. The latter is the only
        # honest "the answer finished" instant available for a full-duplex
        # model: it is frozen into `_answer_end_perf` the moment the user starts
        # their NEXT utterance, which is the same boundary the logged response
        # text is sliced at.
        self._latency_turn_id: int | None = None
        # Logging-only: which turn_id each "fire once per turn" mark name has
        # already been recorded for, so the many per-chunk call sites in the
        # audio/avatar send pipeline (queue enqueue/release, pacing sleep,
        # avatar gate transition) don't repeatedly overwrite the same mark
        # with a later timestamp. See _mark_first_per_turn().
        self._marks_once: dict[str, int] = {}
        self._turn_model_first_audio_marked = False
        self._last_text_emit_perf: float | None = None
        self._answer_end_perf: float | None = None
        self._turn_text_token_count = 0
        # Cross-thread handoff slots. The background thread WRITES these; the
        # GPU thread POLLS them once per 80ms chunk in _consume_pending() and
        # is the only thread allowed to touch self.lm_gen. `pending_lookup` is
        # separate from `pending_ref` because the <lookup> filler is now
        # injected only AFTER the router has decided a search is happening --
        # telling the model to "please wait" and then not searching would
        # leave it stalling for an answer that never arrives.
        self.pending_lookup_tokens: list | None = None
        self.pending_ref_tokens: list | None = None
        self.pending_search_cancelled = False
        # Set by the background thread the moment it is about to hit the search
        # API; the GPU thread turns it into an actual _start_thinking_sound()
        # call on its next chunk. Needed because the model-router path only
        # learns that a search is happening after its forward pass, off-thread.
        self.pending_start_thinking = False
        # While True, _step() forces the model's text stream to its own
        # zero_text_code so it composes NOTHING during a search.
        #
        # Muting the outgoing audio (the thinking sound) was never enough: the
        # model keeps generating text behind the filler, so by the time the
        # <ref> arrives 3-5s later it is already mid-sentence with an invented
        # figure and simply finishes it. conversation_log_2 turn 4 is the
        # proof -- <ref> carried "$4,085 per ounce", the model said
        # "$1,408.50", and its first audio came 0.03s after the question,
        # about five seconds BEFORE the reference existed.
        self.suppress_text_until_ref = False
        self._pending_ref_token_counts = (0, 0)
        self.search_filler_frame_count = 0
        self.search_session_history: list[tuple[str, str]] = []
        self.search_current_transcript = ""
        # Snapshot of len(self.audio_text) at turn start (when the user STOPPED
        # speaking), marking where this turn's assistant response begins.
        self._turn_start_audio_text_len = 0
        # Snapshot of len(self.audio_text) at the moment the user STARTS
        # speaking the next utterance, marking where this turn's response
        # ends. Both bounds are needed: the model is full-duplex and keeps
        # emitting text while the user talks, so slicing to the end of
        # audio_text (as this used to) pulled the NEXT turn's opening words
        # into the previous turn's logged response. Confirmed in
        # conversation_logs_5, where the gold-price turn's logged reply ended
        # with "I' not sure how to test a Tesla" -- the start of the reply to
        # the Tesla question the user asked next.
        self._utterance_start_audio_text_len = 0
        # True time-to-first-spoken-word tracking. This is the ONLY honest
        # latency metric for a full-duplex model: `turn_replied` cannot serve
        # as one, because it fires when the NEXT utterance ends, so its
        # timestamp measures how long the user waited before speaking again.
        # Confirmed in conversation_log_1, where every REPLIED line shares its
        # millisecond timestamp with the following HEARD line -- the 19s/39s
        # "latencies" read off that log were the user's own pauses, not the
        # assistant's delay.
        self._turn_awaiting_first_speech = False
        self._turn_first_speech_epoch = 0
        self.search_thinking_active = False
        self._thinking_sound_cursor = 0
        self._thinking_sound_started_at = 0.0
        self._thinking_sound_play_count = 0

    def _inject_tokens(self, tokens: list[int]) -> None:
        """Force-feed text tokens via the PUBLIC lm_gen.step() (not the
        hidden-capturing `_step` installed above) -- ports server.py's
        <ref>/<lookup> mid-stream injection technique unchanged.
        reset_streaming() is never called: the shared KV-cache (and the
        conversation heard so far) stays intact across the injection."""
        for tok in tokens:
            self.lm_gen.step(
                moshi_tokens=self.lm_gen._encode_zero_frame(),
                text_token=tok,
                input_tokens=self.lm_gen._encode_sine_frame(),
            )

    def _stt_step(self, chunk: torch.Tensor) -> None:
        """Run the separate STT/VAD submodel one 80ms frame forward (same GPU
        thread as everything else in _step()); on a detected end-of-utterance,
        kick off a turn. Never touches self.lm_gen/self.mimi."""
        stt_codes = self.stt_mimi.encode(chunk)
        stt_result = self.stt_lm_gen.step_with_extra_heads(stt_codes)
        if stt_result is None:
            return
        stt_tokens, vad_heads = stt_result
        vad_score = 0.0
        if vad_heads and len(vad_heads) > 2:
            vad_score = float(vad_heads[2][0, 0, 0].cpu().item())
        if stt_tokens is not None:
            self.stt_token_buffer.append(stt_tokens[:, :1, :].cpu())
            text_token = stt_tokens[0, 0, 0].item()
            if text_token not in (0, self.stt_padding_token_id):
                if not self.stt_in_utterance:
                    # First real word of a new user utterance: everything the
                    # model emitted before this point belongs to the PREVIOUS
                    # turn's response, everything after is it reacting to what
                    # it is hearing now.
                    self._utterance_start_audio_text_len = len(self.audio_text)
                    # Same boundary, in time rather than in characters: the last
                    # text token emitted before this word is the end of the
                    # previous turn's answer. Frozen here because
                    # _last_text_emit_perf keeps moving afterwards.
                    self._answer_end_perf = self._last_text_emit_perf
                self.stt_in_utterance = True
                self.stt_last_vad_end = False
        if vad_score > self.vad_threshold:
            self.stt_silence_frame_count += 1
        else:
            self.stt_silence_frame_count = 0
        vad_fired = (
            self.stt_silence_frame_count >= self._VAD_SILENCE_FRAMES_REQUIRED
            and not self.stt_last_vad_end
        )
        self.stt_last_vad_end = self.stt_silence_frame_count >= self._VAD_SILENCE_FRAMES_REQUIRED
        if vad_fired and self.stt_in_utterance and self.stt_token_buffer and not self.search_awaiting_ref:
            import search_helpers

            # t=0 for the whole latency log: the question is complete here, and
            # everything below (STT decode, sanity gate, routing, search,
            # compression, injection, generation) is time the user spends
            # waiting.
            t_speech_end = time.perf_counter()
            n_stt_frames = len(self.stt_token_buffer)
            t_stt0 = time.perf_counter()
            transcript, transcript_token_ids = search_helpers.decode_stt_tokens_with_ids(
                self.stt_token_buffer, self.stt_tokenizer, self.stt_padding_token_id
            )
            stt_decode_elapsed = time.perf_counter() - t_stt0
            self.stt_token_buffer = []
            self.stt_in_utterance = False
            self.stt_silence_frame_count = 0
            if transcript.strip():
                # Log the PREVIOUS turn's assistant response now that we know
                # it's finished (the user has started speaking again).
                # Slice between BOTH turn boundaries (see reset_session):
                # from where the previous turn's answer began, to where this
                # new utterance's first word arrived. Falls back to "rest of
                # audio_text" only if the end bound was never set or looks
                # stale, so a missing bound degrades to the old behavior
                # rather than logging an empty response.
                resp_end = self._utterance_start_audio_text_len
                if resp_end <= self._turn_start_audio_text_len:
                    resp_end = len(self.audio_text)
                prev_response = search_helpers.strip_injected_tags(
                    self.audio_text[self._turn_start_audio_text_len:resp_end]
                )
                if prev_response:
                    self.conv_logger.turn_replied(self.search_turn_epoch, prev_response)
                    self.conv_logger.assistant_response(self.search_current_transcript, prev_response)
                    self.conv_logger.narrate_response(
                        self.search_turn_epoch, self.search_current_transcript, prev_response
                    )
                    self._finish_turn_latency(prev_response)
                    if self.search_current_transcript:
                        self.search_session_history.append((self.search_current_transcript, prev_response))
                        self.search_session_history = self.search_session_history[-6:]
                elif self.search_current_transcript:
                    # The model produced zero spoken text for the previous
                    # turn before the user spoke again. Previously this left
                    # no trace anywhere in either log -- confirmed in
                    # conversation_logs_6, where "Can you hear me?" (turn 1)
                    # got no logged response at all and the only way to find
                    # that was to notice an assistant_response line was
                    # missing. Flagged explicitly now so it's never silent.
                    turn_start_perf = self._turn_timing_start
                    gap_s = max(0.0, time.perf_counter() - turn_start_perf) if turn_start_perf else 0.0
                    self.conv_logger.no_response_warning(
                        self.search_turn_epoch, self.search_current_transcript, gap_s
                    )
                    self.conv_logger.narrate_no_response_warning(
                        self.search_turn_epoch, self.search_current_transcript, gap_s
                    )
                    # Still closed out in the latency log: a turn that produced
                    # nothing is exactly the case where the timings matter most,
                    # and leaving it open would silently drop every stage that
                    # WAS measured for it.
                    self._finish_turn_latency("", outcome="no spoken response was produced")
                # -- Transcript sanity gate --------------------------------
                # Two independent checks, both cheap enough for _step()'s 80ms
                # budget:
                #   1. writing system -- the STT submodel cannot emit Devanagari
                #      or Cyrillic, so those are decode garbage;
                #   2. language -- the model IS bilingual (en/fr) and on
                #      unclear audio hallucinates fluent Spanish/French in
                #      ordinary Latin letters, which check (1) cannot see.
                # Routing on either would answer a question nobody asked and
                # can send nonsense to a paid search API, so the turn is
                # dropped here -- before the router, before any injection. The
                # avatar is unaffected: PersonaPlex heard the real audio
                # directly and keeps replying normally.
                t_check0 = time.perf_counter()
                usable, script_stats = search_helpers.check_transcript_usable(
                    transcript,
                    self.stt_max_non_latin_ratio,
                    require_english=self.stt_require_english,
                )
                transcript_check_elapsed = time.perf_counter() - t_check0
                if not usable and self.stt_reject_foreign_script:
                    self.conv_logger.turn_transcript_rejected(
                        self.search_turn_epoch + 1, transcript, script_stats, transcript_token_ids
                    )
                    hint = ""
                    if script_stats.get("kind") == "script":
                        hint = (
                            f"\n                        first ids={transcript_token_ids[:24]} -- if "
                            f"these look like ordinary ids, the STT tokenizer is likely mismatched; "
                            f"compare the 'stt tokenizer=' line printed at startup."
                        )
                    print(
                        f"[liveTryPlasticity][STT] rejected transcript "
                        f"({script_stats.get('reason', 'unusable')}, "
                        f"{len(transcript_token_ids)} tok): {transcript[:120]!r}{hint}",
                        flush=True,
                    )
                    with contextlib.suppress(Exception):
                        self.stt_lm_gen.reset_streaming()
                        self.stt_mimi.reset_streaming()
                    return

                self.conv_logger.turn_heard(
                    self.search_turn_epoch + 1, transcript, transcript_token_ids, script_stats
                )
                self.conv_logger.narrate_user_message(self.search_turn_epoch + 1, transcript)
                # Handed to _start_turn through attributes rather than
                # arguments so the (unchanged) signature keeps working for the
                # other call sites and subclasses.
                self._turn_speech_end_perf = t_speech_end
                self._pending_stt_stages = {
                    "stt_decode": stt_decode_elapsed,
                    "transcript_check": transcript_check_elapsed,
                }
                self._pending_stt_counts = {
                    "stt_frames_decoded": n_stt_frames,
                    "transcript_tokens": len(transcript_token_ids),
                    "transcript_chars": len(transcript),
                }
                with contextlib.suppress(Exception):
                    self.stt_lm_gen.reset_streaming()
                    self.stt_mimi.reset_streaming()
                self._start_turn(transcript)

    def _begin_casual_turn(self, transcript: str, reason: str) -> None:
        """Mark a turn that will be answered from the model's own knowledge:
        no search, no injection, nothing added to the context. Only turn
        bookkeeping and (optionally) the acknowledgement sound."""
        self.search_turn_epoch += 1
        self.search_current_transcript = transcript
        self._turn_start_audio_text_len = len(self.audio_text)
        self._turn_awaiting_first_speech = True
        self._turn_first_speech_epoch = self.search_turn_epoch
        print(
            f"[liveTryPlasticity][search] no search ({reason}) -- "
            f"answering from the model's own knowledge",
            flush=True,
        )
        # No thinking sound here, by design: nothing is being waited on. The
        # model answers from its own knowledge and starts speaking within a few
        # hundred milliseconds, so a filler clip would only delay and mask a
        # reply that never needed covering.

    def _start_turn(self, transcript: str) -> None:
        """Decide how to answer this utterance.

        Tier 0 (here, on the GPU thread): pure-regex routing, microseconds. A
        confident rule verdict either starts the search immediately or ends the
        turn as a casual one, with no model call at all.

        Tier 1 (background thread): anything the rules could not resolve is
        scored by the Qwen router in `_route_and_search`. That call is tens of
        milliseconds -- far too slow for _step()'s 80ms budget, which is why it
        never runs here.

        Note the ordering change vs. the old document-retrieval path: <lookup>
        is NOT injected up front any more. It is injected only once a search is
        actually committed, because telling the model to "please wait" and then
        deciding not to search would leave it stalling for an answer that never
        arrives."""
        import search_helpers

        # Per-turn stage-timing accumulator (see timing_summary at the end of
        # _consume_pending). Safe as a plain instance attribute:
        # search_awaiting_ref gates re-entry, so only one turn's timing is ever
        # in flight at once, same as pending_ref_tokens below.
        self._turn_timing_start = time.perf_counter()
        self._turn_timing_stages = {}

        # Open this turn's record in the dedicated latency log. Done before the
        # search_enabled early-return so that a plain conversational turn (no
        # routing, no search) is still timed end to end -- that is the baseline
        # every searched turn has to be compared against.
        self._open_turn_latency(self.search_turn_epoch + 1, transcript)
        self.conv_logger.latency.mark(self._latency_turn_id, "routing_start")

        if not getattr(self, "search_enabled", False):
            self._begin_casual_turn(transcript, "search not configured")
            self.conv_logger.latency.mark(
                self._latency_turn_id, "decision_made", "search not configured"
            )
            self.conv_logger.latency.mark(
                self._latency_turn_id, "routing_end", "search not configured"
            )
            return

        t_rules0 = time.perf_counter()
        ruled, rule_reason = search_helpers.rule_route_explain(transcript)
        rules_elapsed = time.perf_counter() - t_rules0
        self._turn_timing_stages["rule_route"] = rules_elapsed
        self.conv_logger.quick_gate_timing(rules_elapsed)
        self.conv_logger.latency.stage(
            self._latency_turn_id, "rule_route", rules_elapsed,
            note=("resolved by rules" if ruled is not None else "undecided, router will run"),
        )

        if ruled is False:
            self.conv_logger.turn_decision(
                self.search_turn_epoch + 1, transcript, needs_search=False, source="rules",
                score=0.0, elapsed_s=rules_elapsed,
                reason=rule_reason,
            )
            self.conv_logger.narrate_router_decision(
                self.search_turn_epoch + 1, transcript, False, "rules", 0.0, rule_reason,
            )
            self._begin_casual_turn(transcript, "rule: static phrase")
            self.conv_logger.turn_done(
                self.search_turn_epoch, "answered from own knowledge",
                time.perf_counter() - self._turn_timing_start,
            )
            self.conv_logger.latency.mark(
                self._latency_turn_id, "decision_made", "rules: no search needed"
            )
            self.conv_logger.latency.mark(
                self._latency_turn_id, "routing_end", "rules: no search needed"
            )
            self.conv_logger.latency.count(self._latency_turn_id, decided_by="rules", searched=False)
            return

        # Either the rules demanded a search (ruled is True) or they could not
        # decide (None) and the router will settle it on the background thread.
        self.search_turn_epoch += 1
        my_epoch = self.search_turn_epoch
        self.search_ref_committed_this_turn = False
        self.search_awaiting_ref = True
        self.search_filler_frame_count = 0
        self.search_current_transcript = transcript
        self.pending_lookup_tokens = None
        self.pending_ref_tokens = None
        self.pending_search_cancelled = False
        self._turn_start_audio_text_len = len(self.audio_text)
        self._turn_awaiting_first_speech = True
        self._turn_first_speech_epoch = my_epoch

        # The thinking sound is NOT started here for the undecided case. Only
        # the rules path knows synchronously that a search is happening; when
        # the model router has to decide, that verdict arrives on the
        # background thread and may well be "no search", in which case no sound
        # should ever have played. _route_and_search raises
        # pending_start_thinking once it is genuinely about to call the search
        # API, and _consume_pending starts the clip on the GPU thread.
        if ruled is True:
            if self._will_actually_search():
                self._start_thinking_sound(my_epoch, transcript)
            self.conv_logger.turn_decision(
                my_epoch, transcript, needs_search=True, source="rules", score=1.0,
                elapsed_s=rules_elapsed,
                reason=rule_reason,
            )
            self.conv_logger.narrate_router_decision(
                my_epoch, transcript, True, "rules", 1.0, rule_reason,
            )
            self.conv_logger.latency.mark(
                self._latency_turn_id, "decision_made", "rules: search online"
            )
            self.conv_logger.latency.mark(
                self._latency_turn_id, "routing_end", "rules: search online"
            )
            self.conv_logger.latency.count(self._latency_turn_id, decided_by="rules", searched=True)

        threading.Thread(
            target=self._route_and_search,
            args=(transcript, my_epoch, ruled is True),
            daemon=True,
            name="query-search",
        ).start()

    def _route_and_search(self, transcript: str, my_epoch: int, rules_said_search: bool) -> None:
        """Background thread: (optionally) run the Qwen router, then — only if
        a search is warranted — web search, clean, and compress into a short
        grounding statement. Only touches the router/compressor objects and
        plain attributes; never self.lm_gen/self.mimi/self.stt_lm_gen. Results
        are handed to the GPU thread through self.pending_* slots, which
        _consume_pending() polls once per chunk."""
        import search_helpers

        try:
            # --- Tier 1: the model-based decision, unless the rules already
            # committed to searching (in which case skip the forward pass). ---
            if not rules_said_search:
                verdict = self.query_router.decide(transcript)
                self._turn_timing_stages["router"] = float(verdict.get("elapsed_s", 0.0))
                self.conv_logger.latency.stage(
                    my_epoch, "router_model", float(verdict.get("elapsed_s", 0.0)),
                    note=f"{verdict['source']}: score {float(verdict['score']):.3f}",
                )
                self.conv_logger.latency.mark(
                    my_epoch, "decision_made",
                    "search online" if verdict["needs_search"] else "answer directly",
                )
                self.conv_logger.latency.mark(
                    my_epoch, "routing_end",
                    f"router: {verdict['source']}: "
                    + ("search online" if verdict["needs_search"] else "answer directly"),
                )
                self.conv_logger.latency.count(
                    my_epoch,
                    decided_by=str(verdict["source"]),
                    searched=bool(verdict["needs_search"]),
                    router_score=round(float(verdict["score"]), 4),
                    router_prompt_tokens=int(getattr(self.query_router, "last_prompt_tokens", 0)),
                )
                self.conv_logger.turn_decision(
                    my_epoch, transcript,
                    needs_search=bool(verdict["needs_search"]),
                    source=str(verdict["source"]),
                    score=float(verdict["score"]),
                    elapsed_s=float(verdict["elapsed_s"]),
                    reason=str(verdict["reason"]),
                )
                self.conv_logger.narrate_router_decision(
                    my_epoch, transcript, bool(verdict["needs_search"]),
                    str(verdict["source"]), float(verdict["score"]), str(verdict["reason"]),
                )
                print(
                    f"[liveTryPlasticity][router] needs_search={verdict['needs_search']} "
                    f"src={verdict['source']} score={verdict['score']:.3f} "
                    f"in {1000.0 * float(verdict['elapsed_s']):.0f}ms :: {transcript!r}",
                    flush=True,
                )
                if not verdict["needs_search"]:
                    # Nothing will be injected. Tell the GPU thread to drop the
                    # thinking sound and let the model answer on its own. The
                    # <lookup> filler was deliberately never injected, so the
                    # context is untouched and the reply is a normal one.
                    self.conv_logger.turn_done(
                        my_epoch, "answered from own knowledge",
                        time.perf_counter() - (self._turn_timing_start or time.perf_counter()),
                    )
                    self._log_timing_summary()
                    if my_epoch == self.search_turn_epoch:
                        self.pending_search_cancelled = True
                    return

            if my_epoch != self.search_turn_epoch or self.search_ref_committed_this_turn:
                return

            # --- A search is happening: only now does the model get told to
            # wait. Injection itself must happen on the GPU thread. ---
            lookup_text = search_helpers.wrap_with_lookup_tags()
            self.pending_lookup_tokens = self.tokenizer.encode(lookup_text)

            hits: list[dict] = []
            if not self.web_search_enabled:
                print(
                    "[liveTryPlasticity][search] a search was needed but web search is "
                    "disabled -- falling back to the model's own knowledge",
                    flush=True,
                )
            elif not self.web_search_api_key:
                print(
                    "[liveTryPlasticity][search] a search was needed but no API key is "
                    "configured -- falling back to the model's own knowledge",
                    flush=True,
                )
            else:
                # An online search is genuinely about to happen -- this is the
                # ONLY place (besides the synchronous rules path in
                # _start_turn) that the thinking sound may begin. Raised here
                # rather than at the routing decision so that a "search"
                # verdict which cannot actually reach the network never makes
                # a sound.
                self.pending_start_thinking = True
                # Logged at dispatch time (not just on completion below) so
                # conversation_<session>.log's own event order matches real
                # chronology -- previously this only appeared after the call
                # finished, which could make a fast-but-later fallback injection
                # look like it preceded a web search that actually started earlier.
                self.conv_logger.event(
                    "web_search_start", query=transcript, provider=self.web_search_provider,
                    triggered_reason="the router decided this needs current information",
                )
                self.conv_logger.narrate_web_search_start(
                    my_epoch, transcript, self.web_search_provider,
                    "the question needs current information",
                )
                t_web0 = time.perf_counter()
                web_hits = search_helpers.web_search_query_sync(
                    transcript, self.web_search_api_key, self.web_search_provider,
                    self.web_search_max_results, self.web_search_timeout,
                )
                web_elapsed = time.perf_counter() - t_web0
                self._turn_timing_stages["web_search"] = web_elapsed
                self.conv_logger.latency.stage(
                    my_epoch, "web_search", web_elapsed,
                    note=f"{self.web_search_provider}: {len(web_hits)} result(s)",
                )
                self.conv_logger.latency.mark(my_epoch, "search_done")
                self.conv_logger.web_search(
                    transcript, self.web_search_provider, len(web_hits), web_elapsed,
                    triggered_reason="router decided live data was required",
                )
                # Forensic finding (conversation_logs_2/3): web results carried
                # no relevance floor at all -- scores as low as 0.04 (clearly
                # unrelated pages) were still summarized/injected as if they
                # were usable context. Search engines always return something,
                # so this floor is the only thing standing between an unrelated
                # page and the assistant's spoken answer.
                t_filter0 = time.perf_counter()
                relevant = [h for h in web_hits if h.get("similarity_score", 0.0) >= self.web_search_min_score]
                sorted_hits = sorted(relevant, key=lambda c: c["similarity_score"], reverse=True)
                hits = sorted_hits[: self.web_search_max_results]
                filter_elapsed = time.perf_counter() - t_filter0
                self._turn_timing_stages["search_filter"] = filter_elapsed
                self.conv_logger.latency.stage(
                    my_epoch, "search_filter", filter_elapsed,
                    note=f"{len(web_hits)} found -> {len(hits)} kept",
                )
                self.conv_logger.latency.count(
                    my_epoch,
                    web_results_found=len(web_hits),
                    web_results_kept=len(hits),
                    web_result_chars=sum(len(str(h.get("text", ""))) for h in hits),
                )
                self.conv_logger.turn_search(
                    my_epoch, self.web_search_provider, len(web_hits), len(hits), web_elapsed
                )
                self.conv_logger.retrieval(transcript, "web", hits, web_elapsed)
                self.conv_logger.narrate_web_search_results(
                    my_epoch, sorted_hits, hits, self.web_search_max_results
                )
                if web_hits and not relevant:
                    print(
                        f"[liveTryPlasticity][search] all {len(web_hits)} web result(s) scored below "
                        f"web_search_min_score={self.web_search_min_score} -- discarding them and "
                        f"falling back to the model's own knowledge",
                        flush=True,
                    )

            if my_epoch != self.search_turn_epoch or self.search_ref_committed_this_turn:
                # Superseded by a newer turn, or the filler timeout already
                # committed a fallback while we were searching -- skip
                # compression entirely rather than spending GPU time on a
                # result that can never be used (previously this ran
                # unconditionally and was observed discarding a fully-computed
                # answer in every logged search turn).
                return

            grounding = ""
            used_fallback = False
            t_compress0 = time.perf_counter()
            if hits and self.context_compressor is not None:
                grounding = self.context_compressor.compress(question=transcript, chunks=hits)
                primary_elapsed = time.perf_counter() - t_compress0
                self._turn_timing_stages["compression"] = (
                    self._turn_timing_stages.get("compression", 0.0) + primary_elapsed
                )
                self.conv_logger.compressor_call(
                    transcript, [h.get("text", "") for h in hits[:2]], grounding,
                    primary_elapsed, used_fallback=False,
                )
                # How many tokens the compressor read and wrote. `output_tokens`
                # is the count for the sentence that actually reaches the
                # model; `generated_tokens` includes what the cleanup stripped,
                # and `hit_token_cap` says the answer was cut off by
                # max_new_tokens rather than ending on its own.
                comp_stats = dict(getattr(self.context_compressor, "last_stats", {}) or {})
                self.conv_logger.latency.stage(
                    my_epoch, "compression", primary_elapsed,
                    note=(
                        f"{comp_stats.get('compressor_output_tokens', 0)} token(s) out"
                        + (" [REJECTED]" if comp_stats.get("compressor_rejected") else "")
                    ),
                )
                if comp_stats:
                    self.conv_logger.latency.count(my_epoch, **comp_stats)
                if grounding:
                    self.conv_logger.narrate_summary(my_epoch, "a web search", len(hits), grounding, used_fallback=False)
            if not grounding and hits:
                used_fallback = True
                t_fallback0 = time.perf_counter()
                grounding = search_helpers.summarize_web_fallback(
                    transcript, hits, max_sentences=2, max_chars=200
                )
                fallback_elapsed = time.perf_counter() - t_fallback0
                self._turn_timing_stages["compression"] = (
                    self._turn_timing_stages.get("compression", 0.0) + fallback_elapsed
                )
                self.conv_logger.compressor_call(
                    transcript, [h.get("text", "") for h in hits[:2]], grounding,
                    fallback_elapsed, used_fallback=True,
                )
                self.conv_logger.latency.stage(
                    my_epoch, "compression_fallback", fallback_elapsed,
                    note=f"{len(grounding)} char(s) out",
                )
                if grounding:
                    self.conv_logger.narrate_summary(my_epoch, "a web search", len(hits), grounding, used_fallback=True)
            if not grounding:
                self.conv_logger.narrate_no_information(my_epoch)
            self.conv_logger.turn_ground(my_epoch, grounding, used_fallback=used_fallback)
            self.conv_logger.latency.mark(
                my_epoch, "grounding_ready",
                ("extractive fallback" if used_fallback else "compressed by the LLM")
                if grounding else "nothing usable was produced",
            )

            if my_epoch != self.search_turn_epoch or self.search_ref_committed_this_turn:
                return

            ref_content = grounding.strip() if grounding else (
                "There's no specific information available on this, so answer from general knowledge."
            )
            t_encode0 = time.perf_counter()
            ids_before_trim = self.tokenizer.encode(ref_content)
            ids = ids_before_trim
            if len(ids) > self.max_ref_tokens:
                ids = ids[: self.max_ref_tokens]
                ref_content = self.tokenizer.decode(ids)
            new_ref_tokens = self.tokenizer.encode(search_helpers.wrap_with_ref_tags(ref_content))
            ref_encode_elapsed = time.perf_counter() - t_encode0
            self._turn_timing_stages["ref_encode"] = ref_encode_elapsed
            self.conv_logger.latency.stage(
                my_epoch, "ref_encode", ref_encode_elapsed,
                note=f"{len(new_ref_tokens)} token(s) ready to inject",
            )
            # Token counts as the SPEAKING model sees them (Moshi's tokenizer),
            # which is a different vocabulary from the compressor's counts
            # above -- both are logged because only this one is bounded by
            # max_ref_tokens and can silently truncate the answer.
            self.conv_logger.latency.count(
                my_epoch,
                grounding_tokens_before_trim=len(ids_before_trim),
                grounding_tokens_injected=len(new_ref_tokens),
                grounding_tokens_trimmed=max(0, len(ids_before_trim) - len(ids)),
                max_ref_tokens=int(self.max_ref_tokens),
            )
            # Set the metadata BEFORE the token list itself: the GPU thread's
            # _consume_pending polls `pending_ref_tokens is not None` as its
            # readiness signal, so the counts must already be in place by the
            # time that becomes true (avoids a narrow race where the GPU thread
            # reads stale/default counts).
            self._pending_ref_token_counts = (len(ids_before_trim), self.max_ref_tokens)
            self.pending_ref_tokens = new_ref_tokens
            print(
                f"[liveTryPlasticity][search] prepared <ref> block "
                f"({len(self.pending_ref_tokens)} tok): {ref_content[:150]!r}",
                flush=True,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[liveTryPlasticity][search] route/search/compress failed: {e!r}\n{tb}", flush=True)
            self.conv_logger.error("route_and_search", e, tb)
            self.conv_logger.narrate_no_information(my_epoch)
            self.conv_logger.latency.mark(my_epoch, "grounding_ready", f"failed: {e!r}")
            if my_epoch == self.search_turn_epoch:
                fallback_tokens = self.tokenizer.encode(search_helpers.wrap_with_ref_tags(
                    "There's no specific information available on this, so answer from general knowledge."
                ))
                self._pending_ref_token_counts = (len(fallback_tokens), self.max_ref_tokens)
                self.pending_ref_tokens = fallback_tokens

    def _open_turn_latency(self, turn_id: int, transcript: str) -> None:
        """Open this turn's record in latency_<session>.log and seed it with
        the stages already measured in _stt_step. Never raises."""
        try:
            self._latency_turn_id = turn_id
            self._turn_text_token_count = 0
            # Reset the per-turn once-only mark bookkeeping used by _step()
            # and the GPU producer thread (model_first_audio, audio pipeline
            # marks, avatar gate marks). Logging-only state.
            self._turn_model_first_audio_marked = False
            t0_for_turn = (
                self._turn_speech_end_perf if self._turn_speech_end_perf is not None else time.perf_counter()
            )
            self.conv_logger.latency.start_turn(
                turn_id,
                t0=t0_for_turn,
                transcript=transcript,
            )
            # Explicit t=0 timeline row: start_turn's t0 IS this instant, but
            # without an actual mark() call here it never appears as a row in
            # the timeline output -- it was previously only implicit.
            self.conv_logger.latency.mark(
                turn_id, "speech_end", "user stopped speaking (t=0)", at=t0_for_turn,
            )
            for name, secs in (self._pending_stt_stages or {}).items():
                self.conv_logger.latency.stage(turn_id, name, secs)
            if self._pending_stt_counts:
                self.conv_logger.latency.count(turn_id, **self._pending_stt_counts)
            self._pending_stt_stages = {}
            self._pending_stt_counts = {}
        except Exception:
            pass

    def _finish_turn_latency(self, response: str, outcome: str = "") -> None:
        """Close the turn currently being timed, once its spoken answer is known
        to be complete (the user has started speaking again). Never raises."""
        try:
            turn_id = self._latency_turn_id
            if turn_id is None:
                return
            # `_answer_end_perf` is the frozen bound set when the user started
            # speaking again. On a session teardown it was never set, so fall
            # back to the last text token emitted -- but only if that token
            # belongs to THIS turn, otherwise a turn that produced no text at
            # all would inherit the previous turn's end time.
            end_perf = self._answer_end_perf
            if end_perf is None:
                rec = self.conv_logger.latency.get(turn_id)
                if rec is not None and self._last_text_emit_perf is not None:
                    if self._last_text_emit_perf >= rec.t0:
                        end_perf = self._last_text_emit_perf
            if end_perf is not None:
                self.conv_logger.latency.mark(
                    turn_id, "answer_complete", "last text token of the reply",
                    at=end_perf,
                )
            self.conv_logger.latency.count(
                turn_id,
                answer_chars=len(response),
                answer_words=len(response.split()),
                answer_text_tokens=self._turn_text_token_count,
            )
            self.conv_logger.latency.finish_turn(turn_id, response=response, outcome=outcome)
            self._latency_turn_id = None
            self._answer_end_perf = None
        except Exception:
            pass

    def _mark_first_per_turn(
        self, name: str, turn_id: int | None, note: str = "", at: float | None = None,
    ) -> None:
        """Fire latency.mark(turn_id, name, ...) at most once per turn_id.
        Logging-only convenience for the per-chunk call sites in the audio /
        avatar send pipeline (queue enqueue/release, pacing sleep, avatar
        gate transition) where marking on every chunk would just overwrite
        the same dict entry with a later timestamp -- keeping only the FIRST
        occurrence is what makes these marks meaningful. Never raises."""
        try:
            if turn_id is None:
                return
            if self._marks_once.get(name) == turn_id:
                return
            self._marks_once[name] = turn_id
            self.conv_logger.latency.mark(turn_id, name, note, at=at)
        except Exception:
            pass

    def _log_timing_summary(self) -> None:
        """Print the consolidated big-to-small timing breakdown for the turn
        that was just committed (real ref, fallback, or router cancel -- all
        call this). Never raises: a missing/stale timing dict must not break
        injection."""
        try:
            start = getattr(self, "_turn_timing_start", None)
            stages = getattr(self, "_turn_timing_stages", None)
            if start is None or not stages:
                return
            total = time.perf_counter() - start
            self.conv_logger.turn_timing_summary(self.search_turn_epoch, total, dict(stages))
            self.conv_logger.narrate_timing_summary(self.search_turn_epoch, total, dict(stages))
        except Exception:
            pass

    def _consume_pending(self) -> None:
        """Called once per chunk while a routing/search decision is in flight.
        This is the ONLY place the background thread's work reaches the LM,
        because token injection must happen on the GPU thread.

        Three outcomes, checked in priority order:
          1. cancelled  -- the router decided no search is needed; stop the
                           thinking sound and let the model answer normally.
                           Nothing is injected, so the context is untouched.
          2. ref ready  -- inject the grounded <ref> block.
          3. timed out  -- after self._SEARCH_MAX_FILLER_FRAMES chunks
                           (--search_max_filler_sec, default 6.0s) inject a
                           generic fallback so the model never hangs waiting on
                           a slow or failed search.
        The <lookup> filler is injected on whatever chunk the background thread
        made it available, which is only ever after a search was committed."""
        import search_helpers

        self.search_filler_frame_count += 1

        if self.pending_search_cancelled:
            self.pending_search_cancelled = False
            self.pending_lookup_tokens = None
            self.pending_start_thinking = False
            self.suppress_text_until_ref = False
            self.search_awaiting_ref = False
            self.search_ref_committed_this_turn = True
            # Normally a no-op now: the router path never starts the sound
            # before deciding, so on a "no search" verdict there is nothing
            # playing to stop. Kept because the rules path CAN have started it
            # already, and a later cancel must still silence it.
            self._stop_thinking_sound(
                self.search_turn_epoch, "no_search_needed",
                "the assistant already knew this and did not need to search",
            )
            self.conv_logger.latency.count(
                self._latency_turn_id, injected_anything=False,
            )
            print(
                "[liveTryPlasticity][search] router said no search -- answering from "
                "the model's own knowledge (nothing injected)",
                flush=True,
            )
            return

        # A search is really in flight -> start the cue (idempotent, so the
        # rules path having already started it is fine). Skipped when the
        # answer is ALREADY waiting: a search that completed inside this same
        # 80ms tick would otherwise start the clip and stop it again a few
        # lines below, emitting a single-chunk blip of noise for no reason.
        if self.pending_start_thinking:
            self.pending_start_thinking = False
            if self.pending_ref_tokens is None:
                self._start_thinking_sound(self.search_turn_epoch, self.search_current_transcript)

        if self.pending_lookup_tokens is not None:
            lookup_tokens = self.pending_lookup_tokens
            self.pending_lookup_tokens = None
            t_lookup0 = time.perf_counter()
            self._inject_tokens(lookup_tokens)
            lookup_elapsed = time.perf_counter() - t_lookup0
            self._turn_timing_stages["lookup_inject"] = lookup_elapsed
            self.conv_logger.latency.stage(
                self._latency_turn_id, "lookup_inject", lookup_elapsed,
                note=f"{len(lookup_tokens)} token(s)",
            )
            self.conv_logger.latency.mark(self._latency_turn_id, "lookup_injected")
            self.conv_logger.ref_injected(
                self.tokenizer.decode(lookup_tokens), len(lookup_tokens), lookup_elapsed, kind="lookup"
            )
            print(
                f"[liveTryPlasticity][search] <lookup> injected ({len(lookup_tokens)} tok) "
                f"-- searching in background",
                flush=True,
            )

        if self.pending_ref_tokens is not None:
            ref_tokens = self.pending_ref_tokens
            self.pending_ref_tokens = None
            self.search_awaiting_ref = False
            self.search_ref_committed_this_turn = True
            # Release the text hold BEFORE injecting: the very next normal
            # generation step must be free to speak the grounded answer.
            self.suppress_text_until_ref = False
            # Stop the thinking sound BEFORE injecting -- the injection burst
            # itself produces no audio (forced-token steps bypass decode), so
            # the very next normal generation step in this same _step() call
            # is the model's real, grounded reply: no silent gap, no overlap.
            self._stop_thinking_sound(self.search_turn_epoch, "ref_ready", "the answer was ready")
            t0 = time.perf_counter()
            self._inject_tokens(ref_tokens)
            elapsed = time.perf_counter() - t0
            self._turn_timing_stages["ref_inject"] = elapsed
            self.conv_logger.latency.stage(
                self._latency_turn_id, "ref_inject", elapsed,
                note=f"{len(ref_tokens)} token(s) fed into the live context",
            )
            self.conv_logger.latency.mark(
                self._latency_turn_id, "ref_injected",
                f"after {self.search_filler_frame_count} filler chunk(s)",
            )
            self.conv_logger.ref_injected(self.tokenizer.decode(ref_tokens), len(ref_tokens), elapsed, kind="ref")
            n_before, max_tok = self._pending_ref_token_counts
            self.conv_logger.narrate_injection(
                self.search_turn_epoch, self.tokenizer.decode(ref_tokens), len(ref_tokens),
                n_before, max_tok, kind="ref",
            )
            self.conv_logger.turn_done(
                self.search_turn_epoch, "grounded from web search",
                time.perf_counter() - (self._turn_timing_start or time.perf_counter()),
            )
            self._log_timing_summary()
            print(
                f"[liveTryPlasticity][search] <ref> injected ({len(ref_tokens)} tok) "
                f"after {self.search_filler_frame_count} filler chunks",
                flush=True,
            )
        elif self.search_filler_frame_count >= self._SEARCH_MAX_FILLER_FRAMES:
            fallback_text = "There's no specific information available on this, so answer from general knowledge."
            fallback = self.tokenizer.encode(search_helpers.wrap_with_ref_tags(fallback_text))
            self.search_awaiting_ref = False
            self.search_ref_committed_this_turn = True
            self.suppress_text_until_ref = False
            self._stop_thinking_sound(
                self.search_turn_epoch, "filler_timeout",
                "the search was taking too long, so the assistant moved on with what it had",
            )
            t0 = time.perf_counter()
            self._inject_tokens(fallback)
            fallback_inject_elapsed = time.perf_counter() - t0
            self._turn_timing_stages["ref_inject"] = fallback_inject_elapsed
            self.conv_logger.latency.stage(
                self._latency_turn_id, "ref_inject", fallback_inject_elapsed,
                note=f"{len(fallback)} token(s), generic fallback after filler timeout",
            )
            self.conv_logger.latency.mark(
                self._latency_turn_id, "ref_injected",
                f"filler timeout after {self.search_filler_frame_count} chunk(s) -- "
                f"the search did not finish in time",
            )
            self.conv_logger.latency.count(
                self._latency_turn_id, search_timed_out=True,
                grounding_tokens_injected=len(fallback),
            )
            self.conv_logger.ref_injected(fallback_text, len(fallback), fallback_inject_elapsed, kind="ref_fallback")
            self.conv_logger.narrate_injection(
                self.search_turn_epoch, fallback_text, len(fallback), len(fallback),
                self.max_ref_tokens, kind="ref_fallback",
            )
            self.conv_logger.turn_done(
                self.search_turn_epoch, "search timed out, answered from own knowledge",
                time.perf_counter() - (self._turn_timing_start or time.perf_counter()),
            )
            self._log_timing_summary()
            print("[liveTryPlasticity][search] <ref> fallback injected after filler timeout", flush=True)

    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:
        self.step += 1
        t0 = time.perf_counter()
        chunk = torch.from_numpy(pcm24).to(self.device, dtype=torch.float32)[None, None]

        t_encode0 = time.perf_counter()
        codes = self.mimi.encode(chunk)
        t_encode1 = time.perf_counter()
        if self.skip_first:
            self.mimi.reset_streaming()
            self.skip_first = False

        # -- STT/VAD forward pass + turn-boundary detection, then consume any
        # in-flight routing/search result. No-op (both guards false) unless
        # --stt_hf_repo/--stt_pkg_dir were configured at launch, so the plain
        # conversational path is untouched.
        #
        # Both calls run inside try/except: this hook runs on every single
        # chunk of the live conversation, so an uncaught exception here would
        # otherwise propagate out of _step() and kill the entire GPU producer
        # thread -- audio and video generation stop forever, with nothing
        # captured in either conversation log. That failure signature (log
        # ends mid-turn, no error line, no further component_status entries)
        # matches what conversation_logs_4 shows, though the exact exception
        # was never captured there, so this is a closed gap, not a proven
        # root cause. On failure, search is disabled for the rest of this
        # session but the avatar keeps talking. --
        # Real per-chunk cost of the STT/VAD forward pass, unconditionally
        # measured (not gated to "the model is speaking" like the gen_lm/
        # gen_audio_* stages below) -- this pass runs on every chunk whether
        # idle or answering, so its cost has to be visible in the idle
        # periods too if it's the thing pushing total_ms past the ~80ms
        # real-time budget of one chunk. Logging-only: never gates behavior.
        t_stt0 = time.perf_counter()
        stt_step_ms = 0.0
        if self.stt_lm_gen is not None and not self.search_hard_disabled:
            try:
                self._stt_step(chunk)
                stt_step_ms = 1000.0 * (time.perf_counter() - t_stt0)
            except Exception as e:
                tb = traceback.format_exc()
                print(
                    f"[liveTryPlasticity][search] _stt_step failed, disabling search "
                    f"for the rest of this session: {e!r}\n{tb}",
                    flush=True,
                )
                self.conv_logger.error("stt_step", e, tb)
                self.search_hard_disabled = True
                self.search_awaiting_ref = False
                self.search_thinking_active = False
                self.suppress_text_until_ref = False
        if self.search_awaiting_ref and not self.search_hard_disabled:
            try:
                self._consume_pending()
            except Exception as e:
                tb = traceback.format_exc()
                print(
                    f"[liveTryPlasticity][search] _consume_pending failed, disabling "
                    f"search for the rest of this session: {e!r}\n{tb}",
                    flush=True,
                )
                self.conv_logger.error("consume_pending", e, tb)
                self.search_hard_disabled = True
                self.search_awaiting_ref = False
                self.search_thinking_active = False
                self.suppress_text_until_ref = False

        # While a search is in flight, hand the model its own "say nothing"
        # token instead of letting it sample text. process_transformer_output
        # honors a provided token rather than sampling, so this is the model's
        # native silence mechanism (the same one _step_audio_silence_core uses)
        # -- not a hack bolted on top. Audio and hidden states keep flowing, so
        # the avatar pipeline and chunk cadence are untouched; only the words
        # are withheld, and they are withheld precisely while the model would
        # otherwise be inventing an answer it is about to be handed.
        t_lm0 = time.perf_counter()
        if self.suppress_text_until_ref and self._step_supports_text_token:
            lm_out = self.lm_gen._step(
                codes[:, :, :1], text_token=getattr(self.lm_gen, "zero_text_code", 3)
            )
        else:
            lm_out = self.lm_gen._step(codes[:, :, :1])
        t_lm1 = time.perf_counter()

        tokens = None
        helium_hidden = None
        if lm_out is not None:
            if not (isinstance(lm_out, tuple) and len(lm_out) == 3):
                raise RuntimeError(f"Moshi graph layer[-2] contract failure: got {type(lm_out)} len={len(lm_out) if isinstance(lm_out, tuple) else 'n/a'}")
            tokens, _transformer_out, layer_hidden = lm_out
            helium_hidden = layer_hidden[:1, -1:].detach().float().cpu()

        token = -1
        token_piece = ""
        decode_ms = 0.0
        reply_codes = None
        if tokens is None:
            reply_pcm = np.zeros(MIMI_FRAME_SIZE, dtype=np.float32)
        else:
            token = int(tokens[0, 0, 0].detach().item())
            token_piece = self.decode_piece(token)
            if token_piece:
                self.audio_text += token_piece
                # Running end-of-answer marker for the latency log: the last
                # text token the model emitted. Frozen in _stt_step the moment
                # the user starts speaking again (see _answer_end_perf).
                self._last_text_emit_perf = time.perf_counter()
                self._turn_text_token_count += 1
            reply_codes = tokens[:, 1:].detach().to(device="cpu", dtype=torch.int16)
            t_decode0 = time.perf_counter()
            reply = self.mimi.decode(tokens[:, 1:])
            reply_pcm = reply[0, 0].detach().float().cpu().numpy()
            decode_ms = 1000.0 * (time.perf_counter() - t_decode0)
            # First decoded audio sample of this turn, independent of the RMS
            # gate below (that one is deliberately kept as the separate
            # "first_word" heuristic for debugging -- this is the honest
            # "the model produced ANY audio" instant).
            if self._latency_turn_id is not None and not self._turn_model_first_audio_marked:
                self._turn_model_first_audio_marked = True
                self.conv_logger.latency.mark(self._latency_turn_id, "model_first_audio")
            if reply_pcm.shape[0] < MIMI_FRAME_SIZE:
                reply_pcm = np.pad(reply_pcm, (0, MIMI_FRAME_SIZE - reply_pcm.shape[0]))
            elif reply_pcm.shape[0] > MIMI_FRAME_SIZE:
                reply_pcm = reply_pcm[:MIMI_FRAME_SIZE]

        # The model's OWN output level, measured before the thinking sound can
        # replace it below: after the swap, `reply_pcm` may be the filler clip,
        # whose level says nothing about whether the model is speaking.
        model_own_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))

        # First real audio of this turn -> the honest end-to-end latency.
        was_awaiting_first_speech = self._turn_awaiting_first_speech
        if self._turn_awaiting_first_speech and model_own_rms > self._SPEECH_RMS_THRESHOLD:
            self._turn_awaiting_first_speech = False
            started = self._turn_timing_start
            if started is not None:
                backlog_s = self.input_backlog_sec()
                self.conv_logger.turn_spoke(
                    self._turn_first_speech_epoch,
                    time.perf_counter() - started,
                    backlog_s,
                )
                self.conv_logger.latency.mark(
                    self._turn_first_speech_epoch, "first_word",
                    f"microphone backlog {backlog_s:.2f}s at this moment",
                )
                self.conv_logger.latency.count(
                    self._turn_first_speech_epoch,
                    input_backlog_s_at_first_word=round(backlog_s, 3),
                )

        # "Thinking sound": while an online search is in flight, replace what
        # the model would otherwise output with the looped clip. The model
        # keeps stepping normally above (KV cache / timing untouched); only the
        # audio actually sent out is swapped. force_idle tells the avatar-motion
        # gate downstream (in _gpu_producer_thread) to stay visually idle rather
        # than lip-syncing to this non-speech sound (its real RMS is well above
        # the speech threshold, so without this the avatar would appear to
        # "talk"). There is no "model started speaking" stop condition any more:
        # the clip now only ever covers a real search, and that has exactly two
        # ends -- the <ref> landing or the filler timeout, both in
        # _consume_pending.
        force_idle = False
        if self.search_thinking_active and self.thinking_sound_pcm is not None:
            reply_pcm = self._next_thinking_sound_chunk()
            force_idle = True

        reply_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))
        reply_peak = float(np.max(np.abs(reply_pcm))) if reply_pcm.size else 0.0
        input_rms = float(np.sqrt(np.mean(np.square(pcm24, dtype=np.float32))))
        encode_ms = 1000.0 * (t_encode1 - t_encode0)
        lm_ms = 1000.0 * (t_lm1 - t_lm0)
        total_ms = 1000.0 * (time.perf_counter() - t0)

        # Charge this 80ms frame's GPU cost to the turn being timed. Accumulated
        # silently (no per-frame log lines -- there are 12.5 of these per
        # second) and reported once in the turn's block, which is what answers
        # "how much of the wait was the model actually generating?".
        #
        # Only frames that belong to the answer are charged: the ones the user
        # spends waiting for the first word, and the ones the model is audibly
        # speaking in. The model keeps stepping at 12.5Hz through the silence
        # after an answer too, and counting that would turn a long pause before
        # the next question into fictitious "generation time" that dwarfs every
        # real stage.
        if self._latency_turn_id is not None and (
            was_awaiting_first_speech or model_own_rms > self._SPEECH_RMS_THRESHOLD
        ):
            lat = self.conv_logger.latency
            turn_id = self._latency_turn_id
            lat.accumulate(turn_id, "gen_lm", (t_lm1 - t_lm0))
            lat.accumulate(turn_id, "gen_audio_encode", (t_encode1 - t_encode0))
            lat.accumulate(turn_id, "gen_audio_decode", decode_ms / 1000.0)
        # Unconditional (not gated to "the model is speaking"): the STT/VAD
        # forward pass runs on every chunk, including the idle stepping
        # BEFORE the user's speech-end fires this turn open, so gating it the
        # same way as gen_lm/gen_audio_* above would hide exactly the cost
        # this is meant to surface. Charged to whichever turn is currently
        # open (or the previous one, during the idle tail before the next
        # turn opens) purely for diagnostic visibility in the per-turn block.
        if self._latency_turn_id is not None and stt_step_ms > 0.0:
            self.conv_logger.latency.accumulate(
                self._latency_turn_id, "stt_forward_per_chunk", stt_step_ms / 1000.0
            )

        reply_i16 = np.clip(reply_pcm, -1.0, 1.0)
        reply_i16 = (reply_i16 * 32767.0).astype(np.int16)
        audio_b64 = base64.b64encode(reply_i16.tobytes()).decode("ascii")

        # One Moshi step corresponds to one ~80ms real-time chunk; total_ms
        # exceeding that is exactly what a standing input backlog is made of
        # (see input_backlog_sec's docstring). Flagged inline so this is
        # grep-able ("OVER_BUDGET") straight out of the console log instead
        # of needing a separate analysis pass.
        over_budget = " OVER_BUDGET" if total_ms > 80.0 else ""
        print(
            "[liveTryStudio] moshi "
            f"step={self.step} token={token} piece={token_piece!r} "
            f"in_rms={input_rms:.5f} reply_rms={reply_rms:.5f} peak={reply_peak:.3f} "
            f"hidden={helium_hidden is not None} "
            f"encode={encode_ms:.1f}ms lm={lm_ms:.1f}ms decode={decode_ms:.1f}ms "
            f"stt={stt_step_ms:.1f}ms total={total_ms:.1f}ms{over_budget}",
            flush=True,
        )

        return {
            "step": int(self.step),
            "sample_rate": TARGET_SR,
            "reply_i16_b64": audio_b64,
            "reply_rms": reply_rms,
            "reply_peak": reply_peak,
            "force_idle": force_idle,
            "input_rms": input_rms,
            "token": token,
            "piece": token_piece,
            "sampled_text": self.sampled_text,
            "audio_text": self.audio_text,
            "encode_ms": encode_ms,
            "lm_ms": lm_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "helium_hidden": helium_hidden,
            "reply_codes": reply_codes,
        }

    @torch.no_grad()
    def process_ready_steps_limited(self, max_steps: int) -> list[dict]:
        """Process a bounded number of Mimi frames.

        The base MoshiOnlyEngine drains the entire input buffer before returning.
        In live mode that is dangerous: mic audio can accumulate while Moshi is
        loading, then avatar frames do not reach the sender until the backlog is
        fully processed. Bounded draining keeps the producer/sender interleaved.
        """
        events: list[dict] = []
        for _ in range(max(1, int(max_steps))):
            if self.input_buffer.shape[0] < MIMI_FRAME_SIZE:
                break
            pcm = self.input_buffer[:MIMI_FRAME_SIZE].copy()
            self.input_buffer = self.input_buffer[MIMI_FRAME_SIZE:].copy()
            events.append(self._step(pcm))
        return events


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(t0: float) -> float:
    return 1000.0 * (time.perf_counter() - t0)


def encode_jpeg_b64(frame_rgb: np.ndarray, quality: int) -> str:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def encode_jpeg_bytes(frame_rgb: np.ndarray, quality: int) -> bytes:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()


def _pcm_f32_to_i16_b64(pcm: np.ndarray) -> str:
    arr = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return base64.b64encode((arr * 32767.0).astype(np.int16).tobytes()).decode("ascii")


def _pcm_f32_to_i16_bytes(pcm: np.ndarray) -> bytes:
    arr = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def split_audio_into_frame_slices(pcm: np.ndarray, fps: float) -> list[np.ndarray]:
    frame_samples = int(round(TARGET_SR / float(fps)))
    arr = np.asarray(pcm, dtype=np.float32)
    n_frames = max(0, int(round(arr.shape[0] / frame_samples)))
    if n_frames == 0:
        return []
    total = n_frames * frame_samples
    if arr.shape[0] < total:
        arr = np.pad(arr, (0, total - arr.shape[0]))
    elif arr.shape[0] > total:
        arr = arr[:total]
    return [arr[i * frame_samples:(i + 1) * frame_samples].copy() for i in range(n_frames)]


def load_audio_24k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).float().numpy()


def load_ref_image(path: str | Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
    return T.ToTensor()(img).unsqueeze(0).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# FM + Renderer weight loading (identical to liveTryFM.py)
# ---------------------------------------------------------------------------

def _clean_generator_state(ckpt: dict) -> dict:
    raw = ckpt.get("ema_state_dict") or ckpt.get("state_dict", ckpt.get("model", ckpt))
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    return {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in raw.items()}


def _load_fm(args: argparse.Namespace, device: torch.device) -> FMGenerator:
    t_total = time.perf_counter()
    fm = FMGenerator(args).to(device).eval()
    ckpt = torch.load(args.generator_path, map_location="cpu")
    cleaned = _clean_generator_state(ckpt)
    missing, unexpected = fm.load_state_dict(cleaned, strict=False)
    print(
        f"[liveTryHeliumFM][FM] base loaded missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    lora_path = str(getattr(args, "lora_generator_path", "") or "")
    if lora_path:
        apply_lora_to_model(
            fm,
            rank=int(getattr(args, "lora_rank", 64) or 64),
            alpha=float(getattr(args, "lora_alpha", 128) or 128),
            dropout=float(getattr(args, "lora_dropout", 0.05)),
            include_pose_lora=not bool(getattr(args, "no_lora_pose_projection", False)),
            include_audio_lora=not bool(getattr(args, "no_lora_audio_projection", False)),
            only_pose_lora=bool(getattr(args, "only_lora_pose_projection", False)),
        )
        lora_ckpt = torch.load(lora_path, map_location="cpu")
        lora_cleaned = _clean_generator_state(lora_ckpt)
        missing_lora, unexpected_lora = fm.load_state_dict(lora_cleaned, strict=False)
        lora_keys = sum(1 for key in lora_cleaned if "lora_" in key)
        print(
            f"[liveTryHeliumFM][FM] lora loaded path={lora_path} "
            f"lora_keys={lora_keys} missing={len(missing_lora)} unexpected={len(unexpected_lora)}",
            flush=True,
        )
    fm.to(device).eval()
    _sync_cuda()
    print(f"[liveTryHeliumFM][FM] loaded in {_ms(t_total):.0f}ms", flush=True)
    return fm


def _load_renderer(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> IMTRenderer:
    t_total = time.perf_counter()
    renderer = IMTRenderer(args).to(device).eval()
    ckpt = torch.load(args.renderer_path, map_location="cpu")
    raw = ckpt.get("state_dict", ckpt.get("model", ckpt))
    cleaned = {k.replace("gen.", "", 1).replace("model.", "", 1): v for k, v in raw.items()}
    missing, unexpected = renderer.load_state_dict(cleaned, strict=False)
    renderer = renderer.to(dtype=dtype)
    _sync_cuda()
    if getattr(args, "compile_renderer", False):
        @torch.no_grad()
        def _fused_render(motion_latent, g_r, m_r, f_r):
            ta_c = renderer.adapt(motion_latent, g_r)
            m_c = renderer.latent_token_decoder(ta_c)
            frames = renderer.decode(m_c, m_r, f_r)
            return frames
        renderer._fused_render = torch.compile(_fused_render)
    print(
        f"[liveTryHeliumFM][renderer] loaded in {_ms(t_total):.0f}ms "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return renderer


# ---------------------------------------------------------------------------
# Helium extractor (chunk-local, batch LM)
# ---------------------------------------------------------------------------

class HeliumExtractor:
    """Prefix-growing raw Helium + global interpolation.

    This matches the best Stage 3 diagnostic:
      audio prefix -> raw Helium -> append only new raw steps
      -> one global interpolation over accumulated raw Helium
      -> emit last target_frames
    """

    def __init__(
        self,
        helium_mimi: "MimiModel",
        helium_lm: "LMModel",
        device: torch.device,
    ) -> None:
        self.helium_mimi = helium_mimi
        self.helium_lm = helium_lm
        self.device = device
        self._lock = threading.Lock()
        self._prefix_pcm = np.empty(0, dtype=np.float32)
        self._raw_parts: list[torch.Tensor] = []
        self._prev_raw_len = 0
        self._emitted_frames = 0

    def reset(self) -> None:
        with self._lock:
            self._prefix_pcm = np.empty(0, dtype=np.float32)
            self._raw_parts = []
            self._prev_raw_len = 0
            self._emitted_frames = 0

    def _extract_raw(self, pcm_np: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(np.asarray(pcm_np, dtype=np.float32)).to(self.device, dtype=torch.float32)[None, None]

        codes = self.helium_mimi.encode(wav)
        codes = codes[:, :MAIN_CODEBOOKS, :].detach()
        batch_size, n_q, total_steps = codes.shape

        dtype = next(self.helium_lm.parameters()).dtype
        input_emb = torch.zeros(
            batch_size, total_steps, self.helium_lm.dim, device=self.device, dtype=dtype
        )
        for q in range(n_q):
            input_emb = input_emb + self.helium_lm.emb[q](codes[:, q].long())

        padding_ids = torch.full(
            (batch_size, total_steps),
            self.helium_lm.existing_text_padding_id,
            dtype=torch.long,
            device=self.device,
        )
        input_emb = input_emb + self.helium_lm.text_emb(padding_ids)

        if getattr(self.helium_lm.transformer, "_streaming_state", None) is not None:
            raise RuntimeError("helium_lm must stay in batch mode (non-streaming)")

        captured: list[torch.Tensor] = []

        def _hook(_mod, _inp, out):
            captured.append(out.detach())

        handle = self.helium_lm.transformer.layers[-2].register_forward_hook(_hook)
        try:
            self.helium_lm.transformer(input_emb)
        finally:
            handle.remove()

        if len(captured) != 1:
            raise RuntimeError(f"Helium hook captured {len(captured)} tensors; expected 1")
        return captured[0].squeeze(0).float().contiguous()  # [T_raw, 4096]

    @torch.no_grad()
    def extract_raw_chunk(self, pcm_np: np.ndarray) -> torch.Tensor:
        """Return only the new raw 12.5Hz Helium steps for one new audio chunk."""
        pcm = np.asarray(pcm_np, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0:
            raise RuntimeError("HeliumExtractor.extract_raw_chunk expects non-empty 1D PCM")

        with self._lock:
            self._prefix_pcm = np.concatenate([self._prefix_pcm, pcm], axis=0)
            raw_prefix = self._extract_raw(self._prefix_pcm)
            new_raw = raw_prefix[self._prev_raw_len:]
            if int(new_raw.shape[0]) == 0 and int(raw_prefix.shape[0]) > 0:
                new_raw = raw_prefix[-1:]
            self._raw_parts.append(new_raw.cpu())
            self._prev_raw_len = int(raw_prefix.shape[0])
            return new_raw.contiguous()

    @torch.no_grad()
    def extract_exact_chunk_from_prefix(
        self,
        pcm_prefix: np.ndarray,
        chunk_start_frame: int,
        target_frames: int,
    ) -> torch.Tensor:
        """Return the raw 12.5Hz Helium slice for a video-frame window.

        This path is used by file-mode lookahead. It must return raw Helium
        steps so the studio adapter remains the only temporal upsampler.
        """
        pcm = np.asarray(pcm_prefix, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0:
            raise RuntimeError("HeliumExtractor.extract_exact_chunk_from_prefix expects non-empty 1D PCM")
        with self._lock:
            raw_prefix = self._extract_raw(pcm)
        start_frame = int(chunk_start_frame)
        end_frame = start_frame + int(target_frames)
        start_raw = int(round(start_frame * 0.5))
        end_raw = int(round(end_frame * 0.5))
        start_raw = max(0, min(start_raw, int(raw_prefix.shape[0])))
        end_raw = max(start_raw + 1, min(end_raw, int(raw_prefix.shape[0])))
        if end_raw > int(raw_prefix.shape[0]):
            raise RuntimeError(
                f"Requested raw slice [{start_raw}, {end_raw}) exceeds prefix Helium length {raw_prefix.shape[0]}"
            )
        return raw_prefix[start_raw:end_raw].contiguous()


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class LiveHeliumFMEngine:
    """Helium extraction + FM + renderer, session-stateful."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.args = args
        self.device = torch.device(args.device)
        renderer_precision = str(
            getattr(args, "renderer_precision", "fp32")
        ).lower()
        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[renderer_precision]
        self.fps = float(args.fps)
        self.audio_chunk_sec = float(getattr(args, "audio_chunk_sec", 0.96))
        self.audio_chunk_samples = int(round(self.audio_chunk_sec * TARGET_SR))
        self.fm_chunk_frames = max(1, int(getattr(args, "fm_chunk_frames", 24)))
        self.live_sliding_window = bool(getattr(args, "enable_live_sliding_window", False))
        self.slide_past_frames = max(0, int(getattr(args, "slide_past_frames", 10)))
        self.slide_future_frames = max(0, int(getattr(args, "slide_future_frames", 3)))
        self.render_sub_batch = max(1, int(args.render_sub_batch))
        self.jpeg_quality = int(args.jpeg_quality)
        trained_window = int(round(float(args.wav2vec_sec) * self.fps))
        if self.fm_chunk_frames != trained_window:
            print(
                f"[liveTryHeliumFM] WARNING fm_chunk_frames={self.fm_chunk_frames} "
                f"but wav2vec_sec*fps={trained_window}",
                flush=True,
            )
        if self.live_sliding_window:
            print(
                f"[liveTryHeliumFM][typeAC] IMTalker lookahead enabled "
                f"past={self.slide_past_frames}f current={self.fm_chunk_frames}f "
                f"future={self.slide_future_frames}f",
                flush=True,
            )

        t_total = time.perf_counter()

        # FM + renderer
        self.fm = _load_fm(args, self.device)
        self.renderer = _load_renderer(args, self.device, self.dtype)

        # Adapter: either the six-layer projected-frontend model or UniTalk's
        # 12-layer model with only its final layer passed to IMTalker.
        t_adapter = time.perf_counter()
        if args.adapter_type == "unitalk_last_layer":
            self.studio_adapter = UniTalkLastLayerLiveAdapter(
                args.wav2vec_model_path,
                args.adapter_dropout,
            ).to(self.device).float().eval()
        else:
            self.studio_adapter = StudioNativeLiveAdapter(
                args.wav2vec_model_path,
                args.adapter_num_layers,
                args.adapter_dropout,
            ).to(self.device).float().eval()
        payload = torch.load(args.adapter_path, map_location="cpu")
        if isinstance(payload, dict) and args.adapter_type == "frontend":
            saved_args = payload.get("args", {})
            if saved_args and int(saved_args.get("num_layers", args.adapter_num_layers)) != int(args.adapter_num_layers):
                print(
                    f"[liveTryHeliumFrontendFM] WARNING checkpoint num_layers={saved_args.get('num_layers')} "
                    f"but CLI adapter_num_layers={args.adapter_num_layers}",
                    flush=True,
                )
            state = payload.get("adapter", payload.get("model", payload))
        else:
            state = payload
        self.studio_adapter.load_state_dict(state, strict=True)
        _sync_cuda()
        print(
            f"[liveTryHeliumFrontendFM][adapter] type={args.adapter_type} loaded in {_ms(t_adapter):.0f}ms "
            f"path={args.adapter_path}",
            flush=True,
        )

        # Raw HF Wav2Vec2 target path used during Helium adapter training.
        t_w2v = time.perf_counter()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_model_path,
            local_files_only=True,
        )
        self.wav2vec_model = self.studio_adapter.wav2vec
        _sync_cuda()
        print(
            f"[liveTryHeliumStudioFM][wav2vec] loaded in {_ms(t_w2v):.0f}ms "
            f"path={args.wav2vec_model_path}",
            flush=True,
        )

        # Reference image: pre-compute identity + motion-ref features once
        ref_tensor = load_ref_image(args.ref_path, self.device, self.dtype)
        with torch.no_grad():
            self.f_r, self.g_r = self.renderer.dense_feature_encoder(ref_tensor)
            self.ref_x = self.renderer.latent_token_encoder(ref_tensor).to(dtype=torch.float32)
            ta_r = self.renderer.adapt(self.ref_x.to(dtype=self.dtype), self.g_r)
            self.m_r = self.renderer.latent_token_decoder(ta_r)
        _sync_cuda()
        self.eye_blink_enabled = bool(getattr(args, "enable_eye_blink_composite", False))
        self._blink_maps: tuple[torch.Tensor, ...] | None = None
        self._eye_masks: tuple[torch.Tensor, ...] | None = None
        self._render_frame_cursor: int = 0
        if self.eye_blink_enabled:
            self._init_eye_blink_composite()

        # Moshi models for Helium extraction
        self._init_moshi(args)

        # Optional local audio file (simulate-live mode)
        self.audio_pcm: np.ndarray | None = None
        if getattr(args, "audio_path", "") and Path(args.audio_path).is_file():
            self.audio_pcm = load_audio_24k(args.audio_path)
            print(
                f"[liveTryHeliumFM] audio_path loaded: {self.audio_pcm.shape[0]/TARGET_SR:.2f}s "
                f"chunk={self.audio_chunk_sec:.3f}s/{self.audio_chunk_samples} samples "
                f"fm_chunk={self.fm_chunk_frames}f",
                flush=True,
            )

        # Shared noise tensor (pre-generated, indexed by absolute frame position)
        self.noise_buf: torch.Tensor | None = None
        if getattr(args, "shared_noise", False):
            max_frames = int(getattr(args, "noise_max_frames", 5000))
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(getattr(args, "noise_seed", 1234)))
            self.noise_buf = torch.randn(
                1, max_frames, int(args.dim_w), device=self.device, generator=gen
            )
            print(f"[liveTryHeliumFM] shared noise buf: {tuple(self.noise_buf.shape)}", flush=True)

        # Per-session state (reset on each new client)
        self.stream_state: dict | None = None
        self.abs_frame: int = 0
        self.helium_context_tail: torch.Tensor | None = None
        self.helium_deque_size: int = 100
        self.helium_deque: torch.Tensor | None = None
        self.helium_deque_filled: int = 0
        self.silence_helium_seed: torch.Tensor | None = None
        silence_helium_path = str(
            getattr(args, "silence_helium_path", "") or ""
        )
        if silence_helium_path:
            payload = torch.load(silence_helium_path, map_location="cpu")
            seed = (
                payload.get("silence_helium_mean")
                if isinstance(payload, dict)
                else payload
            )
            if not isinstance(seed, torch.Tensor) or seed.numel() != 4096:
                raise RuntimeError(
                    f"Invalid silence Helium seed: {silence_helium_path}"
                )
            self.silence_helium_seed = seed.reshape(1, 4096).to(
                device=self.device, dtype=torch.float32
            )
            print(
                f"[liveTryHeliumFM] silence Helium seed loaded: "
                f"{silence_helium_path}",
                flush=True,
            )
        self._pcm_accum: np.ndarray = np.empty(0, dtype=np.float32)
        self.dump_motion = bool(getattr(args, "dump_motion", False))
        self.dump_dir = Path(getattr(args, "dump_dir", ROOT / "live_try_dumps"))
        self._session_motion_parts: list[torch.Tensor] = []
        self._session_helium_parts: list[torch.Tensor] = []
        self._session_adapter_50_parts: list[torch.Tensor] = []
        self._session_adapter_25_parts: list[torch.Tensor] = []
        self._session_projected_audio_parts: list[torch.Tensor] = []
        self._session_audio_parts: list[np.ndarray] = []
        self._session_live_token_parts: list[torch.Tensor] = []
        self._session_chunk_rows: list[dict] = []
        self._session_reply_events: list[dict] = []
        self._session_started_wall: float = time.time()

        # JPEG encoding thread pool (CPU-only work, parallelizable)
        self._jpeg_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="jpeg"
        )

        # Warmup
        self._warmup()

        print(
            f"[liveTryHeliumFM] ready — total startup {_ms(t_total):.0f}ms "
            f"fm_chunk={self.fm_chunk_frames} render_sub={self.render_sub_batch} "
            f"dtype={self.dtype}",
            flush=True,
        )

    def _init_moshi(self, args: argparse.Namespace) -> None:
        if bool(getattr(args, "direct_reply_hidden", False)) and bool(getattr(args, "enable_moshi_reply", False)):
            self.extractor = None
            print("[liveTryHeliumStudioFM] using direct Moshi reply hidden; batch Helium extractor skipped", flush=True)
            return

        from generate_helium import load_mimi_and_lm

        t0 = time.perf_counter()
        helium_mimi, helium_lm, _ = load_mimi_and_lm(args)
        helium_mimi.eval()
        helium_lm.eval()
        print(
            f"[liveTryHeliumFM] Moshi loaded in {_ms(t0):.0f}ms "
            f"dtype={next(helium_lm.parameters()).dtype}",
            flush=True,
        )

        self.extractor = HeliumExtractor(helium_mimi, helium_lm, self.device)

    def reset_session(self) -> None:
        """Call when a new WebSocket client connects or sends 'start'."""
        self.stream_state = None
        self.abs_frame = 0
        self._render_frame_cursor = 0
        self.helium_context_tail = None
        self.helium_deque = None
        self.helium_deque_filled = 0
        self._pcm_accum = np.empty(0, dtype=np.float32)
        if self.extractor is not None:
            self.extractor.reset()
        self._session_motion_parts = []
        self._session_helium_parts = []
        self._session_adapter_50_parts = []
        self._session_adapter_25_parts = []
        self._session_projected_audio_parts = []
        self._session_audio_parts = []
        self._session_live_token_parts = []
        self._session_chunk_rows = []
        self._session_reply_events = []
        self._session_started_wall = time.time()

    @torch.no_grad()
    def _warmup(self) -> None:
        dummy_pcm = np.zeros(self.audio_chunk_samples, dtype=np.float32)

        t0 = time.perf_counter()
        if self.extractor is None:
            raw_steps = max(1, int(round(self.fm_chunk_frames * 12.5 / float(self.fps))))
            if self.silence_helium_seed is not None:
                dummy_helium = self.silence_helium_seed.expand(
                    raw_steps, -1
                ).contiguous()
            else:
                dummy_helium = torch.zeros(
                    raw_steps, 4096, device=self.device, dtype=torch.float32
                )
            print(f"[liveTryHeliumStudioFM][warmup] raw_helium=skipped direct_hidden raw_steps={raw_steps}", flush=True)
        else:
            dummy_helium = self.extractor.extract_raw_chunk(dummy_pcm)
            _sync_cuda()
            print(f"[liveTryHeliumStudioFM][warmup] raw_helium={_ms(t0):.0f}ms", flush=True)
            self.extractor.reset()

        t0 = time.perf_counter()
        motion, _info = self._sample_motion_from_helium(dummy_helium, self.fm_chunk_frames)
        _sync_cuda()
        print(f"[liveTryHeliumFM][warmup] fm={_ms(t0):.0f}ms motion={tuple(motion.shape)}", flush=True)
        self.stream_state = None
        self.abs_frame = 0
        self._render_frame_cursor = 0
        self.helium_context_tail = None
        self.helium_deque = None
        self.helium_deque_filled = 0

        t0 = time.perf_counter()
        dummy_motion = torch.zeros(self.render_sub_batch, 32, device=self.device, dtype=self.dtype)
        _frames, _timings = self._render_motion(dummy_motion)
        _sync_cuda()
        print(f"[liveTryHeliumFM][warmup] renderer={_ms(t0):.0f}ms", flush=True)
        self._render_frame_cursor = 0

        # Warmup JPEG pool
        t0 = time.perf_counter()
        dummy_np = np.zeros((512, 512, 3), dtype=np.uint8)
        _ = encode_jpeg_b64(dummy_np, self.jpeg_quality)
        print(f"[liveTryHeliumFM][warmup] jpeg={_ms(t0):.0f}ms", flush=True)

        self.stream_state = None

    def feed_pcm(self, pcm_s16le_bytes: bytes) -> Optional[tuple[torch.Tensor, dict, np.ndarray]]:
        pcm = np.frombuffer(pcm_s16le_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.feed_pcm_f32(pcm)

    def feed_pcm_f32(self, pcm_f32: np.ndarray) -> Optional[tuple[torch.Tensor, dict, np.ndarray]]:
        pcm = np.asarray(pcm_f32, dtype=np.float32)
        self._pcm_accum = np.concatenate([self._pcm_accum, pcm])
        if self._pcm_accum.shape[0] < self.audio_chunk_samples:
            return None
        chunk = self._pcm_accum[:self.audio_chunk_samples].copy()
        self._pcm_accum = self._pcm_accum[self.audio_chunk_samples:]
        motion, info = self._process_pcm_chunk(chunk, self.fm_chunk_frames)
        self._record_session_chunk(chunk, motion, info)
        return motion, info, chunk

    @torch.no_grad()
    def _process_pcm_chunk(self, pcm_chunk: np.ndarray, target_frames: int) -> tuple[torch.Tensor, dict]:
        timings: dict = {}
        target_frames = max(1, min(int(target_frames), self.fm_chunk_frames))
        if self.extractor is None:
            raise RuntimeError("direct_reply_hidden mode cannot process raw browser audio directly")

        t0 = time.perf_counter()
        helium = self.extractor.extract_raw_chunk(pcm_chunk)
        timings["helium_ms"] = _ms(t0)
        motion, fm_info = self._sample_motion_from_helium(helium, target_frames)
        timings.update(fm_info)
        return motion, timings

    @torch.no_grad()
    def _sample_motion_from_helium(self, helium: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, dict]:
        timings: dict = {}
        t_adapter = time.perf_counter()
        helium = helium.to(self.device, dtype=torch.float32).contiguous()
        target_frames = int(target_frames)
        current_steps = int(helium.shape[0])
        deque_size = int(getattr(self, "helium_deque_size", 100))
        if self.helium_deque is None:
            if self.silence_helium_seed is not None:
                self.helium_deque = self.silence_helium_seed.expand(
                    deque_size, -1
                ).clone()
            else:
                self.helium_deque = torch.zeros(
                    deque_size,
                    helium.shape[1],
                    device=self.device,
                    dtype=torch.float32,
                )
            self.helium_deque_filled = 0
        if current_steps >= deque_size:
            self.helium_deque = helium[-deque_size:].detach().clone()
            self.helium_deque_filled = deque_size
        else:
            self.helium_deque = torch.cat([self.helium_deque[current_steps:], helium], dim=0).contiguous()
            self.helium_deque_filled = min(deque_size, int(self.helium_deque_filled) + current_steps)

        adapter_window_mode = str(
            getattr(self.args, "adapter_window_mode", "tail")
        )
        if adapter_window_mode == "lookahead":
            # Match training: process the full 8-second window at 50 Hz,
            # emit .96 seconds, and retain .48 seconds as future context.
            future_steps = int(getattr(self.args, "adapter_future_steps", 6))
            target_len_50_full = deque_size * 4
            _baseline, _cnn, feat_50_full = self.studio_adapter.forward_single(
                self.helium_deque, target_len_50_full
            )
            if self.live_sliding_window:
                # Type AC: keep the Type A Helium deque + adapter path, but
                # give IMTalker/FM a small [past + current + future] feature
                # window and emit only current frames. This places lookahead in
                # IMTalker instead of discarding future context inside adapter.
                past_25 = int(self.slide_past_frames)
                future_25 = int(self.slide_future_frames)
                current_25 = int(target_frames)
                past_50 = past_25 * 2
                future_50 = future_25 * 2
                current_50 = current_25 * 2
                full_len_50 = int(feat_50_full.shape[0])
                current_end_50 = full_len_50 - future_50
                current_start_50 = current_end_50 - current_50
                window_start_50 = current_start_50 - past_50
                window_end_50 = current_end_50 + future_50
                if window_start_50 < 0 or current_start_50 < 0 or window_end_50 > full_len_50:
                    raise RuntimeError(
                        "Sliding adapter window is out of range: "
                        f"full={full_len_50} start={window_start_50} "
                        f"current_start={current_start_50} end={window_end_50}"
                    )
                feat_50 = feat_50_full[window_start_50:window_end_50].contiguous()
                feat_25 = F.interpolate(
                    feat_50.T.unsqueeze(0),
                    size=past_25 + current_25 + future_25,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T.contiguous()
            else:
                emitted_frames_50 = max(1, current_steps * 4)
                future_frames_50 = max(0, future_steps * 4)
                segment_end = int(feat_50_full.shape[0]) - future_frames_50
                segment_start = segment_end - emitted_frames_50
                if segment_start < 0:
                    raise RuntimeError(
                        "Look-ahead adapter output is shorter than its emit/future region"
                    )
                feat_50 = feat_50_full[segment_start:segment_end].contiguous()
                feat_25 = F.interpolate(
                    feat_50.T.unsqueeze(0),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T.contiguous()
        else:
            # Legacy behavior: predict at 25 Hz and emit the newest tail.
            target_len_25_full = deque_size * 2
            _baseline, _cnn, feat_25_full = self.studio_adapter.forward_single(
                self.helium_deque, target_len_25_full
            )
            fresh_frames = max(1, current_steps * 2)
            if int(feat_25_full.shape[0]) < fresh_frames:
                raise RuntimeError(
                    f"Deque adapter output too short: got "
                    f"{feat_25_full.shape[0]}, need {fresh_frames}"
                )
            feat_25 = feat_25_full[-fresh_frames:].contiguous()
            feat_50 = feat_25
        if (not self.live_sliding_window) and int(feat_25.shape[0]) != target_frames:
            feat_25 = F.interpolate(
                feat_25.T.unsqueeze(0),
                size=target_frames,
                mode="linear",
                align_corners=False,
            ).squeeze(0).T.contiguous()
        projected_a = self.fm._project_audio(feat_25.unsqueeze(0).float())
        timings["adapter_ms"] = _ms(t_adapter)
        timings["helium_ms"] = timings["adapter_ms"]
        timings["helium_deque_filled"] = int(self.helium_deque_filled)

        data: dict = {"a_feat": feat_25.unsqueeze(0).float(), "ref_x": self.ref_x}
        if self.noise_buf is not None:
            if self.live_sliding_window:
                noise_start = max(0, self.abs_frame - int(self.slide_past_frames))
                noise_end = noise_start + int(feat_25.shape[0])
            else:
                noise_start = self.abs_frame
                noise_end = self.abs_frame + target_frames
            data["noise_init"] = self.noise_buf[:, noise_start:noise_end]
        t_fm = time.perf_counter()
        if self.live_sliding_window:
            motion = self.fm.sample(
                data,
                a_cfg_scale=float(self.args.a_cfg_scale),
                nfe=int(self.args.nfe),
            )
        else:
            motion, self.stream_state = self.fm.sample(
                data,
                a_cfg_scale=float(self.args.a_cfg_scale),
                nfe=int(self.args.nfe),
                stream_state=self.stream_state,
                return_state=True,
            )
        timings["fm_ms"] = _ms(t_fm)

        motion = motion.squeeze(0).detach()
        if self.live_sliding_window:
            start = int(self.slide_past_frames)
            motion = motion[start:start + target_frames]
        else:
            motion = motion[:target_frames]
        ref_blend = float(getattr(self.args, "motion_ref_blend", 0.0) or 0.0)
        if ref_blend > 0.0:
            ref_motion = self.ref_x.detach().float()
            if ref_motion.ndim == 3:
                ref_motion = ref_motion[0]
            if ref_motion.ndim == 1:
                ref_motion = ref_motion.unsqueeze(0)
            if int(ref_motion.shape[0]) == 1:
                ref_motion = ref_motion.expand(int(motion.shape[0]), -1)
            elif int(ref_motion.shape[0]) != int(motion.shape[0]):
                ref_motion = F.interpolate(
                    ref_motion.T.unsqueeze(0),
                    size=int(motion.shape[0]),
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T
            ref_motion = ref_motion.to(device=motion.device, dtype=motion.dtype)
            blend = max(0.0, min(1.0, ref_blend))
            motion = motion.mul(1.0 - blend).add(ref_motion, alpha=blend)
        timings["helium_feat"] = helium.detach().cpu()
        timings["adapter_feat_50"] = feat_50.detach().cpu()
        timings["adapter_feat_25"] = feat_25.detach().cpu()
        timings["projected_audio"] = projected_a.squeeze(0).detach().cpu()
        timings["frames"] = int(motion.shape[0])
        timings["abs_start"] = self.abs_frame
        self.abs_frame += timings["frames"]
        return motion, timings

    def _record_session_chunk(self, pcm_chunk: np.ndarray, motion: torch.Tensor, info: dict) -> None:
        self._session_audio_parts.append(np.asarray(pcm_chunk, dtype=np.float32).copy())
        self._session_motion_parts.append(motion.detach().float().cpu().clone())
        helium_feat = info.get("helium_feat")
        if isinstance(helium_feat, torch.Tensor):
            self._session_helium_parts.append(helium_feat.float().cpu().clone())
        adapter_feat_50 = info.get("adapter_feat_50")
        if isinstance(adapter_feat_50, torch.Tensor):
            self._session_adapter_50_parts.append(adapter_feat_50.float().cpu().clone())
        adapter_feat_25 = info.get("adapter_feat_25")
        if isinstance(adapter_feat_25, torch.Tensor):
            self._session_adapter_25_parts.append(adapter_feat_25.float().cpu().clone())
        projected_audio = info.get("projected_audio")
        if isinstance(projected_audio, torch.Tensor):
            self._session_projected_audio_parts.append(projected_audio.float().cpu().clone())
        self._session_chunk_rows.append({
            "chunk": len(self._session_chunk_rows) + 1,
            "abs_start": int(info.get("abs_start", 0)),
            "frames": int(info.get("frames", int(motion.shape[0]))),
            "samples": int(len(pcm_chunk)),
            "helium_ms": float(info.get("helium_ms", 0.0)),
            "fm_ms": float(info.get("fm_ms", 0.0)),
        })

    @torch.no_grad()
    def _extract_wav2vec_raw_50hz(self, audio_24k: np.ndarray) -> torch.Tensor:
        arr = np.asarray(audio_24k, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return torch.empty((0, 768), dtype=torch.float32)
        wav = torch.from_numpy(arr).view(1, -1)
        wav16 = torchaudio.functional.resample(wav, TARGET_SR, WAV2VEC_SR).squeeze(0).contiguous().numpy()
        inputs = self.wav2vec_feature_extractor(
            wav16,
            sampling_rate=WAV2VEC_SR,
            return_tensors="pt",
            padding=True,
        )
        kwargs = {
            "input_values": inputs.input_values.to(self.device),
        }
        if getattr(inputs, "attention_mask", None) is not None:
            kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
        frontend = self.wav2vec_model.extract_projected_frontend(**kwargs)
        feat = self.wav2vec_model.encode_from_projected_frontend(
            frontend
        ).last_hidden_state.detach().float().cpu()[0].contiguous()
        return feat

    def dump_last_session(self, *, source: str = "") -> Optional[Path]:
        if not self.dump_motion or not self._session_motion_parts:
            return None
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        session_dir = self.dump_dir / "last_session"
        session_dir.mkdir(parents=True, exist_ok=True)

        motion = torch.cat(self._session_motion_parts, dim=0).contiguous()
        audio = np.concatenate(self._session_audio_parts, axis=0) if self._session_audio_parts else np.empty(0, dtype=np.float32)

        motion_path = session_dir / "full_motion.pt"
        helium_path = session_dir / "full_helium_raw.pt"
        adapter_50_path = session_dir / "full_adapter_w2v_50hz.pt"
        adapter_25_path = session_dir / "full_adapter_w2v_25fps.pt"
        wav2vec_50_path = session_dir / "full_wav2vec_50hz.pt"
        projected_audio_path = session_dir / "full_projected_audio_32.pt"
        audio_path = session_dir / "full_moshi_reply_24k.wav"
        live_tokens_path = session_dir / "live_mimi_tokens.pt"
        reply_events_path = session_dir / "reply_events.jsonl"
        reply_text_path = session_dir / "reply_text.txt"
        meta_path = session_dir / "meta.json"
        helium = None
        adapter_50 = None
        adapter_25 = None
        wav2vec_50 = None
        projected_audio = None
        live_tokens = None
        if self._session_helium_parts:
            helium = torch.cat(self._session_helium_parts, dim=0).contiguous()
        if self._session_adapter_50_parts:
            adapter_50 = torch.cat(self._session_adapter_50_parts, dim=0).contiguous()
        if self._session_adapter_25_parts:
            adapter_25 = torch.cat(self._session_adapter_25_parts, dim=0).contiguous()
        if self._session_projected_audio_parts:
            projected_audio = torch.cat(self._session_projected_audio_parts, dim=0).contiguous()
        if self._session_live_token_parts:
            live_tokens = torch.cat(self._session_live_token_parts, dim=2).contiguous()
        if audio.size > 0:
            wav2vec_50 = self._extract_wav2vec_raw_50hz(audio)

        torch.save({
            "motion": motion,
            "chunks": self._session_chunk_rows,
            "fps": float(self.fps),
            "audio_chunk_sec": float(self.audio_chunk_sec),
            "fm_chunk_frames": int(self.fm_chunk_frames),
            "audio_feat_dim": int(getattr(self.args, "audio_feat_dim", 768)),
            "audio_adapter_dim": int(getattr(self.args, "audio_adapter_dim", 512)),
            "wav2vec_sec": float(self.args.wav2vec_sec),
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
            "source": source,
        }, motion_path)
        if helium is not None:
            torch.save({
                "helium": helium,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "audio_chunk_sec": float(self.audio_chunk_sec),
                "fm_chunk_frames": int(self.fm_chunk_frames),
                "audio_feat_dim": int(getattr(self.args, "audio_feat_dim", 4096)),
                "source": source,
            }, helium_path)
        if adapter_50 is not None:
            torch.save({
                "adapter_feat_50": adapter_50,
                "chunks": self._session_chunk_rows,
                "source": source,
            }, adapter_50_path)
        if adapter_25 is not None:
            torch.save({
                "adapter_feat_25": adapter_25,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "source": source,
            }, adapter_25_path)
        if wav2vec_50 is not None:
            torch.save({
                "wav2vec_50hz": wav2vec_50,
                "chunks": self._session_chunk_rows,
                "sample_rate": int(WAV2VEC_SR),
                "source": source,
            }, wav2vec_50_path)
        if projected_audio is not None:
            torch.save({
                "projected_audio": projected_audio,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "source": source,
            }, projected_audio_path)
        if live_tokens is not None:
            torch.save({
                "live_mimi_tokens": live_tokens,
                "chunks": self._session_chunk_rows,
                "source": source,
            }, live_tokens_path)
        if audio.size > 0:
            torchaudio.save(str(audio_path), torch.from_numpy(audio).view(1, -1), TARGET_SR)
        if self._session_reply_events:
            with reply_events_path.open("w", encoding="utf-8") as f:
                for row in self._session_reply_events:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            reply_text = "".join(str(row.get("piece", "")) for row in self._session_reply_events)
            reply_text_path.write_text(reply_text, encoding="utf-8")
        meta_path.write_text(json.dumps({
            "source": source,
            "session_started_wall": float(self._session_started_wall),
            "fps": float(self.fps),
            "audio_chunk_sec": float(self.audio_chunk_sec),
            "fm_chunk_frames": int(self.fm_chunk_frames),
            "motion_frames": int(motion.shape[0]),
            "helium_frames": int(helium.shape[0]) if helium is not None else 0,
            "adapter_50_frames": int(adapter_50.shape[0]) if adapter_50 is not None else 0,
            "adapter_25_frames": int(adapter_25.shape[0]) if adapter_25 is not None else 0,
            "wav2vec_50_frames": int(wav2vec_50.shape[0]) if wav2vec_50 is not None else 0,
            "projected_audio_frames": int(projected_audio.shape[0]) if projected_audio is not None else 0,
            "audio_samples": int(audio.shape[0]),
            "audio_seconds": float(audio.shape[0] / TARGET_SR) if audio.size > 0 else 0.0,
            "reply_text_chars": int(sum(len(str(row.get("piece", ""))) for row in self._session_reply_events)),
            "reply_events": int(len(self._session_reply_events)),
            "chunks": self._session_chunk_rows,
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
        }, indent=2), encoding="utf-8")
        print(f"[liveTryHeliumFM] dumped last session -> {session_dir}", flush=True)
        return session_dir

    def _extract_motion_tensor_from_payload(self, payload, path: str) -> torch.Tensor:
        if isinstance(payload, torch.Tensor):
            motion = payload
        elif isinstance(payload, dict):
            candidates = ["motion", "motion_latents", "latents", "full_motion", "pred_motion", "x"]
            motion = None
            for key in candidates:
                value = payload.get(key)
                if isinstance(value, torch.Tensor):
                    motion = value
                    break
            if motion is None:
                tensor_items = [
                    value
                    for value in payload.values()
                    if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[-1] == int(self.args.dim_w)
                ]
                if tensor_items:
                    motion = tensor_items[0]
            if motion is None:
                raise ValueError(
                    f"No motion tensor found in blink_motion_path={path}. "
                    f"Available keys={list(payload.keys())[:20]}"
                )
        else:
            raise TypeError(f"Unsupported blink motion payload type {type(payload)} from {path}")

        motion = motion.detach().float().cpu()
        if motion.ndim == 3 and motion.shape[0] == 1:
            motion = motion[0]
        if motion.ndim != 2 or motion.shape[-1] != int(self.args.dim_w):
            raise ValueError(
                f"Blink motion must have shape (T,{int(self.args.dim_w)}) or (1,T,{int(self.args.dim_w)}); "
                f"got {tuple(motion.shape)} from {path}"
            )
        return motion.contiguous()

    def _make_eye_mask(self, height: int, width: int) -> torch.Tensor:
        yy = torch.linspace(0.0, 1.0, int(height), device=self.device, dtype=self.dtype).view(1, 1, height, 1)
        xx = torch.linspace(0.0, 1.0, int(width), device=self.device, dtype=self.dtype).view(1, 1, 1, width)
        center_y = float(getattr(self.args, "eye_center_y", 0.405))
        radius_x = max(float(getattr(self.args, "eye_radius_x", 0.145)), 1e-6)
        radius_y = max(float(getattr(self.args, "eye_radius_y", 0.070)), 1e-6)
        feather = max(float(getattr(self.args, "eye_feather", 0.10)), 1e-6)

        mask = torch.zeros(1, 1, height, width, device=self.device, dtype=self.dtype)
        for center_x in (
            float(getattr(self.args, "eye_left_x", 0.36)),
            float(getattr(self.args, "eye_right_x", 0.64)),
        ):
            dist = ((xx - center_x) / radius_x).square() + ((yy - center_y) / radius_y).square()
            ellipse = ((1.0 + feather) - dist).clamp(0.0, feather) / feather
            mask = torch.maximum(mask, ellipse)
        return mask.clamp(0.0, 1.0).contiguous()

    @torch.no_grad()
    def _init_eye_blink_composite(self) -> None:
        blink_path = str(getattr(self.args, "blink_motion_path", "") or "")
        if not blink_path:
            raise ValueError("--enable_eye_blink_composite requires --blink_motion_path")
        if not Path(blink_path).is_file():
            raise FileNotFoundError(f"blink_motion_path does not exist: {blink_path}")

        payload = torch.load(blink_path, map_location="cpu")
        blink_motion = self._extract_motion_tensor_from_payload(payload, blink_path)
        blink_maps_parts: list[list[torch.Tensor]] = []
        chunk = max(1, int(getattr(self.args, "render_sub_batch", 8)))
        for start in range(0, int(blink_motion.shape[0]), chunk):
            sub = blink_motion[start:start + chunk].to(self.device, dtype=self.dtype)
            g_sub = self.g_r.expand(int(sub.shape[0]), -1)
            ta_b = self.renderer.adapt(sub, g_sub)
            maps_b = self.renderer.latent_token_decoder(ta_b)
            if not blink_maps_parts:
                blink_maps_parts = [[] for _ in range(len(maps_b))]
            for idx, map_b in enumerate(maps_b):
                blink_maps_parts[idx].append(map_b.detach())

        self._blink_maps = tuple(torch.cat(parts, dim=0).contiguous() for parts in blink_maps_parts)
        self._eye_masks = tuple(self._make_eye_mask(m.shape[-2], m.shape[-1]) for m in self._blink_maps)
        _sync_cuda()
        shapes = [tuple(m.shape) for m in self._blink_maps]
        print(
            f"[liveTryHeliumFM][blink] cached blink maps from {blink_path}: "
            f"frames={int(blink_motion.shape[0])} shapes={shapes}",
            flush=True,
        )

    def _composite_eye_blink_maps(
        self,
        current_maps: tuple[torch.Tensor, ...] | list[torch.Tensor],
        start_frame: int,
        num_frames: int,
    ) -> tuple[torch.Tensor, ...]:
        if self._blink_maps is None or self._eye_masks is None:
            return tuple(current_maps)
        blink_len = int(self._blink_maps[0].shape[0])
        if blink_len <= 0:
            return tuple(current_maps)
        indices = (torch.arange(int(num_frames), device=self.device) + int(start_frame)) % blink_len
        composited: list[torch.Tensor] = []
        for cur, blink_all, mask in zip(current_maps, self._blink_maps, self._eye_masks):
            blink = blink_all.index_select(0, indices).to(device=cur.device, dtype=cur.dtype)
            mask = mask.to(device=cur.device, dtype=cur.dtype)
            composited.append(blink * mask + cur * (1.0 - mask))
        return tuple(composited)

    @torch.no_grad()
    def _render_motion(self, motion: torch.Tensor) -> tuple[np.ndarray, dict]:
        timings: dict = {}
        t_total = time.perf_counter()
        motion = motion.to(self.device, dtype=self.dtype)
        n = int(motion.shape[0])

        g_r_sub = self.g_r.expand(n, -1)
        m_r_sub = tuple(m.expand(n, -1, -1, -1) for m in self.m_r)
        f_r_sub = [f.expand(n, -1, -1, -1) for f in self.f_r]

        render_start = self._render_frame_cursor
        fused = getattr(self.renderer, '_fused_render', None)
        if self.eye_blink_enabled:
            fused = None
        if fused is not None:
            frames = fused(motion, g_r_sub, m_r_sub, f_r_sub)
        else:
            ta_c = self.renderer.adapt(motion, g_r_sub)
            m_c = self.renderer.latent_token_decoder(ta_c)
            if self.eye_blink_enabled:
                m_c = self._composite_eye_blink_maps(m_c, render_start, n)
            frames = self.renderer.decode(m_c, m_r_sub, f_r_sub)
        self._render_frame_cursor += n

        frames_np = frames.detach().float().clamp(0, 1).mul(255).to(torch.uint8)
        frames_np = frames_np.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        timings["total_ms"] = _ms(t_total)
        return frames_np, timings

    def render_and_encode_subbatch(
        self,
        motion_sub: torch.Tensor,
        audio_slices: list[np.ndarray],
        abs_start: int,
        text_payload: str,
        avatar_chunk_id: int,
        total_gen_ms: float,
        turn_id: int | None = None,
        speaking: bool = False,
    ) -> list[dict]:
        """Render a sub-batch of frames, JPEG-encode in parallel, return packet dicts."""
        frames_np, _render_info = self._render_motion(motion_sub)

        jpeg_futures = []
        for frame_rgb in frames_np:
            jpeg_futures.append(
                self._jpeg_pool.submit(encode_jpeg_bytes, frame_rgb, self.jpeg_quality)
            )

        packets = []
        gen_ms_i = int(round(float(total_gen_ms)))
        sr_i = int(round(float(TARGET_SR)))
        for j, fut in enumerate(jpeg_futures):
            idx = abs_start + j
            audio_slice = audio_slices[j] if j < len(audio_slices) else np.zeros(
                int(round(TARGET_SR / self.fps)), dtype=np.float32
            )
            jpeg_bytes = fut.result()
            output_audio_codec = str(
                getattr(self.args, "output_audio_codec", "pcm")
            ).lower()
            pcm_b = (
                b""
                if output_audio_codec == "opus"
                else _pcm_f32_to_i16_bytes(audio_slice)
            )
            blob = _wsbin.pack_av_frame(
                idx,
                idx + 1,
                gen_ms_i,
                sr_i,
                jpeg_bytes,
                pcm_b,
                text_payload,
                int(avatar_chunk_id),
                turn_id=int(turn_id) if turn_id is not None else 0,
                speaking=bool(speaking),
            )
            packets.append(
                {
                    "frame_number": idx,
                    "ws_kind": "bytes",
                    "data": blob,
                    "audio_pcm": np.asarray(audio_slice, dtype=np.float32).copy(),
                    "t_ready": time.perf_counter(),
                }
            )
        return packets

    def audio_slice(self, frame_idx: int) -> np.ndarray:
        if self.audio_pcm is None:
            frame_samples = int(round(TARGET_SR / self.fps))
            return np.zeros(frame_samples, dtype=np.float32)
        frame_samples = int(round(TARGET_SR / self.fps))
        start = frame_idx * frame_samples
        chunk = self.audio_pcm[start:start + frame_samples]
        if chunk.shape[0] < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - chunk.shape[0]))
        return chunk


# ---------------------------------------------------------------------------
# WebSocket streaming coroutine (file-driven mode, unchanged)
# ---------------------------------------------------------------------------

async def stream_from_file(ws: WebSocket, engine: LiveHeliumFMEngine) -> None:
    """Simulate live streaming using --audio_path, sending frames back over WS."""
    if engine.audio_pcm is None:
        await ws.send_json({"type": "error", "msg": "No --audio_path given for file-streaming mode"})
        return

    audio = engine.audio_pcm
    lookahead_chunks = max(0, int(getattr(engine.args, "file_chunk_lookahead", 0)))
    total_chunks = int(np.ceil(len(audio) / engine.audio_chunk_samples))
    start_wall = time.perf_counter()
    emitted = 0

    print(
        f"[liveTryHeliumFM] stream_from_file: {total_chunks} chunks "
        f"lookahead={lookahead_chunks}",
        flush=True,
    )

    async def _emit_motion_chunk(
        motion: torch.Tensor,
        fm_info: dict,
        chunk_label: int,
        emitted_so_far: int,
    ) -> int:
        helium_ms = float(fm_info["helium_ms"])
        fm_ms = float(fm_info["fm_ms"])
        n_frames = int(motion.shape[0])
        all_frames_np: list[np.ndarray] = []
        render_ms = 0.0
        for sb_start in range(0, n_frames, engine.render_sub_batch):
            sub = motion[sb_start:sb_start + engine.render_sub_batch].to(
                engine.device, dtype=engine.dtype
            )
            frames_np, render_info = engine._render_motion(sub)
            render_ms += float(render_info["total_ms"])
            all_frames_np.extend(frames_np)

        print(
            f"[liveTryHeliumFM][chunk#{chunk_label}] "
            f"helium={helium_ms:.0f}ms fm={fm_ms:.0f}ms "
            f"render={render_ms:.0f}ms frames={n_frames} "
            f"abs_start={fm_info['abs_start']}",
            flush=True,
        )

        for j, frame_rgb in enumerate(all_frames_np):
            idx = emitted_so_far + j
            chunk_id = idx + 1
            audio_b64 = _pcm_f32_to_i16_b64(engine.audio_slice(idx))
            jpeg_b64 = encode_jpeg_b64(frame_rgb, engine.jpeg_quality)

            await ws.send_json({
                "type": "chunk_audio",
                "chunk_id": chunk_id,
                "sample_rate": TARGET_SR,
                "pcm_s16le_b64": audio_b64,
            })
            await ws.send_json({
                "type": "chunk_frame",
                "chunk_id": chunk_id,
                "frame_idx": 0,
                "jpeg_b64": jpeg_b64,
                "moshi_text": (
                    f"Helium+FM | chunk#{chunk_label} "
                    f"helium={helium_ms:.0f}ms fm={fm_ms:.0f}ms "
                    f"render={render_ms:.0f}ms"
                ),
                "server_fps": round(float(engine.fps), 1),
                "chunks_done": chunk_label,
            })
            target_t = start_wall + (idx + 1) / engine.fps
            await asyncio.sleep(max(0.0, target_t - time.perf_counter()))
        return emitted_so_far + len(all_frames_np)

    if lookahead_chunks <= 0:
        for chunk_idx in range(total_chunks):
            pcm_chunk = audio[
                chunk_idx * engine.audio_chunk_samples:(chunk_idx + 1) * engine.audio_chunk_samples
            ]
            pcm_real = pcm_chunk.copy()
            target_frames = int(round(pcm_chunk.shape[0] * engine.fps / TARGET_SR))
            if pcm_chunk.shape[0] < engine.audio_chunk_samples:
                pcm_chunk = np.pad(pcm_chunk, (0, engine.audio_chunk_samples - pcm_chunk.shape[0]))

            motion, fm_info = engine._process_pcm_chunk(pcm_chunk, target_frames)
            engine._record_session_chunk(pcm_real, motion, fm_info)
            emitted = await _emit_motion_chunk(motion, fm_info, chunk_idx + 1, emitted)
    else:
        pending_real: list[np.ndarray] = []
        prefix_audio = np.empty(0, dtype=np.float32)
        chunk_counter = 0

        def _process_exact_pending(pcm_real: np.ndarray) -> tuple[torch.Tensor, dict]:
            nonlocal prefix_audio
            target_frames = int(round(pcm_real.shape[0] * engine.fps / TARGET_SR))
            t0 = time.perf_counter()
            helium = engine.extractor.extract_exact_chunk_from_prefix(
                prefix_audio,
                engine.abs_frame,
                target_frames,
            )
            _sync_cuda()
            fm_info: dict = {"helium_ms": _ms(t0)}
            motion, sample_info = engine._sample_motion_from_helium(helium, target_frames)
            fm_info.update(sample_info)
            return motion, fm_info

        for chunk_idx in range(total_chunks):
            pcm_real = audio[
                chunk_idx * engine.audio_chunk_samples:(chunk_idx + 1) * engine.audio_chunk_samples
            ].copy()
            prefix_audio = np.concatenate([prefix_audio, pcm_real], axis=0)
            pending_real.append(pcm_real)

            while len(pending_real) > lookahead_chunks:
                chunk_counter += 1
                oldest = pending_real.pop(0)
                motion, fm_info = _process_exact_pending(oldest)
                engine._record_session_chunk(oldest, motion, fm_info)
                emitted = await _emit_motion_chunk(motion, fm_info, chunk_counter, emitted)

        while pending_real:
            chunk_counter += 1
            oldest = pending_real.pop(0)
            motion, fm_info = _process_exact_pending(oldest)
            engine._record_session_chunk(oldest, motion, fm_info)
            emitted = await _emit_motion_chunk(motion, fm_info, chunk_counter, emitted)

    await ws.send_json({"type": "stream_end", "total_frames": emitted})
    engine.dump_last_session(source=str(engine.args.audio_path))
    print(f"[liveTryHeliumFM] stream done: {emitted} frames", flush=True)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class LiveHeliumFMOptions(BaseOptions):
    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        parser.set_defaults(wav2vec_sec=0.96)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8998)
        parser.add_argument("--html_path", default=str(ROOT / "static" / "index_v3_binary_fullscreen.html"))
        parser.add_argument("--generator_path", required=True)
        parser.add_argument("--lora_generator_path", default="", help="Optional LoRA generator checkpoint to apply on top of --generator_path")
        parser.add_argument("--lora_rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=float, default=128.0)
        parser.add_argument("--lora_dropout", type=float, default=0.05)
        parser.add_argument("--no_lora_pose_projection", action="store_true")
        parser.add_argument("--no_lora_audio_projection", action="store_true")
        parser.add_argument("--only_lora_pose_projection", action="store_true")
        parser.add_argument("--renderer_path", required=True)
        parser.add_argument("--adapter_path", required=True, help="Frontend fp32 Helium->Wav2Vec2 projected-frontend adapter checkpoint")
        parser.add_argument(
            "--adapter_type",
            choices=("frontend", "unitalk_last_layer"),
            default="frontend",
            help="Adapter architecture. UniTalk passes only layer 12 to IMTalker.",
        )
        parser.add_argument(
            "--adapter_window_mode",
            choices=("tail", "lookahead"),
            default="tail",
        )
        parser.add_argument(
            "--adapter_future_steps",
            type=int,
            default=6,
            help="12.5 Hz future-context steps retained in lookahead mode.",
        )
        parser.add_argument("--adapter_num_layers", type=int, default=6, help="Transformer layers in the frontend adapter checkpoint")
        parser.add_argument("--adapter_dropout", type=float, default=0.1, help="Dropout value used when constructing the frontend adapter")
        parser.add_argument("--stats_path", default="", help="Unused for frontend adapter mode; accepted for compatibility")
        parser.add_argument(
            "--silence_helium_path",
            default="",
            help="Optional real-silence Helium mean used to initialize the deque.",
        )
        parser.add_argument("--ref_path", required=True)
        parser.add_argument("--audio_path", default="", help="WAV to stream in fixed chunks (simulate-live mode)")
        # Moshi
        parser.add_argument("--moshi_root", default="/workspace/moshi")
        parser.add_argument("--mimi_hf_repo", default="kyutai/moshiko-pytorch-bf16")
        parser.add_argument("--moshi_weight", default="", help="Optional local PersonaPlex/Moshi LM checkpoint")
        parser.add_argument("--mimi_weight", default="", help="Optional local Mimi checkpoint")
        parser.add_argument("--tokenizer", default="", help="Optional local sentencepiece tokenizer")
        parser.add_argument("--quantize_4bit", action="store_true", help="Load PersonaPlex/Moshi LM with bnb 4-bit quantization")
        parser.add_argument("--num_codebooks", type=int, default=8, help="PersonaPlex/Moshi audio codebooks")
        parser.add_argument("--moshi_context", type=int, default=0, help="Optional PersonaPlex/Moshi KV context length")
        parser.add_argument("--voice_prompt", default="", help="PersonaPlex voice prompt filename, e.g. NATM0.pt")
        parser.add_argument("--voice_prompt_dir", default="", help="Optional PersonaPlex voice prompt directory")
        parser.add_argument("--text_prompt", default="", help="Optional PersonaPlex system text prompt")
        parser.add_argument("--moshi_reply_device", default=None, help="Optional separate CUDA device for Moshi reply generation")
        parser.add_argument("--enable_moshi_reply", action="store_true", help="Mic -> Moshi reply audio -> Helium/FM avatar")
        parser.add_argument("--moshi_cfg_coef", type=float, default=1.0)
        parser.add_argument("--direct_reply_hidden", action="store_true", default=True, help="Use Moshi generation hidden directly instead of re-encoding reply audio")
        parser.add_argument("--no_direct_reply_hidden", dest="direct_reply_hidden", action="store_false")
        parser.add_argument(
            "--disable_assistant_output_gate",
            action="store_true",
            help="Disable assistant-output RMS gating and always drive IMTalker from live PersonaPlex hidden states",
        )
        parser.add_argument(
            "--assistant_speech_rms_threshold",
            type=float,
            default=0.006,
            help="RMS threshold on PersonaPlex assistant reply audio before hidden states are allowed to drive avatar motion",
        )
        parser.add_argument(
            "--assistant_speech_hold_chunks",
            type=int,
            default=1,
            help="Keep assistant gate open for this many avatar chunks after reply audio drops below threshold",
        )
        # STT + query routing + web search (all optional; omitting every flag
        # below reproduces the plain conversational behavior of this script)
        parser.add_argument("--ref_lora_dir", default="", help="Dir containing lora/ with the <lookup>/<ref> LoRA adapter")
        parser.add_argument("--merge_ref_lora", action="store_true", help="Merge the reference LoRA into base weights instead of keeping it unmerged (QLoRA-style; default is unmerged)")
        parser.add_argument("--max_ref_tokens", type=int, default=250, help="Cap on injected <ref> block length, in tokens")
        parser.add_argument(
            "--router_threshold", type=float, default=0.40,
            help="P(needs live data) at or above which the router sends a question to web search. "
                 "Deliberately below 0.5: searching unnecessarily costs ~2s of thinking sound and is "
                 "recoverable, whereas failing to search produces a confidently wrong answer spoken "
                 "aloud. Raise it if the assistant searches too eagerly.",
        )
        parser.add_argument(
            "--router_rules", type=int, default=1,
            help="1 (default) runs the instant regex pre-pass before the model router, so obvious "
                 "cases ('what is the price today', 'thank you') are decided in microseconds with no "
                 "forward pass at all. Set 0 to route every turn through the model.",
        )
        parser.add_argument(
            "--search_max_filler_sec", type=float, default=6.0,
            help="Max seconds to wait for routing+search+compression before giving up and injecting a "
                 "generic 'no information' fallback instead. Forensic logs (conversation_logs_1/2/3) "
                 "showed real search+compression regularly taking 2.5-3.7s end to end, so the old "
                 "fixed ~2.0s cap discarded a correctly-computed answer in every observed search turn; "
                 "6.0s leaves comfortable margin above the worst latency actually measured.",
        )
        parser.add_argument(
            "--web_search_min_score", type=float, default=0.15,
            help="Discard web-search results scoring below this relevance floor before summarizing/"
                 "injecting them. Forensic logs showed clearly-irrelevant web results scoring 0.04-0.13 "
                 "(e.g. unrelated news for an unrelated query) versus clearly-relevant ones scoring "
                 "0.37+ -- this floor sits in the gap between those two clusters.",
        )
        parser.add_argument("--stt_hf_repo", default="", help="HF repo for the STT/VAD submodel, e.g. kyutai/stt-1b-en_fr-candle. Omit to disable routing/search (no transcript, no turn-detection signal).")
        parser.add_argument("--stt_pkg_dir", default="", help="Dir containing an isolated `pip install --no-deps --target <dir> moshi` install of the upstream Kyutai moshi package")
        parser.add_argument(
            "--max_input_buffer_sec", type=float, default=2.0,
            help="Maximum seconds of unprocessed microphone audio to keep. The GPU producer is "
                 "rate-limited to exactly real time by frame_q backpressure, so it can never drain "
                 "a backlog -- without this cap, any transient stall becomes a PERMANENT reply "
                 "delay for the rest of the session (conversation_log_1 showed replies landing "
                 "tens of seconds late while chunks were produced at a steady 0.96s each). When the "
                 "cap is exceeded the OLDEST audio is dropped and the drop is logged. Raise it to "
                 "trade latency for completeness, or set 0 for the old unbounded behavior.",
        )
        parser.add_argument("--vad_threshold", type=float, default=0.5)
        parser.add_argument(
            "--suppress_text_during_search", type=int, default=1,
            help="1 (default) forces the model's text stream to its own zero_text_code for the "
                 "whole time a web search is in flight, so it composes nothing until the <ref> "
                 "arrives. Muting the outgoing audio alone was never enough: the model kept "
                 "generating behind the filler and was already mid-sentence with an invented "
                 "number by the time the real one landed (conversation_log_2 turn 4 -- <ref> said "
                 "$4,085/oz, the model said $1,408.50, and it began speaking 0.03s after the "
                 "question, ~5s before the reference existed). Set 0 to restore the old behavior.",
        )
        parser.add_argument(
            "--prompt_settle_sec", type=float, default=0.0,
            help="Seconds of forced silence appended after the system prompt. The prompt is "
                 "force-fed as the assistant's OWN speech, so generation otherwise resumes from a "
                 "context ending mid-self-description and the model carries on describing itself "
                 "instead of answering the first question (conversation_log_2 turn 1). The fork "
                 "appends only ~0.5s, which does not read as end-of-turn. Set 0 to disable.",
        )
        parser.add_argument(
            "--stt_reject_foreign_script", type=int, default=1,
            help="1 (default) discards any STT transcript that is mostly non-Latin script. The "
                 "bundled STT model (kyutai/stt-1b-en_fr) can only produce English and French, so "
                 "such a transcript is decode garbage, not a language surprise -- routing on it "
                 "would answer a question nobody asked and can send nonsense to a paid search API. "
                 "The avatar is unaffected either way: PersonaPlex hears the real audio directly. "
                 "Set 0 only when deliberately running a multilingual STT checkpoint.",
        )
        parser.add_argument(
            "--stt_require_english", type=int, default=1,
            help="1 (default) also discards transcripts that are in another LANGUAGE but still "
                 "Latin script. The bundled STT model is bilingual (en/fr) and on unclear audio "
                 "hallucinates fluent Spanish/French -- conversation_log_1 contained "
                 "'Bien nino, tu eres mi How Bitcoin World.' and 'Allez, Helmut.', both of which "
                 "the script check cannot see. Detection is lexical (function words + accent rate), "
                 "so it costs microseconds and needs no extra model. Deliberately conservative: it "
                 "demands positive evidence of another language and defaults to accepting, because "
                 "rejecting a valid question means no answer at all. Set 0 to allow French/Spanish.",
        )
        parser.add_argument(
            "--stt_max_non_latin_ratio", type=float, default=0.15,
            help="Fraction of non-Latin LETTERS (digits/punctuation ignored) a transcript may "
                 "contain before --stt_reject_foreign_script discards it. Not 0: one stray decoded "
                 "symbol in an otherwise clean English sentence should not throw the turn away.",
        )
        parser.add_argument("--compressor_model", default="", help="HF model id for the small instruct model that BOTH routes queries and compresses web results, e.g. Qwen/Qwen2.5-1.5B-Instruct. Omit to disable routing and search entirely.")
        parser.add_argument("--compressor_device", default="cuda")
        parser.add_argument("--compressor_4bit", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--compressor_max_passages", type=int, default=2)
        parser.add_argument("--web_search_enabled", action="store_true")
        parser.add_argument("--web_search_api_key", default=None, help="Never hardcode this; pass via env var / secure prompt")
        parser.add_argument("--web_search_provider", default="tavily", choices=("tavily", "serper", "bing"))
        parser.add_argument("--web_search_max_results", type=int, default=3)
        parser.add_argument("--web_search_timeout", type=float, default=3.0)
        parser.add_argument(
            "--thinking_sound_path",
            default=str(ROOT.parent / "personaplex" / "ai-thinking-sound.wav"),
            help="WAV played over the assistant audio channel ONLY while an online search is "
                 "actually running (looped if shorter than the wait, stopped the instant the "
                 "grounded <ref> is injected or the search times out). Turns answered from the "
                 "model's own knowledge stay silent -- the model starts speaking within a few "
                 "hundred ms there, so a filler would only delay and mask the reply. "
                 "Set to '' to disable.",
        )
        parser.add_argument(
            "--conversation_log_dir", default="",
            help="Directory for structured per-session conversation logs (user transcripts, "
                 "router decisions, web search, compressor prompts/responses, assistant responses, "
                 "timing, errors) as both a .log and a .jsonl file. Empty disables file logging "
                 "(console logging via print() always happens regardless).",
        )
        # FM
        parser.add_argument("--audio_chunk_sec", type=float, default=0.96)
        parser.add_argument("--fm_chunk_frames", type=int, default=24, help="Must match wav2vec_sec×fps")
        parser.add_argument(
            "--enable_live_sliding_window",
            action="store_true",
            help="Run IMTalker/FM on a past/current/future feature window and emit only the current slice.",
        )
        parser.add_argument(
            "--slide_past_frames",
            type=int,
            default=10,
            help="Past 25Hz frames used by --enable_live_sliding_window.",
        )
        parser.add_argument(
            "--slide_future_frames",
            type=int,
            default=3,
            help="Future 25Hz frames used by --enable_live_sliding_window; 3 frames = 120ms.",
        )
        parser.add_argument("--skip_fm_audio_encoder", action="store_true", help="Skip FMGenerator's raw-audio Wav2Vec encoder; live PersonaPlex passes precomputed adapter features")
        parser.add_argument("--reply_hidden_steps_per_chunk", type=int, default=0, help="Raw Moshi 12.5Hz hidden steps per avatar chunk; 0 derives from fm_chunk_frames/fps")
        parser.add_argument("--prebuffer_chunks", type=int, default=3, help="Avatar chunks queued before sender starts pacing")
        parser.add_argument("--frame_q_backpressure", type=int, default=160)
        parser.add_argument(
            "--audio_q_backpressure", type=int, default=160,
            help=(
                "Pause the GPU producer once audio_q holds this many queued "
                "frames' worth of audio, mirroring --frame_q_backpressure. "
                "Without this, audio_q (maxsize=512, no active cap) can carry "
                "several seconds of already-generated-but-unsent audio as a "
                "silent standing backlog that adds directly to reply latency "
                "-- confirmed via latency_logger's audio_queue_depth/"
                "pending_audio_duration_s fields. Lower this to cut that "
                "backlog at the cost of less burst tolerance."
            ),
        )
        parser.add_argument(
            "--file_chunk_lookahead",
            type=int,
            default=0,
            help="For --audio_path mode, wait this many future chunks before emitting the oldest chunk",
        )
        parser.add_argument("--render_sub_batch", type=int, default=8)
        parser.add_argument("--jpeg_quality", type=int, default=86)
        parser.add_argument("--reply_audio_gain", type=float, default=1.0, help="Accepted for launch-script compatibility")
        parser.add_argument(
            "--motion_ref_blend",
            type=float,
            default=0.0,
            help="Blend generated motion latent toward the reference image latent to reduce head tilt.",
        )
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--buffer_ms", type=int, default=80)
        parser.add_argument(
            "--renderer_precision",
            choices=("fp32", "fp16", "bf16"),
            default="fp32",
        )
        parser.add_argument(
            "--output_audio_codec",
            choices=("pcm", "opus"),
            default="pcm",
            help="Assistant audio transport; Opus is one persistent stream per session.",
        )
        parser.add_argument("--dump_motion", action="store_true", help="Dump last session motion/audio to disk")
        parser.add_argument("--dump_dir", default=str(ROOT / "live_try_dumps"))
        # Shared noise
        parser.add_argument("--shared_noise", action="store_true")
        parser.add_argument("--noise_seed", type=int, default=1234)
        parser.add_argument("--noise_max_frames", type=int, default=5000)
        parser.add_argument(
            "--noise_temporal_corr",
            type=float,
            default=0.0,
            help="AR(1)-style temporal correlation applied to FM initial noise.",
        )
        parser.add_argument(
            "--motion_prior_noise_blend",
            type=float,
            default=0.0,
            help="Small blend from previous/generated motion into FM initial noise.",
        )
        # Precision
        parser.add_argument("--fp32", action="store_true")
        parser.add_argument("--tf32", action="store_true")
        parser.add_argument("--compile_renderer", action="store_true")
        # Eye blink motion-map compositing
        parser.add_argument("--enable_eye_blink_composite", action="store_true")
        parser.add_argument("--blink_motion_path", default="", help="Cached blink motion latent .pt file")
        parser.add_argument("--eye_left_x", type=float, default=0.36)
        parser.add_argument("--eye_right_x", type=float, default=0.64)
        parser.add_argument("--eye_center_y", type=float, default=0.405)
        parser.add_argument("--eye_radius_x", type=float, default=0.145)
        parser.add_argument("--eye_radius_y", type=float, default=0.070)
        parser.add_argument("--eye_feather", type=float, default=0.10)
        return parser


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="IMTalker Helium MotionField Deque FM liveTry")
    assets_dir = ROOT / "static" / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    started_at = time.perf_counter()
    html_path = Path(args.html_path)
    engine: LiveHeliumFMEngine | None = LiveHeliumFMEngine(args)
    moshi_engine: MoshiOnlyEngine | None = None

    def get_engine() -> LiveHeliumFMEngine:
        nonlocal engine
        if engine is None:
            engine = LiveHeliumFMEngine(args)
        return engine

    def get_moshi_engine() -> MoshiOnlyEngine:
        nonlocal moshi_engine
        if moshi_engine is None:
            moshi_engine = MoshiOnlyEngineWithHidden(
                moshi_root=args.moshi_root,
                mimi_hf_repo=args.mimi_hf_repo,
                device=getattr(args, "moshi_reply_device", None) or args.device,
                cfg_coef=float(args.moshi_cfg_coef),
                placeholder_jpeg_b64="",
                moshi_weight=getattr(args, "moshi_weight", ""),
                mimi_weight=getattr(args, "mimi_weight", ""),
                tokenizer=getattr(args, "tokenizer", ""),
                quantize_4bit=bool(getattr(args, "quantize_4bit", False)),
                num_codebooks=int(getattr(args, "num_codebooks", 8)),
                context=(int(args.moshi_context) if int(getattr(args, "moshi_context", 0)) > 0 else None),
                voice_prompt=getattr(args, "voice_prompt", ""),
                voice_prompt_dir=getattr(args, "voice_prompt_dir", ""),
                text_prompt=getattr(args, "text_prompt", ""),
                ref_lora_dir=getattr(args, "ref_lora_dir", ""),
                merge_ref_lora=bool(getattr(args, "merge_ref_lora", False)),
                max_ref_tokens=int(getattr(args, "max_ref_tokens", 250)),
                router_threshold=float(getattr(args, "router_threshold", 0.40)),
                router_use_rules=bool(int(getattr(args, "router_rules", 1))),
                stt_hf_repo=getattr(args, "stt_hf_repo", ""),
                stt_pkg_dir=getattr(args, "stt_pkg_dir", ""),
                vad_threshold=float(getattr(args, "vad_threshold", 0.5)),
                suppress_text_during_search=bool(int(getattr(args, "suppress_text_during_search", 1))),
                prompt_settle_sec=float(getattr(args, "prompt_settle_sec", 0.0)),
                stt_reject_foreign_script=bool(int(getattr(args, "stt_reject_foreign_script", 1))),
                stt_max_non_latin_ratio=float(getattr(args, "stt_max_non_latin_ratio", 0.15)),
                stt_require_english=bool(int(getattr(args, "stt_require_english", 1))),
                max_input_buffer_sec=float(getattr(args, "max_input_buffer_sec", 2.0)),
                compressor_model=getattr(args, "compressor_model", ""),
                compressor_device=getattr(args, "compressor_device", "cuda"),
                compressor_4bit=bool(getattr(args, "compressor_4bit", True)),
                compressor_max_passages=int(getattr(args, "compressor_max_passages", 2)),
                web_search_enabled=bool(getattr(args, "web_search_enabled", False)),
                web_search_api_key=getattr(args, "web_search_api_key", None),
                web_search_provider=getattr(args, "web_search_provider", "tavily"),
                web_search_max_results=int(getattr(args, "web_search_max_results", 3)),
                web_search_timeout=float(getattr(args, "web_search_timeout", 3.0)),
                web_search_min_score=float(getattr(args, "web_search_min_score", 0.15)),
                conversation_log_dir=getattr(args, "conversation_log_dir", ""),
                thinking_sound_path=getattr(args, "thinking_sound_path", ""),
                search_max_filler_sec=float(getattr(args, "search_max_filler_sec", 6.0)),
            )
        return moshi_engine

    split_sessions: dict[str, dict] = {}

    async def _get_split_media_epoch(session: dict) -> float:
        while not session["prebuffer_ready"].is_set() and not session.get("closed", False):
            await asyncio.sleep(0.01)
        async with session["media_epoch_lock"]:
            if session.get("media_epoch") is None:
                session["media_epoch"] = time.perf_counter() + 0.08
            return float(session["media_epoch"])

    if bool(getattr(args, "enable_moshi_reply", False)):
        print("[liveTryHeliumFM_ws_binary] eager-loading Moshi/PersonaPlex reply engine", flush=True)
        get_moshi_engine()

    @app.get("/")
    async def index():
        if html_path.is_file():
            return FileResponse(
                html_path,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return HTMLResponse(f"<h1>Missing HTML</h1><p>Expected: {html_path}</p>", status_code=500)

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "stage": "live_moshi_reply_helium_fm" if args.enable_moshi_reply else "live_helium_fm",
            "uptime_sec": round(time.perf_counter() - started_at, 3),
            "loaded": engine is not None,
        })

    @app.websocket("/ws/video")
    async def video_stream(ws: WebSocket):
        await ws.accept()
        session_id = str(ws.query_params.get("session_id", ""))
        session = split_sessions.get(session_id)
        if session is None:
            await ws.send_json({
                "type": "error",
                "message": "unknown or expired video session",
            })
            await ws.close()
            return

        frame_q: asyncio.Queue = session["frame_q"]
        send_start_wall = await _get_split_media_epoch(session)
        # Shared singleton (see get_moshi_engine): used only to reach the
        # per-turn latency logger from this endpoint, never to touch
        # generation state.
        reply_engine = get_moshi_engine() if bool(getattr(args, "enable_moshi_reply", False)) else None
        frames_sent = 0
        starvation_events = 0
        starve_start: float | None = None
        # Forensic fix: unlike _audio_sender, this loop had NO catch-up bias
        # at all -- it paced strictly at exactly 1/fps to send_start_wall+idx,
        # so once frame_q fell behind (any GPU/render stall), it could only
        # ever hold steady or grow, never shrink, for the rest of the session.
        # Mirrors _audio_sender's mechanism: 20% faster than real time only
        # while behind schedule (the max() below), reverting to exact 1/fps
        # once caught up -- never speeds up video that isn't backlogged.
        min_send_interval_s = max(0.001, 0.80 / float(args.fps))
        next_send_wall = send_start_wall
        print(
            f"[AJ][VIDEO] connected session={session_id[:8]} "
            f"queued={frame_q.qsize()}",
            flush=True,
        )
        try:
            while True:
                try:
                    packet = frame_q.get_nowait()
                except asyncio.QueueEmpty:
                    if starve_start is None:
                        starve_start = time.perf_counter()
                        starvation_events += 1
                    await asyncio.sleep(0.004)
                    if session.get("closed", False) and frame_q.empty():
                        break
                    continue

                if packet is None:
                    break

                if starve_start is not None:
                    gap_ms = 1000.0 * (time.perf_counter() - starve_start)
                    if gap_ms > 100:
                        print(
                            f"[AJ][VIDEO] STARVED {gap_ms:.0f}ms "
                            f"(event #{starvation_events}) "
                            f"frame_q={frame_q.qsize()} sent={frames_sent}",
                            flush=True,
                        )
                    starve_start = None

                idx = int(packet["frame_number"])
                target_t = send_start_wall + idx / float(args.fps)
                target_t = max(target_t, next_send_wall)
                sleep_s = target_t - time.perf_counter()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                next_send_wall = max(target_t, time.perf_counter()) + min_send_interval_s

                _pkt_turn = packet.get("turn_id")
                if reply_engine is not None:
                    reply_engine._mark_first_per_turn(
                        "server_avatar_frame_send", _pkt_turn,
                        note=f"frame_q depth at send={frame_q.qsize()}",
                    )
                    if _pkt_turn is not None:
                        reply_engine.conv_logger.latency.queue_state(
                            _pkt_turn, at_mark="server_avatar_frame_send",
                            frame_queue_depth=frame_q.qsize(),
                        )
                if packet.get("ws_kind") == "bytes":
                    await ws.send_bytes(packet["data"])
                else:
                    await ws.send_json(packet["msg"])
                frames_sent += 1

                late_s = time.perf_counter() - target_t
                if late_s > 0.5:
                    print(
                        f"[AJ][VIDEO] frame {idx} is {late_s*1000:.0f}ms late",
                        flush=True,
                    )
                if frames_sent % 50 == 0:
                    elapsed = time.perf_counter() - send_start_wall
                    print(
                        f"[AJ][VIDEO] sent={frames_sent} frame={idx} "
                        f"frame_q={frame_q.qsize()} elapsed={elapsed:.1f}s "
                        f"starve_events={starvation_events}",
                        flush=True,
                    )
        except (WebSocketDisconnect, RuntimeError, Exception) as exc:
            print(f"[AJ][VIDEO] closed session={session_id[:8]} {exc!r}", flush=True)
        finally:
            print(f"[AJ][VIDEO] done session={session_id[:8]} sent={frames_sent}", flush=True)

    @app.websocket("/ws/conversation")
    async def conversation(ws: WebSocket):
        await ws.accept()
        fm_engine = get_engine()
        fm_engine.reset_session()
        reply_engine = get_moshi_engine() if args.enable_moshi_reply else None
        if reply_engine is not None:
            reply_engine.reset_session()
        browser_input_sr = 48000
        opus_reader = sphn.OpusStreamReader(TARGET_SR)
        audio_packets_seen = 0

        # -- Queues for the reply pipeline --
        # mic_q: (raw_bytes, input_sr) or None to stop
        # frame_q: per-frame packet dicts or None to stop
        mic_q: queue.Queue[tuple[bytes, int] | None] | None = None
        frame_q: asyncio.Queue[dict | None] | None = None
        audio_q: asyncio.Queue[dict | None] | None = None
        gpu_thread: threading.Thread | None = None
        sender_task: asyncio.Task | None = None
        audio_sender_task: asyncio.Task | None = None
        event_loop: asyncio.AbstractEventLoop | None = None
        ws_send_lock = asyncio.Lock()
        media_epoch_lock = asyncio.Lock()
        media_epoch: float | None = None
        session_id = uuid.uuid4().hex
        prebuffer_ready = threading.Event()
        session_started = threading.Event()
        last_mic_level_log_wall = 0.0

        stream_task: asyncio.Task | None = None

        if reply_engine is not None:
            mic_q = queue.Queue(maxsize=512)
            frame_q = asyncio.Queue(maxsize=512)
            audio_q = asyncio.Queue(maxsize=512)
            event_loop = asyncio.get_running_loop()
            split_sessions[session_id] = {
                "frame_q": frame_q,
                "prebuffer_ready": prebuffer_ready,
                "media_epoch_lock": media_epoch_lock,
                "media_epoch": media_epoch,
                "closed": False,
                "created_at": time.perf_counter(),
                # Estimated (client wall clock - server wall clock), seconds.
                # 0.0 until a "clock_sync" reply comes back (see msg_type ==
                # "clock_sync" below) -- every client_timing mark is corrected
                # by this before being merged into a turn's latency block.
                # Forensic finding: without this, "server -> client" and
                # "user -> first avatar frame" showed a near-constant ~11-13s
                # gap on EVERY turn regardless of answer length or network
                # load -- the signature of a fixed clock offset (RunPod
                # container clock vs. browser clock), not real transport time.
                "clock_offset_s": 0.0,
            }

        await ws.send_json({
            "type": "server_ready",
            # t=0 for the clock-sync RTT handshake the client performs
            # immediately on receiving this message (see the "clock_sync"
            # handler below and the client's onmessage for "server_ready").
            "t_server_epoch": time.time(),
            "variant": (
                "AJ-NETWORK-ISO-CACHED-FP32"
                if os.environ.get("IMTALKER_CACHED_ENGINE", "0").strip().lower()
                in {"1", "true", "yes", "on"}
                else "AJ-NETWORK-ISO-FP32"
            ),
            "session_id": session_id,
            "video_ws_path": f"/ws/video?session_id={session_id}",
            "sample_rate": TARGET_SR,
            "output_audio_codec": str(args.output_audio_codec),
            "model_type": "moshi_reply+helium_fm+renderer" if args.enable_moshi_reply else "helium_fm+renderer",
            "tokens_per_chunk": int(args.fm_chunk_frames),
            "has_audio_file": fm_engine.audio_pcm is not None,
            "buffer_ms": int(args.buffer_ms),
            "av_transport": "binary",
            "target_fps": round(float(args.fps), 2),
        })
        print("[liveTryHeliumFM] websocket connected; sent server_ready", flush=True)

        if reply_engine is not None:
            def _gpu_producer_thread() -> None:
                """GPU thread: Moshi -> Helium -> FM -> render -> JPEG -> frame_q.

                Runs Moshi at maximum GPU speed. When real mic audio is not
                yet available, pads Moshi input with silence so it can keep
                generating reply audio without waiting for real-time mic
                arrival. This dramatically cuts first-reply latency.
                """
                assert (
                    mic_q is not None
                    and frame_q is not None
                    and audio_q is not None
                    and event_loop is not None
                )
                pending_reply_steps: list[dict] = []
                pending_reply_hidden: list[torch.Tensor] = []
                pending_reply_audio: list[np.ndarray] = []
                reply_step_history: list[dict] = []
                reply_audio_history: list[np.ndarray] = []
                reply_avatar_chunk_idx = 0
                chunk_produce_count = 0
                stopped = False
                # Pause producer when queue is this deep (qsize() is heuristic across threads).
                FRAME_Q_BACKPRESS = max(1, int(getattr(args, "frame_q_backpressure", 96)))
                # Same idea as FRAME_Q_BACKPRESS, but for audio_q: without this,
                # audio_q (maxsize=512) has no active ceiling and was observed
                # to sit at 17-40 queued packets (0.7-1.6s of audio) as a
                # standing backlog that never drains -- see --audio_q_backpressure.
                AUDIO_Q_BACKPRESS = max(1, int(getattr(args, "audio_q_backpressure", 96)))
                prebuffer_chunks = max(0, int(getattr(args, "prebuffer_chunks", PREBUFFER_CHUNKS)))
                hidden_steps_per_chunk = int(getattr(args, "reply_hidden_steps_per_chunk", 0))
                if hidden_steps_per_chunk <= 0:
                    hidden_steps_per_chunk = int(round(float(args.fm_chunk_frames) * 12.5 / float(args.fps)))
                hidden_steps_per_chunk = max(1, hidden_steps_per_chunk)
                max_moshi_steps_per_loop = 1
                was_silent = True
                assistant_gate_hold = 0
                last_motion_frame: torch.Tensor | None = None

                def _safe_put_nowait(q: "asyncio.Queue", pkt: dict, label: str) -> None:
                    """Runs ON THE EVENT LOOP THREAD via call_soon_threadsafe (see
                    below) -- never call directly from the GPU thread. put_nowait
                    is synchronous and non-blocking, so this returns immediately
                    once scheduled; it does not wait for a sender to drain
                    anything. QueueFull is only possible if several packets get
                    enqueued within one backpressure-checked GPU-thread iteration
                    (the check runs once per iteration, not once per packet) --
                    caught and logged, never raised into the event loop."""
                    try:
                        q.put_nowait(pkt)
                    except asyncio.QueueFull:
                        print(
                            f"[GPU] WARNING {label} full at put_nowait, dropped "
                            f"frame={pkt.get('frame_number')} (qsize={q.qsize()})",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[GPU] WARNING {label} put_nowait failed: {e!r}", flush=True)

                def _enqueue_frame(pkt: dict) -> None:
                    """Hand pkt to frame_q without blocking the GPU thread.

                    Forensic fix: this used to be
                    `asyncio.run_coroutine_threadsafe(frame_q.put(pkt), event_loop).result(timeout=...)`,
                    which blocks the GPU producer thread until the asyncio EVENT
                    LOOP THREAD actually gets around to running that coroutine.
                    The event loop spends most of its time inside
                    `await asyncio.sleep(...)` in the paced audio/video senders,
                    so every single frame/audio enqueue could stall the GPU
                    thread for however long the event loop happened to be
                    mid-sleep -- invisible to every CUDA-side stage timer, and a
                    direct cause of the avatar-motion backlog growing
                    unpredictably regardless of measured per-chunk GPU cost.
                    call_soon_threadsafe + put_nowait is the standard
                    non-blocking, thread-safe way to hand data to an asyncio
                    queue from another thread; it is safe here specifically
                    because the FRAME_Q_BACKPRESS/AUDIO_Q_BACKPRESS wait loops
                    above already keep the queue under its cap before this is
                    called, so put_nowait normally cannot fail.
                    """
                    _pkt_turn = pkt.get("turn_id")
                    # t_ready was already stamped in render_and_encode_subbatch
                    # at the moment render+JPEG-encode finished -- surface the
                    # first one per turn as a proper mark instead of only
                    # living inside the packet dict.
                    reply_engine._mark_first_per_turn(
                        "server_avatar_frame_ready", _pkt_turn, at=pkt.get("t_ready"),
                    )
                    event_loop.call_soon_threadsafe(_safe_put_nowait, frame_q, pkt, "frame_q")

                def _enqueue_audio(pkt: dict) -> None:
                    """Hand pkt to audio_q without blocking the GPU thread. See
                    _enqueue_frame's docstring for why this is call_soon_threadsafe
                    + put_nowait rather than a blocking run_coroutine_threadsafe."""
                    _pkt_turn = pkt.get("turn_id")
                    reply_engine._mark_first_per_turn(
                        "server_audio_queue_enter", _pkt_turn,
                        note=f"audio_q depth at enqueue={audio_q.qsize()}",
                    )
                    if _pkt_turn is not None:
                        reply_engine.conv_logger.latency.queue_state(
                            _pkt_turn, at_mark="server_audio_queue_enter",
                            audio_queue_depth=audio_q.qsize(),
                            # Each queued item is exactly one frame's worth of
                            # audio (1/fps seconds) -- derived from a real
                            # queue depth, not an invented timestamp.
                            pending_audio_duration_s=round(audio_q.qsize() / float(args.fps), 3),
                            input_backlog_s=round(reply_engine.input_backlog_sec(), 3),
                        )
                    event_loop.call_soon_threadsafe(_safe_put_nowait, audio_q, pkt, "audio_q")

                if prebuffer_chunks <= 0 and not prebuffer_ready.is_set():
                    prebuffer_ready.set()
                    print("[GPU] prebuffer=0, sender starts immediately", flush=True)

                while not stopped:
                    # --- Phase 1: drain all available mic audio (non-blocking) ---
                    while True:
                        try:
                            item = mic_q.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            stopped = True
                            break
                        raw_bytes, input_sr = item
                        if raw_bytes:
                            reply_engine.append_browser_pcm(
                                np.frombuffer(raw_bytes, dtype=np.int16), input_sr
                            )

                    if stopped:
                        break

                    # --- Phase 2: yield if outbound queue is deep (sender is the bottleneck) ---
                    while frame_q.qsize() >= FRAME_Q_BACKPRESS:
                        time.sleep(0.004)
                    while audio_q.qsize() >= AUDIO_Q_BACKPRESS:
                        time.sleep(0.004)

                    if not session_started.is_set():
                        time.sleep(0.003)
                        continue

                    if reply_engine.input_buffer.shape[0] < MIMI_FRAME_SIZE:
                        time.sleep(0.003)
                        continue

                    # --- Phase 3: run all Moshi steps and feed FM ---
                    t_recv = time.perf_counter()
                    for ev in reply_engine.process_ready_steps_limited(max_moshi_steps_per_loop):
                        pending_reply_steps.append(ev)
                        fm_engine._session_reply_events.append({
                            "step": int(ev.get("step", -1)),
                            "token": int(ev.get("token", -1)),
                            "piece": str(ev.get("piece", "")),
                            "audio_text": str(ev.get("audio_text", "")),
                            "reply_rms": float(ev.get("reply_rms", 0.0)),
                            "reply_peak": float(ev.get("reply_peak", 0.0)),
                            "input_rms": float(ev.get("input_rms", 0.0)),
                            "hidden": bool(isinstance(ev.get("helium_hidden"), torch.Tensor)),
                            "total_ms": float(ev.get("total_ms", 0.0)),
                        })
                        reply_pcm = (
                            np.frombuffer(base64.b64decode(ev["reply_i16_b64"]), dtype=np.int16)
                            .astype(np.float32) / 32768.0
                        )
                        if bool(getattr(args, "direct_reply_hidden", False)):
                            hidden = ev.get("helium_hidden")
                            if not isinstance(hidden, torch.Tensor):
                                continue
                            pending_reply_hidden.append(hidden.squeeze(0).contiguous())
                            pending_reply_audio.append(reply_pcm)
                            if len(pending_reply_hidden) < hidden_steps_per_chunk:
                                continue

                            used_hidden = pending_reply_hidden[:hidden_steps_per_chunk]
                            used_audio = pending_reply_audio[:hidden_steps_per_chunk]
                            used_steps = pending_reply_steps[:hidden_steps_per_chunk]
                            pending_reply_hidden = pending_reply_hidden[hidden_steps_per_chunk:]
                            pending_reply_audio = pending_reply_audio[hidden_steps_per_chunk:]
                            pending_reply_steps = pending_reply_steps[hidden_steps_per_chunk:]

                            if str(getattr(args, "adapter_window_mode", "tail")) == "lookahead":
                                history_size = int(fm_engine.helium_deque_size)
                                future_steps = int(
                                    getattr(args, "adapter_future_steps", 6)
                                )
                                reply_audio_history.extend(used_audio)
                                reply_step_history.extend(used_steps)
                                reply_audio_history = reply_audio_history[-history_size:]
                                reply_step_history = reply_step_history[-history_size:]

                                zero_audio = np.zeros_like(used_audio[0])
                                audio_window = (
                                    [zero_audio]
                                    * (history_size - len(reply_audio_history))
                                    + reply_audio_history
                                )
                                silent_step = {
                                    "token": 0,
                                    "reply_rms": 0.0,
                                    "total_ms": 0.0,
                                    "audio_text": "",
                                }
                                step_window = (
                                    [silent_step]
                                    * (history_size - len(reply_step_history))
                                    + reply_step_history
                                )
                                output_end = history_size - future_steps
                                output_start = (
                                    output_end - hidden_steps_per_chunk
                                )
                                used_audio = audio_window[output_start:output_end]
                                used_steps = step_window[output_start:output_end]

                            pcm_chunk = np.concatenate(used_audio, axis=0).astype(np.float32, copy=False)
                            reply_rms = float(np.sqrt(np.mean(np.square(pcm_chunk)))) if pcm_chunk.size else 0.0
                            step_rms = max(
                                (
                                    float(s.get("reply_rms", 0.0) or 0.0)
                                    for s in used_steps
                                ),
                                default=0.0,
                            )
                            speech_threshold = float(
                                getattr(args, "assistant_speech_rms_threshold", 0.006)
                            )
                            # The "thinking sound" is real (non-speech) audio played
                            # over the assistant channel -- its RMS is well above
                            # speech_threshold, so without this override the gate
                            # below would misread it as the assistant talking and
                            # animate the avatar accordingly. force_idle keeps the
                            # avatar visually idle (silence_helium_seed) while it plays.
                            force_idle_window = any(bool(s.get("force_idle")) for s in used_steps)
                            if force_idle_window:
                                assistant_active_now = False
                                assistant_gate_hold = 0
                            else:
                                assistant_active_now = max(reply_rms, step_rms) > speech_threshold
                                if assistant_active_now:
                                    assistant_gate_hold = max(
                                        0,
                                        int(getattr(args, "assistant_speech_hold_chunks", 1)),
                                    )
                                elif assistant_gate_hold > 0:
                                    assistant_gate_hold -= 1
                            assistant_active = assistant_active_now or assistant_gate_hold > 0
                            if bool(getattr(args, "disable_assistant_output_gate", False)):
                                assistant_active = True

                            previous_active = not was_silent
                            transition = assistant_active != previous_active
                            if transition:
                                print(
                                    f"[GPU] Avatar gate transition "
                                    f"{'speech' if assistant_active else 'idle'} "
                                    f"reply_rms={reply_rms:.5f} step_rms={step_rms:.5f}",
                                    flush=True,
                                )
                                if assistant_active:
                                    # Server's own detection of when it started
                                    # feeding real lip-sync motion (as opposed
                                    # to idle motion) for this turn.
                                    reply_engine._mark_first_per_turn(
                                        "avatar_first_mouth_movement",
                                        getattr(reply_engine, "_latency_turn_id", None),
                                        note=f"reply_rms={reply_rms:.5f} step_rms={step_rms:.5f}",
                                    )
                            was_silent = not assistant_active

                            if assistant_active:
                                helium_chunk = torch.cat(used_hidden, dim=0)
                            else:
                                if fm_engine.silence_helium_seed is not None:
                                    helium_chunk = fm_engine.silence_helium_seed.expand(
                                        hidden_steps_per_chunk, -1
                                    ).contiguous()
                                else:
                                    hidden_dim = int(used_hidden[0].shape[-1])
                                    helium_chunk = torch.zeros(
                                        hidden_steps_per_chunk,
                                        hidden_dim,
                                        device=fm_engine.device,
                                        dtype=torch.float32,
                                    )
                            target_frames = max(1, int(round(len(pcm_chunk) * float(args.fps) / TARGET_SR)))
                            motion, fm_info = fm_engine._sample_motion_from_helium(helium_chunk, target_frames)
                            if transition and last_motion_frame is not None:
                                blend_frames = min(5, int(motion.shape[0]))
                                alpha = torch.linspace(
                                    0.0,
                                    1.0,
                                    blend_frames + 2,
                                    device=motion.device,
                                    dtype=motion.dtype,
                                )[1:-1].unsqueeze(-1)
                                previous = last_motion_frame.to(
                                    device=motion.device,
                                    dtype=motion.dtype,
                                ).unsqueeze(0)
                                motion[:blend_frames] = (
                                    previous * (1.0 - alpha)
                                    + motion[:blend_frames] * alpha
                                )
                            last_motion_frame = motion[-1].detach().clone()
                            fm_info["assistant_active"] = bool(assistant_active)
                            fm_info["assistant_reply_rms"] = float(reply_rms)
                            fm_info["assistant_step_rms"] = float(step_rms)
                            used_codes = [
                                s["reply_codes"].to(dtype=torch.int16).contiguous()
                                for s in used_steps
                                if isinstance(s.get("reply_codes"), torch.Tensor)
                            ]
                            if used_codes:
                                fm_engine._session_live_token_parts.extend(used_codes)
                            fm_engine._record_session_chunk(pcm_chunk, motion, fm_info)
                        else:
                            result = fm_engine.feed_pcm_f32(reply_pcm)
                            if result is None:
                                continue

                            motion, fm_info, pcm_chunk = result
                            steps_per_avatar_chunk = max(1, int(round(len(pcm_chunk) / MIMI_FRAME_SIZE)))
                            used_steps = pending_reply_steps[:steps_per_avatar_chunk]
                            pending_reply_steps = pending_reply_steps[steps_per_avatar_chunk:]
                            used_codes = [
                                s["reply_codes"].to(dtype=torch.int16).contiguous()
                                for s in used_steps
                                if isinstance(s.get("reply_codes"), torch.Tensor)
                            ]
                            if used_codes:
                                fm_engine._session_live_token_parts.extend(used_codes)

                        reply_avatar_chunk_idx += 1
                        chunk_produce_count += 1

                        moshi_total_ms = sum(float(s.get("total_ms", 0.0)) for s in used_steps)

                        avatar_chunk_id = len(fm_engine._session_chunk_rows)
                        frame_audio = split_audio_into_frame_slices(pcm_chunk, args.fps)
                        _lat_turn_now = getattr(reply_engine, "_latency_turn_id", None)
                        reply_engine._mark_first_per_turn("audio_chunk_ready", _lat_turn_now)
                        n_frames = int(motion.shape[0])
                        emitted = int(fm_info["abs_start"])
                        output_step = used_steps[-1] if used_steps else ev
                        text_payload = (
                            output_step.get("audio_text")
                            or output_step.get("sampled_text")
                            or ""
                        )
                        total_gen_ms = (
                            moshi_total_ms + float(fm_info["helium_ms"]) + float(fm_info["fm_ms"])
                        )

                        for frame_offset, audio_slice in enumerate(frame_audio[:n_frames]):
                            _enqueue_audio({
                                "frame_number": emitted + frame_offset,
                                "audio_pcm": np.asarray(
                                    audio_slice,
                                    dtype=np.float32,
                                ).copy(),
                                "turn_id": _lat_turn_now,
                            })

                        t_chunk_start = time.perf_counter()

                        for sb_start in range(0, n_frames, fm_engine.render_sub_batch):
                            sb_end = min(sb_start + fm_engine.render_sub_batch, n_frames)
                            sub_motion = motion[sb_start:sb_end]
                            sub_audio = frame_audio[sb_start:sb_end]

                            packets = fm_engine.render_and_encode_subbatch(
                                sub_motion,
                                sub_audio,
                                abs_start=emitted + sb_start,
                                text_payload=text_payload,
                                avatar_chunk_id=avatar_chunk_id,
                                total_gen_ms=total_gen_ms,
                                turn_id=_lat_turn_now,
                                speaking=bool(fm_info.get("assistant_active", False)),
                            )

                            for pkt in packets:
                                pkt["turn_id"] = _lat_turn_now
                                _enqueue_frame(pkt)

                        chunk_wall_ms = _ms(t_chunk_start)
                        produce_latency_ms = _ms(t_recv)
                        q_depth = frame_q.qsize() if frame_q is not None else -1

                        # Charge the avatar pipeline's cost for this 0.96s chunk
                        # to the turn currently being timed. This runs on the
                        # same GPU thread as _step(), so reading
                        # _latency_turn_id here needs no synchronization. It is
                        # part of the user's wait for a VISIBLE answer, which is
                        # why it belongs in the latency log alongside the LM.
                        #
                        # Idle chunks are skipped for the same reason the LM
                        # accumulation above skips them: this pipeline runs
                        # continuously whether or not anyone is speaking, so
                        # only the chunks carrying the assistant's own speech
                        # are part of answering the question.
                        _lat_turn = getattr(reply_engine, "_latency_turn_id", None)
                        if _lat_turn is not None and fm_info.get("assistant_active", True):
                            _lat = reply_engine.conv_logger.latency
                            _lat.accumulate(
                                _lat_turn, "avatar_motion",
                                (float(fm_info["helium_ms"]) + float(fm_info["fm_ms"])) / 1000.0,
                            )
                            _lat.accumulate(_lat_turn, "avatar_render", chunk_wall_ms / 1000.0)

                        print(
                            f"[GPU][chunk#{reply_avatar_chunk_idx}] "
                            f"moshi={moshi_total_ms:.0f}ms "
                            f"helium={float(fm_info['helium_ms']):.0f}ms "
                            f"fm={float(fm_info['fm_ms']):.0f}ms "
                            f"render+jpeg={chunk_wall_ms:.0f}ms "
                            f"frames={n_frames} "
                            f"produce_latency={produce_latency_ms:.0f}ms "
                            f"frame_q={q_depth} "
                            f"abs={emitted}",
                            flush=True,
                        )

                        if reply_avatar_chunk_idx % 50 == 0:
                            # Microphone backlog is the single best predictor of
                            # perceived reply delay: the producer is pinned to
                            # real time by frame_q backpressure, so a backlog
                            # here is a delay that never shrinks on its own.
                            backlog_s = reply_engine.input_backlog_sec()
                            if backlog_s >= 1.0:
                                print(
                                    f"[GPU] WARNING microphone backlog {backlog_s:.1f}s -- replies "
                                    f"are running this far behind and the producer cannot catch up "
                                    f"(it is rate-limited to real time). Reduce per-chunk cost or "
                                    f"lower --max_input_buffer_sec.",
                                    flush=True,
                                )
                            status_kwargs = {
                                "chunk": reply_avatar_chunk_idx,
                                "frame_q_depth": q_depth,
                                "input_backlog_s": round(backlog_s, 2),
                                "input_dropped_s": round(
                                    reply_engine._input_dropped_samples / TARGET_SR, 2
                                ),
                                "search_enabled": getattr(reply_engine, "search_enabled", False),
                                "stt_active": reply_engine.stt_lm_gen is not None,
                                "compressor_active": reply_engine.context_compressor is not None,
                                "router_active": getattr(reply_engine, "query_router", None) is not None,
                            }
                            if torch.cuda.is_available():
                                status_kwargs["gpu_mem_allocated_gb"] = round(
                                    torch.cuda.memory_allocated() / (1 << 30), 2
                                )
                                status_kwargs["gpu_mem_reserved_gb"] = round(
                                    torch.cuda.memory_reserved() / (1 << 30), 2
                                )
                            reply_engine.conv_logger.component_status(**status_kwargs)

                        if (
                            chunk_produce_count == prebuffer_chunks
                            and not prebuffer_ready.is_set()
                        ):
                            prebuffer_ready.set()
                            print(
                                f"[GPU] prebuffer ready: {q_depth} frames queued "
                                f"after {chunk_produce_count} chunks",
                                flush=True,
                            )

                fut_done = asyncio.run_coroutine_threadsafe(frame_q.put(None), event_loop)
                try:
                    fut_done.result(timeout=30.0)
                except Exception as e:
                    print(f"[GPU] WARNING sentinel put: {e!r}", flush=True)
                fut_audio_done = asyncio.run_coroutine_threadsafe(
                    audio_q.put(None),
                    event_loop,
                )
                try:
                    fut_audio_done.result(timeout=30.0)
                except Exception as e:
                    print(f"[GPU] WARNING audio sentinel put: {e!r}", flush=True)

            async def _get_media_epoch() -> float:
                nonlocal media_epoch
                while not prebuffer_ready.is_set():
                    await asyncio.sleep(0.01)
                async with media_epoch_lock:
                    session = split_sessions.get(session_id)
                    if media_epoch is None and session is not None and session.get("media_epoch") is not None:
                        media_epoch = float(session["media_epoch"])
                    if media_epoch is None:
                        media_epoch = time.perf_counter() + 0.08
                        if session is not None:
                            session["media_epoch"] = media_epoch
                    return media_epoch

            async def _reply_sender() -> None:
                """Wait for prebuffer, then drain frame_q at 25fps."""
                assert frame_q is not None
                send_start_wall = await _get_media_epoch()
                frames_sent = 0
                starvation_events = 0
                starve_start: float | None = None
                ws_closed = False

                q_depth_at_start = frame_q.qsize()
                print(
                    f"[SENDER] prebuffer filled, starting pacing with "
                    f"{q_depth_at_start} frames queued",
                    flush=True,
                )

                while True:
                    if ws_closed:
                        break

                    try:
                        packet = frame_q.get_nowait()
                    except asyncio.QueueEmpty:
                        if send_start_wall is not None and starve_start is None:
                            starve_start = time.perf_counter()
                            starvation_events += 1
                        await asyncio.sleep(0.004)
                        continue

                    if packet is None:
                        break

                    if starve_start is not None:
                        gap_ms = 1000.0 * (time.perf_counter() - starve_start)
                        if gap_ms > 100:
                            print(
                                f"[SENDER] STARVED {gap_ms:.0f}ms "
                                f"(event #{starvation_events}) "
                                f"frame_q={frame_q.qsize()} sent={frames_sent}",
                                flush=True,
                            )
                        starve_start = None

                    idx = int(packet["frame_number"])

                    target_t = send_start_wall + idx / float(args.fps)
                    sleep_s = target_t - time.perf_counter()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

                    try:
                        async with ws_send_lock:
                            if packet.get("ws_kind") == "bytes":
                                await ws.send_bytes(packet["data"])
                            else:
                                await ws.send_json(packet["msg"])
                    except (WebSocketDisconnect, RuntimeError, Exception):
                        ws_closed = True
                        break
                    frames_sent += 1

                    late_s = time.perf_counter() - target_t
                    if late_s > 0.5:
                        print(
                            f"[SENDER] video frame {idx} is {late_s*1000:.0f}ms late",
                            flush=True,
                        )

                    if frames_sent % 50 == 0:
                        q_depth = frame_q.qsize()
                        elapsed = time.perf_counter() - send_start_wall
                        print(
                            f"[SENDER] sent={frames_sent} frame={idx} "
                            f"frame_q={q_depth} "
                            f"elapsed={elapsed:.1f}s "
                            f"starve_events={starvation_events}",
                            flush=True,
                        )

            async def _audio_sender() -> None:
                assert audio_q is not None
                output_audio_codec = str(args.output_audio_codec).lower()
                if output_audio_codec != "opus":
                    raise RuntimeError("AJ requires --output_audio_codec opus")
                opus_writer = sphn.OpusStreamWriter(TARGET_SR)
                send_start_wall = await _get_media_epoch()
                packets_sent = 0
                bytes_sent = 0
                # AG: prevent catch-up bursts after GPU/render stalls, while
                # still actually draining backlog over time (see below) rather
                # than free-running at full speed the instant it appears.
                #
                # Forensic finding (conversation_log_2/3/4): audio_q sits at a
                # standing 17-40 packets (0.7-1.6s) essentially all the time,
                # confirmed never to drain -- at the old 0.92 factor, catching
                # up a 20-packet backlog takes ~12.5 packets sent per packet of
                # backlog recovered, i.e. ~10 real seconds, far longer than a
                # typical turn lasts, so in practice it never visibly shrinks.
                # Raised to 0.80 (20% faster than real time while behind
                # schedule, silently reverting to exact real-time once caught
                # up -- see the max(target_t, next_send_wall) below) so a
                # backlog actually drains within a few seconds instead of
                # persisting indefinitely as a fixed floor under every turn's
                # "user -> first avatar frame"/"actual audio playback" latency.
                min_send_interval_s = max(0.001, 0.80 / float(args.fps))
                next_send_wall = send_start_wall

                while True:
                    packet = await audio_q.get()
                    if packet is None:
                        break
                    _pkt_turn = packet.get("turn_id")
                    reply_engine._mark_first_per_turn(
                        "server_audio_queue_release", _pkt_turn,
                        note=f"audio_q depth at release={audio_q.qsize()}",
                    )
                    if _pkt_turn is not None:
                        reply_engine.conv_logger.latency.queue_state(
                            _pkt_turn, at_mark="server_audio_queue_release",
                            audio_queue_depth=audio_q.qsize(),
                            pending_audio_duration_s=round(audio_q.qsize() / float(args.fps), 3),
                        )
                    idx = int(packet["frame_number"])
                    target_t = send_start_wall + idx / float(args.fps)
                    target_t = max(target_t, next_send_wall)
                    sleep_s = target_t - time.perf_counter()
                    if sleep_s > 0:
                        # This IS the shared real-time pacing schedule anchored
                        # to session start (send_start_wall / frame_number),
                        # not a direct wait on frame_q/avatar readiness -- see
                        # the module-level investigation notes. Named per the
                        # spec's requested vocabulary, but the note states the
                        # true mechanism found by reading this code.
                        reply_engine._mark_first_per_turn(
                            "audio_waiting_for_avatar", _pkt_turn,
                            note=(
                                f"shared real-time pacing schedule anchored to session start, "
                                f"not a direct wait on frame_q/avatar readiness; "
                                f"pacing sleep {sleep_s:.3f}s, frame_number={idx}, "
                                f"audio_q_depth={audio_q.qsize()}"
                            ),
                        )
                        await asyncio.sleep(sleep_s)
                        reply_engine._mark_first_per_turn(
                            "audio_released_by_avatar_sync", _pkt_turn,
                        )
                    next_send_wall = max(target_t, time.perf_counter()) + min_send_interval_s

                    audio_pcm = np.asarray(
                        packet["audio_pcm"],
                        dtype=np.float32,
                    ).reshape(-1)
                    opus_payload = opus_writer.append_pcm(audio_pcm)
                    if opus_payload is None and hasattr(opus_writer, "read_bytes"):
                        opus_payload = opus_writer.read_bytes()
                    if opus_payload:
                        # The REAL send-call timestamp -- taken immediately
                        # before ws.send_bytes, not at enqueue/generation time.
                        reply_engine._mark_first_per_turn("server_audio_send", _pkt_turn)
                        try:
                            async with ws_send_lock:
                                await ws.send_bytes(b"\x01" + opus_payload)
                        except (WebSocketDisconnect, RuntimeError, Exception):
                            break
                        packets_sent += 1
                        bytes_sent += len(opus_payload)

                    if packets_sent and packets_sent % 50 == 0:
                        print(
                            f"[AUDIO] sent={packets_sent} frame={idx} "
                            f"audio_q={audio_q.qsize()} bytes={bytes_sent}",
                            flush=True,
                        )

            def _gpu_producer_thread_safe() -> None:
                """Defense-in-depth guard around the whole GPU producer loop.
                Without this, an uncaught exception ANYWHERE in that loop
                (Moshi step, Helium hidden capture, FM motion sampling,
                render/encode -- none of which have their own try/except)
                kills only this one daemon thread: audio/video generation
                stops forever, the websocket stays open, and neither
                conversation log records anything, so the session just hangs
                with the user hearing nothing -- the same failure signature
                seen in conversation_logs_4 (log ends mid-turn, no error
                line, no further component_status entries), though the exact
                exception was never captured there, so this closes a
                confirmed gap rather than a proven root cause. This wrapper
                makes any future occurrence diagnosable (full traceback goes
                to the conversation log) and ends the session cleanly instead
                of hanging indefinitely."""
                try:
                    _gpu_producer_thread()
                except Exception as e:
                    tb = traceback.format_exc()
                    print(
                        f"[GPU] FATAL uncaught exception in producer thread: {e!r}\n{tb}",
                        flush=True,
                    )
                    with contextlib.suppress(Exception):
                        reply_engine.conv_logger.error("gpu_producer_thread", e, tb)
                    for q in (frame_q, audio_q):
                        if q is None:
                            continue
                        with contextlib.suppress(Exception):
                            asyncio.run_coroutine_threadsafe(q.put(None), event_loop).result(timeout=30.0)

            gpu_thread = threading.Thread(target=_gpu_producer_thread_safe, daemon=True, name="gpu-producer")
            gpu_thread.start()
            print(
                f"[AJ][AUDIO] session={session_id[:8]} "
                f"video_path=/ws/video?session_id={session_id}",
                flush=True,
            )
            audio_sender_task = asyncio.create_task(_audio_sender())

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is not None:
                    if reply_engine is None:
                        continue
                    assert mic_q is not None
                    packet_sr = int(browser_input_sr)
                    if len(data) > 0 and data[0] == 1:
                        decoded_pcm = opus_reader.append_bytes(bytes(data[1:]))
                        if decoded_pcm.shape[-1] == 0:
                            continue
                        decoded_pcm = np.asarray(decoded_pcm, dtype=np.float32).reshape(-1)
                        decoded_i16 = (
                            np.clip(decoded_pcm, -1.0, 1.0) * 32767.0
                        ).astype(np.int16)
                        data = decoded_i16.tobytes()
                        packet_sr = TARGET_SR
                    audio_packets_seen += 1
                    pcm_i16 = np.frombuffer(data, dtype=np.int16)
                    mic_rms = float(np.sqrt(np.mean((pcm_i16.astype(np.float32) / 32768.0) ** 2))) if pcm_i16.size else 0.0
                    mic_peak = float(np.max(np.abs(pcm_i16.astype(np.float32) / 32768.0))) if pcm_i16.size else 0.0
                    now_wall = time.perf_counter()
                    if audio_packets_seen <= 3 or now_wall - last_mic_level_log_wall >= 1.0:
                        voice = "VOICE" if mic_rms >= 0.02 else "quiet"
                        print(
                            f"[MIC] packet={audio_packets_seen} rms={mic_rms:.5f} "
                            f"peak={mic_peak:.3f} sr={packet_sr} {voice}",
                            flush=True,
                        )
                        last_mic_level_log_wall = now_wall
                    if audio_packets_seen <= 3:
                        print(
                            f"[liveTryHeliumFM] rx binary mic packet#{audio_packets_seen} "
                            f"bytes={len(data)} sr={packet_sr}",
                            flush=True,
                        )
                    if not session_started.is_set():
                        session_started.set()
                        print(
                            "[liveTryHeliumFM] auto-started session from first binary mic packet",
                            flush=True,
                        )
                    try:
                        mic_q.put_nowait((bytes(data), int(packet_sr)))
                    except queue.Full:
                        pass
                    continue
                text = msg.get("text")
                if text is None:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                msg_type = str(payload.get("type", "")).lower()
                print(f"[liveTryHeliumFM] rx text type={msg_type or '<empty>'}", flush=True)

                if msg_type == "start":
                    browser_input_sr = int(payload.get("sample_rate", payload.get("sampleRate", browser_input_sr)))
                    if fm_engine.audio_pcm is not None and (stream_task is None or stream_task.done()):
                        fm_engine.reset_session()
                        stream_task = asyncio.create_task(stream_from_file(ws, fm_engine))
                    if reply_engine is not None:
                        # NOT a second reset_session(). The connection handler
                        # already reset the engine at accept time, and doing it
                        # again here re-ran the entire prompt sequence -- voice
                        # prompt replay, cache restore, 159 system-prompt
                        # tokens and the settle silence -- several seconds
                        # later, by which point microphone audio was already
                        # arriving and buffered. conversation_log_3 shows the
                        # cost: two full sequences 5s apart (06:34:52.898 and
                        # 06:34:57.885), the second one wiping input_buffer
                        # mid-stream. One reset per connection is correct.
                        session_started.set()
                    print(
                        "[liveTryHeliumFM] start → "
                        + ("streaming from file" if fm_engine.audio_pcm is not None else "live Moshi reply mode"),
                        flush=True,
                    )

                elif msg_type == "chunk_audio":
                    pcm_b64 = payload.get("pcm_s16le_b64", "")
                    if not pcm_b64:
                        continue
                    pcm_bytes = base64.b64decode(pcm_b64)
                    result = fm_engine.feed_pcm(pcm_bytes)
                    if result is not None:
                        motion, fm_info, _pcm_chunk = result
                        avatar_chunk_id = len(fm_engine._session_chunk_rows)
                        n_frames = int(motion.shape[0])
                        emitted = fm_info["abs_start"]
                        for sb_start in range(0, n_frames, fm_engine.render_sub_batch):
                            sub = motion[sb_start:sb_start + fm_engine.render_sub_batch].to(
                                fm_engine.device, dtype=fm_engine.dtype
                            )
                            frames_np, _ = fm_engine._render_motion(sub)
                            for j, frame_rgb in enumerate(frames_np):
                                idx = emitted + sb_start + j
                                await ws.send_json({
                                    "type": "chunk_frame",
                                    "chunk_id": idx + 1,
                                    "frame_idx": 0,
                                    "jpeg_b64": encode_jpeg_b64(frame_rgb, fm_engine.jpeg_quality),
                                    "moshi_text": (
                                        f"live Helium+FM "
                                        f"helium={fm_info['helium_ms']:.0f}ms "
                                        f"fm={fm_info['fm_ms']:.0f}ms"
                                    ),
                                    "server_fps": round(float(args.fps), 1),
                                    "chunks_done": avatar_chunk_id,
                                })

                elif msg_type == "clock_sync":
                    # RTT-based clock-offset estimate, replacing the previous
                    # "assume server and client share a wall clock" approach
                    # that produced a near-constant ~11-13s "server -> client"
                    # artifact on every turn (a fixed clock offset, not real
                    # latency -- see the comment on split_sessions' initial
                    # clock_offset_s above). Client echoes t_server_epoch (from
                    # server_ready) back immediately along with its own
                    # Date.now() reading; standard NTP-style offset estimate:
                    # offset = t_client - (t_server_sent + rtt/2), i.e. the
                    # client clock's reading at the midpoint of the round trip
                    # minus what the server clock read when it sent the ping.
                    # Best-effort: a bad/missing value just leaves the offset
                    # at 0.0 (the old, unfixed behavior), never raises.
                    try:
                        t_server_sent = float(payload.get("t_server_epoch"))
                        t_client_recv = float(payload.get("t_client_recv_ms")) / 1000.0
                        t_server_now = time.time()
                        rtt = max(0.0, t_server_now - t_server_sent)
                        offset = t_client_recv - (t_server_sent + rtt / 2.0)
                        session = split_sessions.get(session_id)
                        if session is not None:
                            session["clock_offset_s"] = offset
                        print(
                            f"[liveTryHeliumFM] clock_sync: rtt={rtt * 1000.0:.1f}ms "
                            f"estimated_offset={offset * 1000.0:.1f}ms (client - server)",
                            flush=True,
                        )
                    except Exception:
                        pass

                elif msg_type == "client_timing":
                    # Best-effort client-reported wall-clock timestamp, merged
                    # into this turn's latency block after correcting for the
                    # clock_offset_s estimated above. See LatencyLogger.mark_wall
                    # for the remaining caveat -- this is a one-time RTT
                    # estimate, not a synchronized clock, so it removes the
                    # fixed offset but not per-message network jitter. Malformed
                    # or late messages (e.g. for a turn that already finished)
                    # must never affect the live pipeline.
                    try:
                        if reply_engine is not None:
                            cli_turn_id = payload.get("turn_id")
                            cli_event = str(payload.get("event", ""))
                            cli_t_wall_ms = payload.get("t_wall_ms")
                            if cli_turn_id is not None and cli_event and cli_t_wall_ms is not None:
                                session = split_sessions.get(session_id) or {}
                                offset = float(session.get("clock_offset_s", 0.0))
                                corrected_s = float(cli_t_wall_ms) / 1000.0 - offset
                                reply_engine.conv_logger.latency.mark_wall(
                                    int(cli_turn_id), cli_event, corrected_s,
                                    note=(
                                        f"client wall-clock, merged best-effort, "
                                        f"clock_offset={offset * 1000.0:.0f}ms corrected"
                                    ),
                                )
                    except Exception:
                        pass

                elif msg_type == "stop":
                    print("[liveTryHeliumFM] stop requested", flush=True)
                    break

        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            session = split_sessions.get(session_id)
            if session is not None:
                session["closed"] = True
            if stream_task is not None:
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            if mic_q is not None:
                with contextlib.suppress(queue.Full):
                    mic_q.put_nowait(None)
            if gpu_thread is not None:
                await asyncio.to_thread(gpu_thread.join, 30.0)
            if sender_task is not None:
                sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await sender_task
            if audio_sender_task is not None:
                audio_sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await audio_sender_task
            fm_engine.dump_last_session(source="websocket_live")
            split_sessions.pop(session_id, None)
            print("[liveTryHeliumFM] websocket closed", flush=True)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = LiveHeliumFMOptions()
    args = parser.parse()
    args.rank = args.device
    parser.print_options()

    app = build_app(args)

    import uvicorn

    print(f"[liveTryHeliumFM_ws_binary] serving {args.html_path} (binary av_transport)")
    print(f"[liveTryHeliumFM] open http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

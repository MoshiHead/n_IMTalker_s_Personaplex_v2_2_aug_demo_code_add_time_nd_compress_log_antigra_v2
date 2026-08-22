"""latency_logger.py — a dedicated, self-contained timing/latency log for the
live PersonaPlex + IMTalker pipeline.

Why a separate file at all: `conversation_logger.py` answers "what happened and
why"; this answers "where did the seconds go". Mixing the two made per-stage
timings impossible to read -- the timing lines were scattered between
transcripts, search results and narrative paragraphs, and the only end-to-end
number available was `turn_done`, which stops at the moment the grounding text
was injected, i.e. BEFORE the model had said a single word.

Two files per server run, both append-only:

  1. `latency_<session>.log`  -- human-readable. A live line per stage as it
     completes, then one self-contained block per turn: a timeline of when each
     component finished (as an offset from the moment the user stopped
     speaking), the stage durations sorted slowest-first, the token counts
     (including how many tokens the compressor produced), and the headline
     question->first word / question->finished answer latencies.
  2. `latency_<session>.jsonl` -- one JSON object per finished turn, with every
     stage, mark, counter and headline metric as a flat field, for plotting or
     regression-checking a change against a previous run.

Everything here is best-effort: a logging failure must never break the live
pipeline, so every public method swallows its own exceptions. Thread-safety:
`start_turn`/`mark`/`stage`/`count` are called from both the GPU thread and the
background routing/search thread, so all state mutation and file I/O happens
under one lock.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

# Human-readable names for every stage/mark the pipeline reports. Anything not
# listed still logs fine -- it just shows up under its raw name.
STAGE_LABELS: dict[str, str] = {
    "stt_decode": "speech-to-text decode (audio -> transcript)",
    "transcript_check": "transcript sanity check (script/language)",
    "rule_route": "quick rule check: is a search needed?",
    "router_model": "router model decision: is a search needed?",
    "decision": "search/no-search decision (total)",
    "web_search": "online search (network round trip)",
    "search_filter": "score, filter and sort the search results",
    "compression": "compress results into one spoken sentence (LLM)",
    "compression_fallback": "extractive fallback summary (no LLM)",
    "ref_encode": "tokenize + trim the grounding text",
    "lookup_inject": "inject the <lookup> 'please wait' note",
    "ref_inject": "inject the grounding <ref> into the live context",
    "thinking_sound": "thinking sound played to cover the wait",
    "gen_lm": "answer generation: Moshi LM forward passes (GPU)",
    "gen_audio_decode": "answer generation: Mimi audio decode (GPU)",
    "gen_audio_encode": "answer generation: Mimi input encode (GPU)",
    "avatar_motion": "avatar motion synthesis (Helium adapter + flow matching)",
    "avatar_render": "avatar render + JPEG encode",
    "stt_forward_per_chunk": "STT/VAD forward pass (runs every chunk, idle or not)",
}

MARK_LABELS: dict[str, str] = {
    "speech_end": "user stopped speaking (end of question)",
    "turn_start": "turn opened, routing begins",
    "routing_start": "routing decision begins",
    "routing_end": "routing decision made (alias of decision_made)",
    "decision_made": "search/no-search decision made",
    "search_done": "online search returned",
    "grounding_ready": "grounding sentence ready",
    "ref_injected": "grounding handed to the model",
    "lookup_injected": "'please wait' note handed to the model",
    "first_word": "assistant's FIRST spoken word (what the user feels)",
    "model_first_audio": "model decoded its first audio sample for this turn",
    "audio_chunk_ready": "first PCM chunk sliced into per-frame audio",
    "server_audio_queue_enter": "first audio packet handed to audio_q",
    "server_audio_queue_release": "first audio packet dequeued by the sender",
    "audio_waiting_for_avatar": "sender begins its pacing sleep before send",
    "audio_released_by_avatar_sync": "sender's pacing sleep completed",
    "server_audio_send": "first audio packet actually sent on the wire",
    "server_avatar_frame_ready": "first avatar frame finished render+encode",
    "server_avatar_frame_send": "first avatar frame actually sent on the wire",
    "avatar_first_mouth_movement": "server started feeding real lip-sync motion",
    "answer_complete": "assistant's LAST word of the answer",
    # -- client-reported marks: wall-clock, merged best-effort (see mark_wall) --
    "client_audio_receive": "[client wall-clock, merged best-effort] browser received an audio packet",
    "audio_buffer_start": "[client wall-clock, merged best-effort] PCM handed to the output worklet",
    "actual_audio_playback_start": "[client wall-clock, merged best-effort] worklet emitted real audio samples",
    "client_avatar_frame_receive": "[client wall-clock, merged best-effort] browser received an AV frame",
    "avatar_first_frame_rendered": "[client wall-clock, merged best-effort] first frame drawn to canvas",
    "avatar_first_mouth_movement_client": "[client wall-clock, merged best-effort] first drawn frame flagged speaking",
    "first_meaningful_avatar_speech_frame": "[client wall-clock, merged best-effort] first VISIBLE+AUDIBLE speaking frame",
}

# Headline metrics, in the order they are printed. (field, label)
HEADLINES: list[tuple[str, str]] = [
    ("question_to_decision_s", "question -> search decision"),
    ("question_to_search_done_s", "question -> search results in"),
    ("question_to_grounding_s", "question -> grounding sentence ready"),
    ("question_to_ref_injected_s", "question -> grounding given to model"),
    ("question_to_first_word_s", "question -> FIRST spoken word"),
    ("question_to_answer_complete_s", "question -> answer fully spoken"),
    ("answer_speaking_s", "duration of the spoken answer"),
    ("user_to_routing_decision_s", "user -> routing decision"),
    ("user_to_model_first_audio_s", "user -> model first audio"),
    ("model_output_to_audio_ready_s", "model output -> audio ready"),
    ("audio_ready_to_server_send_s", "audio ready -> server send"),
    ("server_queue_wait_s", "server queue wait (enter -> send)"),
    ("server_to_client_s", "server -> client (send -> client receive)"),
    ("client_receive_to_buffer_s", "client receive -> audio buffer"),
    ("buffer_to_playback_s", "audio buffer -> actual playback"),
    ("user_to_actual_playback_s", "user -> actual audio playback"),
    ("user_to_first_avatar_frame_s", "user -> first avatar frame"),
    ("user_to_first_mouth_movement_s", "user -> first avatar mouth movement"),
    ("user_to_first_meaningful_avatar_speech_s", "user -> FIRST_MEANINGFUL_AVATAR_SPEECH_FRAME"),
    ("user_to_complete_answer_s", "user -> complete answer"),
]

_MAX_RETAINED_TURNS = 16


class TurnLatency:
    """Mutable timing record for one question -> answer cycle."""

    def __init__(self, turn_id: Any, t0: float, transcript: str = "") -> None:
        self.turn_id = turn_id
        self.transcript = transcript
        self.t0 = float(t0)
        self.wall_start = time.time()
        self.stages: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self.counters: dict[str, Any] = {}
        self.notes: dict[str, str] = {}
        self.outcome = ""
        self.response = ""
        self.closed = False

    def offset(self, now: float | None = None) -> float:
        return max(0.0, (now if now is not None else time.perf_counter()) - self.t0)


class LatencyLogger:
    def __init__(self, log_dir: str = "", session_id: str = "") -> None:
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = str(log_dir or "")
        self.path: str | None = None
        self.jsonl_path: str | None = None
        self._lock = threading.RLock()
        self._turns: dict[Any, TurnLatency] = {}
        self._order: list[Any] = []
        # Session-wide roll-up, printed with every turn block so a drift over a
        # long conversation is visible without post-processing the file.
        self._n_turns = 0
        self._sum_first_word = 0.0
        self._n_first_word = 0
        self._sum_total = 0.0
        self._n_total = 0
        self._n_searched = 0

        if not self.log_dir:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            self.path = os.path.join(self.log_dir, f"latency_{self.session_id}.log")
            self.jsonl_path = os.path.join(self.log_dir, f"latency_{self.session_id}.jsonl")
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(
                    "=" * 88 + "\n"
                    f"Latency / timing log -- session {self.session_id}\n"
                    f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    "Every timing in this file is measured from the moment the user STOPPED\n"
                    "speaking (t=0, when the question is complete) to the moment the assistant\n"
                    "finished speaking its answer. Lines marked 'stage' are measured durations of\n"
                    "one component; lines marked 'mark' are the instants a component finished.\n"
                    "Nothing here is ever overwritten -- each turn is appended below the last.\n"
                    + "=" * 88 + "\n\n"
                )
            print(
                f"[latency_logger] logging per-stage timings to {self.path} and {self.jsonl_path}",
                flush=True,
            )
        except Exception as e:  # pragma: no cover - logging must never break the pipeline
            print(f"[latency_logger] disabled, could not open log files: {e!r}", flush=True)
            self.path = None
            self.jsonl_path = None

    # -- low-level ---------------------------------------------------------

    @staticmethod
    def _now_ts() -> str:
        now = time.time()
        return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"

    def _write(self, text: str) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[latency_logger] failed to write log line: {e!r}", flush=True)

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not self.jsonl_path:
            return
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[latency_logger] failed to write jsonl record: {e!r}", flush=True)

    # -- turn lifecycle ----------------------------------------------------

    def start_turn(self, turn_id: Any, t0: float | None = None, transcript: str = "") -> None:
        """Open a record for `turn_id`. `t0` should be the perf_counter reading
        taken when the user's utterance ENDED (VAD fired) -- not when this call
        happens -- so that every offset below is measured from the instant the
        question was complete, which is what the user experiences as "I asked,
        then I waited"."""
        try:
            with self._lock:
                rec = TurnLatency(turn_id, t0 if t0 is not None else time.perf_counter(), transcript)
                self._turns[turn_id] = rec
                self._order.append(turn_id)
                # Keep only the most recent handful: a turn is finalized when
                # the NEXT one starts, so at most two are ever live, but a turn
                # that never produced a response would otherwise leak.
                while len(self._order) > _MAX_RETAINED_TURNS:
                    self._turns.pop(self._order.pop(0), None)
            self._write(
                f"[{self._now_ts()}] TURN {turn_id} | +{0.0:6.3f}s | START    | "
                f'question: "{transcript}"\n'
            )
        except Exception:
            pass

    def get(self, turn_id: Any) -> TurnLatency | None:
        with self._lock:
            return self._turns.get(turn_id)

    def stage(self, turn_id: Any, name: str, seconds: float, note: str = "") -> None:
        """Record a measured duration for one component and echo it live."""
        try:
            secs = float(seconds)
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None:
                    return
                rec.stages[name] = rec.stages.get(name, 0.0) + secs
                if note:
                    rec.notes[name] = note
                offset = rec.offset()
            suffix = f" | {note}" if note else ""
            self._write(
                f"[{self._now_ts()}] TURN {turn_id} | +{offset:6.3f}s | stage    | "
                f"{name:<20} {secs * 1000.0:8.1f} ms{suffix}\n"
            )
        except Exception:
            pass

    def accumulate(self, turn_id: Any, name: str, seconds: float) -> None:
        """Add to a stage total WITHOUT writing a line. For per-80ms-chunk costs
        (LM forward, audio decode, avatar render) that would otherwise emit
        thousands of lines per turn; they are reported once in the turn block."""
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None:
                    return
                rec.stages[name] = rec.stages.get(name, 0.0) + float(seconds)
        except Exception:
            pass

    def mark(self, turn_id: Any, name: str, note: str = "", at: float | None = None) -> None:
        """Record the instant a component finished, as an offset from t=0."""
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None:
                    return
                offset = rec.offset(at)
                rec.marks[name] = offset
                if note:
                    rec.notes[name] = note
            suffix = f" | {note}" if note else ""
            self._write(
                f"[{self._now_ts()}] TURN {turn_id} | +{offset:6.3f}s | mark     | {name}{suffix}\n"
            )
        except Exception:
            pass

    def mark_wall(self, turn_id: Any, name: str, wall_time_s: float, note: str = "") -> None:
        """Like `mark()`, but for a timestamp reported by the CLIENT as a
        wall-clock (time.time()-style) value rather than this process's
        perf_counter. `mark()` cannot be reused for this: its `at` parameter
        is perf_counter-space and subtracts `rec.t0` (also perf_counter); a
        client's Date.now() lives on a different clock entirely, so it is
        converted here via `rec.wall_start` (captured with time.time() at
        start_turn) instead.

        This is a best-effort cross-clock merge, NOT a synchronized
        measurement: it assumes the server and client share a wall clock
        (same host, or NTP-synced on the same LAN). Callers should label
        anything derived from it accordingly (see MARK_LABELS's "[client
        wall-clock, merged best-effort]" prefixes) -- it is not equally
        precise as a server-side perf_counter mark."""
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None:
                    return
                offset = max(0.0, float(wall_time_s) - rec.wall_start)
                rec.marks[name] = offset
                if note:
                    rec.notes[name] = note
            suffix = f" | {note}" if note else ""
            self._write(
                f"[{self._now_ts()}] TURN {turn_id} | +{offset:6.3f}s | mark(cli) | {name}{suffix}\n"
            )
        except Exception:
            pass

    def queue_state(self, turn_id: Any, at_mark: str = "", **kwargs: Any) -> None:
        """Snapshot queue/backlog depths at a named moment (e.g. the same
        instant a mark() was just recorded). Stored into rec.counters with an
        `{at_mark}__` prefix so repeated snapshots across the turn (enter,
        release, send, ...) don't clobber each other -- mirrors count()'s
        write pattern otherwise."""
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None:
                    return
                prefix = f"{at_mark}__" if at_mark else ""
                clean = {f"{prefix}{k}": v for k, v in kwargs.items() if v is not None}
                rec.counters.update(clean)
            if clean:
                body = " ".join(f"{k}={v}" for k, v in clean.items())
                self._write(
                    f"[{self._now_ts()}] TURN {turn_id} | {'':>7} | queue    | {body}\n"
                )
        except Exception:
            pass

    def count(self, turn_id: Any, **counters: Any) -> None:
        """Record counts (tokens, results, characters) for this turn."""
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None:
                    return
                clean = {k: v for k, v in counters.items() if v is not None}
                rec.counters.update(clean)
            if clean:
                body = " ".join(f"{k}={v}" for k, v in clean.items())
                self._write(
                    f"[{self._now_ts()}] TURN {turn_id} | {'':>7} | count    | {body}\n"
                )
        except Exception:
            pass

    def note_transcript(self, turn_id: Any, transcript: str) -> None:
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is not None and transcript:
                    rec.transcript = transcript
        except Exception:
            pass

    # -- finalization ------------------------------------------------------

    def finish_turn(self, turn_id: Any, response: str = "", outcome: str = "") -> None:
        """Close the turn and write its self-contained block + JSONL record.
        Safe to call twice (the second call is a no-op) and safe to call for a
        turn that was never opened."""
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None or rec.closed:
                    return
                rec.closed = True
                rec.response = response or ""
                rec.outcome = outcome or rec.outcome
                metrics = self._headline_metrics(rec)
                self._n_turns += 1
                if "question_to_first_word_s" in metrics:
                    self._sum_first_word += metrics["question_to_first_word_s"]
                    self._n_first_word += 1
                if "question_to_answer_complete_s" in metrics:
                    self._sum_total += metrics["question_to_answer_complete_s"]
                    self._n_total += 1
                if rec.stages.get("web_search"):
                    self._n_searched += 1
                block = self._render_block(rec, metrics)
                full_breakdown = self.render_full_breakdown(rec, metrics)
                search_breakdown = self._render_search_breakdown(rec)
                record = self._render_record(rec, metrics)
                headline = self._render_console(rec, metrics)
            self._write(block)
            self._write(full_breakdown)
            self._write(search_breakdown)
            self._write_jsonl(record)
            print(headline, flush=True)
            print(full_breakdown, flush=True)
        except Exception as e:
            print(f"[latency_logger] failed to finalize turn {turn_id}: {e!r}", flush=True)

    @staticmethod
    def _headline_metrics(rec: TurnLatency) -> dict[str, float]:
        m: dict[str, float] = {}
        pairs = [
            ("question_to_decision_s", "decision_made"),
            ("question_to_search_done_s", "search_done"),
            ("question_to_grounding_s", "grounding_ready"),
            ("question_to_ref_injected_s", "ref_injected"),
            ("question_to_first_word_s", "first_word"),
            ("question_to_answer_complete_s", "answer_complete"),
        ]
        for field, mark in pairs:
            if mark in rec.marks:
                m[field] = round(rec.marks[mark], 3)
        if "first_word" in rec.marks and "answer_complete" in rec.marks:
            m["answer_speaking_s"] = round(
                max(0.0, rec.marks["answer_complete"] - rec.marks["first_word"]), 3
            )

        def _delta(field: str, a: str, b: str) -> None:
            """m[field] = marks[b] - marks[a], only if both marks exist."""
            if a in rec.marks and b in rec.marks:
                m[field] = round(max(0.0, rec.marks[b] - rec.marks[a]), 3)

        mk = rec.marks
        # user -> routing decision
        if "routing_end" in mk:
            m["user_to_routing_decision_s"] = round(mk["routing_end"], 3)
        elif "decision_made" in mk:
            m["user_to_routing_decision_s"] = round(mk["decision_made"], 3)
        # user -> model first audio
        if "model_first_audio" in mk:
            m["user_to_model_first_audio_s"] = round(mk["model_first_audio"], 3)
        # model output -> audio ready (first decode -> first chunk sliced)
        _delta("model_output_to_audio_ready_s", "model_first_audio", "audio_chunk_ready")
        # audio ready -> server send (queue enter -> actual send)
        _delta("audio_ready_to_server_send_s", "server_audio_queue_enter", "server_audio_send")
        # server queue wait: enqueue -> dequeue by the sender
        _delta("server_queue_wait_s", "server_audio_queue_enter", "server_audio_queue_release")
        # server -> client: real send -> client's reported receive
        _delta("server_to_client_s", "server_audio_send", "client_audio_receive")
        # client receive -> handed to the audio buffer/worklet
        _delta("client_receive_to_buffer_s", "client_audio_receive", "audio_buffer_start")
        # audio buffer -> actual playback (worklet emits real samples)
        _delta("buffer_to_playback_s", "audio_buffer_start", "actual_audio_playback_start")
        # user -> actual audio playback (headline: what the user actually hears)
        if "actual_audio_playback_start" in mk:
            m["user_to_actual_playback_s"] = round(mk["actual_audio_playback_start"], 3)
        # user -> first avatar frame / mouth movement / meaningful speech frame
        if "avatar_first_frame_rendered" in mk:
            m["user_to_first_avatar_frame_s"] = round(mk["avatar_first_frame_rendered"], 3)
        if "avatar_first_mouth_movement" in mk:
            m["user_to_first_mouth_movement_s"] = round(mk["avatar_first_mouth_movement"], 3)
        elif "avatar_first_mouth_movement_client" in mk:
            m["user_to_first_mouth_movement_s"] = round(mk["avatar_first_mouth_movement_client"], 3)
        if "first_meaningful_avatar_speech_frame" in mk:
            m["user_to_first_meaningful_avatar_speech_s"] = round(
                mk["first_meaningful_avatar_speech_frame"], 3
            )
        # user -> complete answer (alias of the existing headline, spec's name)
        if "answer_complete" in mk:
            m["user_to_complete_answer_s"] = round(mk["answer_complete"], 3)
        # synchronization: sender's pacing sleep, if it ever waited
        _delta("audio_pacing_wait_s", "audio_waiting_for_avatar", "audio_released_by_avatar_sync")
        return m

    def _render_block(self, rec: TurnLatency, metrics: dict[str, float]) -> str:
        w = 88
        lines = ["", "=" * w]
        # ASCII only in this block: it is read with `tail`/`cat` on hosts whose
        # console is not guaranteed to be UTF-8, and a mojibake dash in the
        # header is the first thing a reader sees.
        lines.append(f"TURN {rec.turn_id} -- where the time went")
        lines.append(f'Question : "{rec.transcript}"')
        if rec.response:
            lines.append(f'Answer   : "{rec.response}"')
        if rec.outcome:
            lines.append(f"Outcome  : {rec.outcome}")
        lines.append(
            f"Clock    : question ended at "
            f"{time.strftime('%H:%M:%S', time.localtime(rec.wall_start))}"
            f".{int(rec.wall_start % 1 * 1000):03d}"
        )
        lines.append("-" * w)

        lines.append("Timeline (seconds after the user stopped speaking):")
        if rec.marks:
            for name, offset in sorted(rec.marks.items(), key=lambda kv: kv[1]):
                label = MARK_LABELS.get(name, name)
                lines.append(f"  +{offset:7.3f}s  {label}")
        else:
            lines.append("  (no timeline marks were recorded for this turn)")

        lines.append("")
        lines.append("Component times (slowest first):")
        total_measured = sum(rec.stages.values())
        span = metrics.get(
            "question_to_answer_complete_s", metrics.get("question_to_first_word_s", 0.0)
        )
        if rec.stages:
            for name, secs in sorted(rec.stages.items(), key=lambda kv: kv[1], reverse=True):
                label = STAGE_LABELS.get(name, name)
                pct = f"{secs / span * 100.0:5.1f}%" if span > 0 else "    -"
                note = rec.notes.get(name, "")
                note_s = f"   [{note}]" if note else ""
                lines.append(f"  {secs:7.3f}s  {pct} of total   {label}{note_s}")
            lines.append(f"  {total_measured:7.3f}s          all measured components added up")
        else:
            lines.append("  (no component timings were recorded for this turn)")

        if rec.counters:
            lines.append("")
            lines.append("Sizes / token counts:")
            for name, value in rec.counters.items():
                lines.append(f"  {name:<32} {value}")

        lines.append("")
        lines.append("Headline latencies:")
        for field, label in HEADLINES:
            if field in metrics:
                lines.append(f"  {label:<40} {metrics[field]:7.3f}s")
        if "question_to_first_word_s" not in metrics:
            lines.append(
                "  (the assistant never produced audible speech for this turn -- "
                "no first-word latency exists)"
            )

        avg_first = self._sum_first_word / self._n_first_word if self._n_first_word else 0.0
        avg_total = self._sum_total / self._n_total if self._n_total else 0.0
        lines.append("")
        lines.append(
            f"Session so far: {self._n_turns} turn(s), {self._n_searched} with an online search, "
            f"average question->first word {avg_first:.2f}s, "
            f"average question->finished answer {avg_total:.2f}s"
        )
        lines.append("=" * w)
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_record(rec: TurnLatency, metrics: dict[str, float]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": "turn_latency",
            "ts": time.time(),
            "turn": rec.turn_id,
            "wall_start": rec.wall_start,
            "transcript": rec.transcript,
            "response": rec.response,
            "outcome": rec.outcome,
        }
        record.update(metrics)
        for name, secs in rec.stages.items():
            record[f"stage_{name}_s"] = round(float(secs), 4)
        for name, offset in rec.marks.items():
            record[f"at_{name}_s"] = round(float(offset), 4)
        record.update(rec.counters)
        record["stages_total_s"] = round(sum(rec.stages.values()), 4)
        return record

    @staticmethod
    def _render_console(rec: TurnLatency, metrics: dict[str, float]) -> str:
        first = metrics.get("question_to_first_word_s")
        total = metrics.get("question_to_answer_complete_s")
        parts = [
            f"first_word={first:.2f}s" if first is not None else "first_word=n/a",
            f"answer_done={total:.2f}s" if total is not None else "answer_done=n/a",
        ]
        for name in ("router_model", "web_search", "compression", "compression_fallback"):
            if rec.stages.get(name):
                parts.append(f"{name}={rec.stages[name]:.2f}s")
        if "compressor_output_tokens" in rec.counters:
            parts.append(f"compressor_out_tok={rec.counters['compressor_output_tokens']}")
        return f"[LATENCY] turn {rec.turn_id}: " + " ".join(parts)

    def render_full_breakdown(self, rec: TurnLatency, metrics: dict[str, float]) -> str:
        """Second, more granular block: clock-time timeline of the send/receive
        pipeline plus a SYNCHRONIZATION and BOTTLENECK section. Appended after
        the existing `_render_block` output (never replaces it)."""
        try:
            w = 88
            lines = ["", "-" * w, "FULL SEND/RECEIVE BREAKDOWN (turn %s)" % rec.turn_id, "-" * w]

            def clock(offset: float) -> str:
                t = rec.wall_start + offset
                return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}"

            pipeline_marks = [
                "speech_end", "routing_start", "routing_end",
                "model_first_audio", "audio_chunk_ready",
                "server_audio_queue_enter", "audio_waiting_for_avatar",
                "audio_released_by_avatar_sync", "server_audio_queue_release",
                "server_audio_send", "client_audio_receive", "audio_buffer_start",
                "actual_audio_playback_start",
                "server_avatar_frame_ready", "server_avatar_frame_send",
                "client_avatar_frame_receive", "avatar_first_frame_rendered",
                "avatar_first_mouth_movement", "avatar_first_mouth_movement_client",
                "first_meaningful_avatar_speech_frame", "answer_complete",
            ]
            any_shown = False
            for name in pipeline_marks:
                if name in rec.marks:
                    any_shown = True
                    label = MARK_LABELS.get(name, name)
                    lines.append(f"  {clock(rec.marks[name])}  +{rec.marks[name]:7.3f}s  {label}")
            if not any_shown:
                lines.append("  (no send/receive pipeline marks were recorded for this turn)")

            lines.append("")
            lines.append("Headline deltas:")
            any_headline = False
            for field, label in HEADLINES:
                if field in metrics:
                    any_headline = True
                    lines.append(f"  {label:<48} {metrics[field]:7.3f}s")
            if not any_headline:
                lines.append("  (no headline deltas could be computed -- required marks missing)")

            lines.append("")
            lines.append("SYNCHRONIZATION:")
            if "audio_pacing_wait_s" in metrics:
                note = rec.notes.get("audio_waiting_for_avatar", "")
                lines.append(
                    f"  audio sender pacing wait               {metrics['audio_pacing_wait_s']:7.3f}s"
                    + (f"   [{note}]" if note else "")
                )
            else:
                lines.append("  (no pacing wait was recorded -- audio was sent with no sleep)")

            lines.append("")
            lines.append("BOTTLENECK:")
            candidates = [
                (field, label) for field, label in HEADLINES
                if field in metrics and field != "user_to_complete_answer_s"
            ]
            # Only compare independent (non-cumulative) deltas so the biggest
            # single GAP is reported, not the biggest running total.
            gap_fields = {
                "user_to_routing_decision_s", "model_output_to_audio_ready_s",
                "audio_ready_to_server_send_s", "server_to_client_s",
                "client_receive_to_buffer_s", "buffer_to_playback_s",
            }
            gaps = [(f, l) for f, l in candidates if f in gap_fields]
            if gaps:
                worst_field, worst_label = max(gaps, key=lambda fl: metrics[fl[0]])
                lines.append(f"  largest gap: {worst_label} = {metrics[worst_field]:.3f}s")
            else:
                lines.append("  (not enough marks were recorded to identify a bottleneck)")
            lines.append("-" * w)
            lines.append("")
            return "\n".join(lines) + "\n"
        except Exception:
            return ""

    def _render_search_breakdown(self, rec: TurnLatency) -> str:
        """Search-pipeline breakdown, only emitted when this turn actually
        searched (mirrors the existing web_search-stage check at
        finish_turn/_headline_metrics). No-search turns get a one-line
        summary instead, so the two pipelines are visibly distinguished."""
        try:
            w = 88
            lines = ["-" * w]
            if rec.stages.get("web_search"):
                lines.append(f"SEARCH-PIPELINE BREAKDOWN (turn {rec.turn_id})")
                if "routing_end" in rec.marks:
                    lines.append(f"  search decision made      +{rec.marks['routing_end']:7.3f}s")
                elif "decision_made" in rec.marks:
                    lines.append(f"  search decision made      +{rec.marks['decision_made']:7.3f}s")
                lines.append(f"  web search duration         {rec.stages.get('web_search', 0.0):7.3f}s")
                comp = rec.stages.get("compression") or rec.stages.get("compression_fallback")
                if comp:
                    kind = "compression" if rec.stages.get("compression") else "compression_fallback (extractive)"
                    lines.append(f"  {kind:<26} {comp:7.3f}s")
                if "ref_injected" in rec.marks:
                    lines.append(f"  grounding injected into model +{rec.marks['ref_injected']:7.3f}s")
                if "model_first_audio" in rec.marks:
                    lines.append(f"  model generation resumed  +{rec.marks['model_first_audio']:7.3f}s")
            else:
                lines.append(
                    f"NO-SEARCH turn {rec.turn_id}: user -> routing -> direct model answer -> "
                    f"audio/avatar -> playback"
                )
            lines.append("-" * w)
            lines.append("")
            return "\n".join(lines) + "\n"
        except Exception:
            return ""

    # -- session close -----------------------------------------------------

    def session_summary(self) -> None:
        """Optional end-of-session roll-up. Never called automatically -- the
        server is normally killed rather than shut down cleanly -- but useful
        from a notebook or a signal handler."""
        try:
            avg_first = self._sum_first_word / self._n_first_word if self._n_first_word else 0.0
            avg_total = self._sum_total / self._n_total if self._n_total else 0.0
            self._write(
                "\n" + "=" * 88 + "\n"
                f"SESSION SUMMARY {self.session_id}\n"
                f"  turns logged                     {self._n_turns}\n"
                f"  turns with an online search      {self._n_searched}\n"
                f"  average question->first word     {avg_first:.3f}s\n"
                f"  average question->finished answer{avg_total:.3f}s\n"
                + "=" * 88 + "\n\n"
            )
        except Exception:
            pass

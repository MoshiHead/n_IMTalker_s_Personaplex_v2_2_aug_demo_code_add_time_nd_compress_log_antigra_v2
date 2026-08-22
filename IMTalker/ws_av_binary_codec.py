"""Binary WebSocket payload for combined audio+JPEG (Moshi reply path).

Layout (little-endian, 48-byte header + payloads):
  bytes 0:3   magic ASCII 'AV01'
  bytes 4:7   version u32 (must be 2)
  bytes 8:11  frame_number u32 (absolute frame index)
  bytes 12:15 chunk_id u32 (1-based, same as JSON chunk_id)
  bytes 16:19 gen_ms u32 (rounded chunk gen time)
  bytes 20:23 sample_rate u32 (e.g. 24000)
  bytes 24:27 jpeg_len u32
  bytes 28:31 pcm_s16le_len u32 (byte length of PCM int16 mono)
  bytes 32:35 text_utf8_len u32
  bytes 36:39 chunks_done u32 (avatar chunk index, same as JSON chunks_done)
  bytes 40:43 turn_id u32 (latency-log turn id active when this frame was
              rendered; 0 if no turn was open -- logging/correlation only,
              never used for playback/animation decisions)
  bytes 44:47 flags u32 (bit 0 = speaking: reflects the assistant_active
              lip-sync gate at render time; other bits reserved, always 0)
  (48-byte header total)
Then: jpeg_bytes | pcm_s16le_bytes | text_utf8_bytes

Version 1 -> 2 change: the old version-1 header ended at byte 44 with a
reserved/always-zero u32 (bytes 40:43); that slot is now turn_id, and flags
is a newly appended field (bytes 44:47). Every field kept from version 1
keeps its exact previous meaning and byte offset.

Mirror this in static/index_v3_binary_fullscreen_aj_nodrop.html (handleBinaryAv).
"""
from __future__ import annotations

import struct

MAGIC = b"AV01"
VERSION = 2
_HEADER = struct.Struct("<4s11I")  # 4 + 44 = 48 bytes


def pack_av_frame(
    frame_number: int,
    chunk_id_ui: int,
    gen_ms: int,
    sample_rate: int,
    jpeg: bytes,
    pcm_s16le: bytes,
    text: str,
    chunks_done: int,
    turn_id: int = 0,
    speaking: bool = False,
) -> bytes:
    tb = text.encode("utf-8", errors="replace")[:16000]
    if len(jpeg) > 12_000_000:
        raise ValueError("jpeg payload too large")
    flags = 1 if speaking else 0
    hdr = _HEADER.pack(
        MAGIC,
        VERSION & 0xFFFFFFFF,
        int(frame_number) & 0xFFFFFFFF,
        int(chunk_id_ui) & 0xFFFFFFFF,
        int(gen_ms) & 0xFFFFFFFF,
        int(sample_rate) & 0xFFFFFFFF,
        len(jpeg) & 0xFFFFFFFF,
        len(pcm_s16le) & 0xFFFFFFFF,
        len(tb) & 0xFFFFFFFF,
        int(chunks_done) & 0xFFFFFFFF,
        int(turn_id) & 0xFFFFFFFF,
        int(flags) & 0xFFFFFFFF,
    )
    return hdr + jpeg + pcm_s16le + tb

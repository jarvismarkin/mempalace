"""
media_miner.py — File photos and voice notes into the palace.

Each image becomes one drawer and each audio file one drawer per 30-second
segment. The vector comes from the media itself (embeddinggemma-2 puts images
and audio in the same space as text), so a plain text search like "whiteboard
sketch of the rotation schedule" can find a photo. The drawer's document is a
short description with the file path, so the agent can open the original.

Voice notes are transcribed locally (mlx-whisper) and the transcript is what
gets embedded; raw-audio vectors are only a fallback for clips with no speech,
because they recall spoken content much less reliably.
"""

import base64
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime

from .embedding import embed_media
from .palace import get_collection, file_already_mined

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif"}
AUDIO_EXTS = {".ogg", ".oga", ".opus", ".mp3", ".m4a", ".wav", ".aac"}
SEGMENT_S = 30
OVERLAP_S = 3
MAX_IMAGE_PX = 1024
WHISPER = os.path.expanduser("~/.local/bin/mlx_whisper")
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


def media_kind(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _image_b64(path: str, tmp: str) -> str:
    """JPEG, longest side <= 1024px (sips also handles HEIC and WebP)."""
    out = os.path.join(tmp, "img.jpg")
    subprocess.run(
        ["sips", "-s", "format", "jpeg", "-Z", str(MAX_IMAGE_PX), path, "--out", out],
        check=True, capture_output=True,
    )
    return _b64(out)


def _audio_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True,
    )
    return float(r.stdout.strip() or 0)


def _audio_segments(path: str, tmp: str):
    """Yield (start_s, end_s, b64 16 kHz mono WAV) for overlapping 30 s windows."""
    duration = _audio_duration(path)
    start = 0.0
    while True:
        end = min(start + SEGMENT_S, duration)
        out = os.path.join(tmp, "seg.wav")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(end - start),
             "-i", path, "-ac", "1", "-ar", "16000", "-f", "wav", out],
            check=True, capture_output=True,
        )
        yield start, end, duration, _b64(out)
        if end >= duration:
            break
        start = end - OVERLAP_S


def _transcribe(path: str, tmp: str):
    """Whisper segments [{start, end, text}] via the local mlx-whisper CLI, or []."""
    try:
        subprocess.run(
            [WHISPER, "--model", WHISPER_MODEL, "-f", "json", "-o", tmp, "--verbose", "False", path],
            check=True, capture_output=True, timeout=900,
        )
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(tmp, stem + ".json")) as f:
            return [s for s in json.load(f)["segments"] if s["text"].strip()]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return []


def _transcript_windows(segments):
    """Group whisper segments into ~30 s windows: [(start, end, text)]."""
    windows, cur = [], []
    for seg in segments:
        if cur and seg["end"] - cur[0]["start"] > SEGMENT_S:
            windows.append(cur)
            cur = []
        cur.append(seg)
    if cur:
        windows.append(cur)
    return [(w[0]["start"], w[-1]["end"], " ".join(s["text"].strip() for s in w)) for w in windows]


def _clock(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def mine_media_file(collection, path: str, wing: str, channel: str = "inbound") -> int:
    """File one media file. Returns drawers added (0 if skipped or already mined)."""
    kind = media_kind(path)
    if not kind or file_already_mined(collection, path):
        return 0
    received = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    room = "photos" if kind == "image" else "voice_notes"
    base = {
        "wing": wing,
        "room": room,
        "source_file": path,
        "added_by": "media_miner",
        "filed_at": datetime.now().isoformat(),
        "ingest_mode": "media",
        "media_type": kind,
        "received_at": received,
    }
    ids, docs, metas, embs = [], [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "image":
            embs.append(embed_media("image", _image_b64(path, tmp)))
            docs.append(f"[photo] {channel} image received {received}\nfile: {path}")
            metas.append({**base, "chunk_index": 0})
        else:
            duration = _audio_duration(path)
            windows = _transcript_windows(_transcribe(path, tmp))
            if windows:
                # Speech: embed the transcript text (the collection's embedding
                # function does it) — far better recall than raw-audio vectors.
                embs = None
                for i, (start, end, text) in enumerate(windows):
                    span = f"{_clock(start)}–{_clock(end)} of {_clock(duration)}"
                    docs.append(f"[voice note] {channel}, received {received}, {span}\n{text}\nfile: {path}")
                    metas.append({**base, "chunk_index": i, "offset_s": round(start, 1),
                                  "duration_s": round(duration, 1), "transcribed": True})
            else:
                # No speech recognised: fall back to raw-audio embeddings.
                for i, (start, end, duration, data) in enumerate(_audio_segments(path, tmp)):
                    embs.append(embed_media("audio", data))
                    span = f"{_clock(start)}–{_clock(end)} of {_clock(duration)}"
                    docs.append(f"[audio] {channel}, received {received}, {span} (no speech recognised)\nfile: {path}")
                    metas.append({**base, "chunk_index": i, "offset_s": round(start, 1),
                                  "duration_s": round(duration, 1), "transcribed": False})
    for meta in metas:
        key = f"{path}#{meta['chunk_index']}".encode()
        ids.append(f"drawer_{wing}_{room}_{hashlib.sha256(key).hexdigest()[:24]}")
    if embs is None:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
    else:
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    return len(ids)


def mine_media(palace_path: str, paths: list, wing: str, channel: str = "inbound") -> dict:
    """File a list of media files into one palace. Errors are per-file, not fatal."""
    collection = get_collection(palace_path)
    stats = {"files": 0, "drawers": 0, "skipped": 0, "errors": []}
    for path in paths:
        try:
            n = mine_media_file(collection, path, wing, channel)
        except Exception as e:
            stats["errors"].append(f"{os.path.basename(path)}: {e}")
            continue
        if n:
            stats["files"] += 1
            stats["drawers"] += n
        else:
            stats["skipped"] += 1
    return stats

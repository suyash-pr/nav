"""Indexing the `.bin` + `.meta.json` groups the recorder writes for RGB and depth.

A group's sidecar indexes every frame in it by `{timestamp_us, seq, length}`, which is what lets
a reader slice the concatenated packets back apart. That index is also the join target for RGB:
the sensor frontier records arrival for `image_raw`, but only `image_encoded` is recorded, so a
cycle's RGB frame is found by looking its capture timestamp up here.

That join rests on an assumption karter flags as UNVERIFIED ON HARDWARE (commit e926a211f): that
DepthAI propagates a frame's timestamp unchanged through the encoder. The frontier's `seq` is the
*raw*-stream seq and need not match the encoded stream's. `resolve_frames` therefore reports
which strategy actually worked rather than assuming the best case.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BIN_EXT = ".bin"
META_EXT = ".meta.json"

JOIN_EXACT = "exact"
JOIN_CONSTANT_OFFSET = "constant_offset"
JOIN_APPROXIMATE = "approximate"

# How far a capture timestamp may sit from a sidecar frame before a match is not credible. One
# frame period at 10 Hz; a residual larger than this means the streams are not the same clock.
DEFAULT_MATCH_EPSILON_US = 50_000


@dataclass(frozen=True)
class GroupIndex:
    """Every frame of every group in one channel directory, plus the groups' decode parameters."""

    frames: pd.DataFrame  # timestamp_us, seq, bin_path, frame_index, length
    meta: dict[Path, dict]

    def __len__(self) -> int:
        return len(self.frames)

    def group_meta(self, bin_path: Path) -> dict:
        return self.meta[bin_path]

    def extradata(self, bin_path: Path) -> bytes | None:
        """FFV1 keeps its parameter set out of band; a single packet will not decode without it."""
        raw = self.meta[bin_path].get("extradata")
        return base64.b64decode(raw) if raw else None

    def quantisation(self, bin_path: Path) -> tuple[float, int] | None:
        """(horizon_m, step_mm) for a depth group, or None if it was recorded unquantised."""
        meta = self.meta[bin_path]
        if "step_mm" not in meta:
            return None
        return float(meta["horizon_m"]), int(meta["step_mm"])


def index_groups(channel_dir: Path) -> GroupIndex:
    """Index every `.bin` + `.meta.json` pair under `channel_dir`, ordered by timestamp.

    A sidecar naming a missing `.bin`, or a `.bin` with no sidecar, is skipped rather than
    raised: one damaged group should cost its own frames and nothing else. A `.bin` without a
    sidecar is unreadable by construction -- the index is what slices it apart.
    """
    channel_dir = Path(channel_dir)
    rows: list[dict] = []
    meta_by_bin: dict[Path, dict] = {}

    if not channel_dir.is_dir():
        return GroupIndex(frames=_empty_frames(), meta={})

    for meta_path in sorted(channel_dir.glob(f"*{META_EXT}")):
        bin_path = meta_path.with_suffix("").with_suffix(BIN_EXT)
        if not bin_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        meta_by_bin[bin_path] = meta
        for frame_index, entry in enumerate(meta.get("frames", [])):
            rows.append(
                {
                    "timestamp_us": int(entry["timestamp_us"]),
                    "seq": int(entry.get("seq", -1)),
                    "bin_path": bin_path,
                    "frame_index": frame_index,
                    "length": int(entry["length"]),
                }
            )

    if not rows:
        return GroupIndex(frames=_empty_frames(), meta=meta_by_bin)

    frames = pd.DataFrame(rows).sort_values("timestamp_us", kind="stable").reset_index(drop=True)
    return GroupIndex(frames=frames, meta=meta_by_bin)


def _empty_frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_us": pd.Series(dtype="int64"),
            "seq": pd.Series(dtype="int64"),
            "bin_path": pd.Series(dtype="object"),
            "frame_index": pd.Series(dtype="int64"),
            "length": pd.Series(dtype="int64"),
        }
    )


@dataclass(frozen=True)
class FrameResolution:
    """Which indexed frame each requested capture timestamp resolved to, and how."""

    row: np.ndarray  # int64 index into GroupIndex.frames, -1 where unresolved
    strategy: str
    offset_us: int
    residual_us: np.ndarray
    match_rate: float


def resolve_frames(
    index: GroupIndex,
    capture_us: np.ndarray,
    *,
    epsilon_us: int = DEFAULT_MATCH_EPSILON_US,
) -> FrameResolution:
    """Resolve capture timestamps onto indexed frames, reporting which strategy was needed.

    Tried in order, so a session's provenance is never implicit:

    `exact`            every requested timestamp is present in the index verbatim -- the
                       assumption karter could not verify on hardware does hold here.
    `constant_offset`  the encoder shifted timestamps by a stable amount; the modal delta is
                       learned and applied, and matching then proceeds exactly.
    `approximate`      neither holds. Falls back to a *causal* as-of -- the newest frame at or
                       before the capture time, never the nearest, because a nearest match can
                       select a frame recorded after the decision it supposedly informed. That
                       is the exact defect this rewrite exists to remove.
    """
    capture_us = np.asarray(capture_us, dtype=np.int64)
    wanted = capture_us >= 0
    row = np.full(len(capture_us), -1, dtype=np.int64)
    residual = np.zeros(len(capture_us), dtype=np.int64)

    if len(index) == 0 or not wanted.any():
        return FrameResolution(row=row, strategy=JOIN_APPROXIMATE, offset_us=0, residual_us=residual, match_rate=0.0)

    indexed_us = index.frames["timestamp_us"].to_numpy(dtype=np.int64)

    exact_row = _match_exact(indexed_us, capture_us, wanted)
    if _rate(exact_row, wanted) == 1.0:
        row[wanted] = exact_row[wanted]
        return FrameResolution(row=row, strategy=JOIN_EXACT, offset_us=0, residual_us=residual, match_rate=1.0)

    offset = _modal_offset(indexed_us, capture_us, wanted, epsilon_us)
    if offset is not None:
        shifted_row = _match_exact(indexed_us, capture_us + offset, wanted)
        rate = _rate(shifted_row, wanted)
        if rate == 1.0:
            row[wanted] = shifted_row[wanted]
            residual[wanted] = offset
            return FrameResolution(
                row=row, strategy=JOIN_CONSTANT_OFFSET, offset_us=offset, residual_us=residual, match_rate=rate
            )

    asof_row = _match_asof(indexed_us, capture_us, wanted, epsilon_us)
    matched = asof_row >= 0
    row[matched] = asof_row[matched]
    residual[matched] = capture_us[matched] - indexed_us[asof_row[matched]]
    return FrameResolution(
        row=row,
        strategy=JOIN_APPROXIMATE,
        offset_us=0,
        residual_us=residual,
        match_rate=_rate(asof_row, wanted),
    )


def _match_exact(indexed_us: np.ndarray, capture_us: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    out = np.full(len(capture_us), -1, dtype=np.int64)
    position = np.searchsorted(indexed_us, capture_us)
    inside = wanted & (position < len(indexed_us))
    hit = np.zeros(len(capture_us), dtype=bool)
    hit[inside] = indexed_us[position[inside]] == capture_us[inside]
    out[hit] = position[hit]
    return out


def _match_asof(indexed_us: np.ndarray, capture_us: np.ndarray, wanted: np.ndarray, epsilon_us: int) -> np.ndarray:
    """Newest indexed frame at or before each capture time, within epsilon. Never forward."""
    out = np.full(len(capture_us), -1, dtype=np.int64)
    position = np.searchsorted(indexed_us, capture_us, side="right") - 1
    ok = wanted & (position >= 0)
    within = np.zeros(len(capture_us), dtype=bool)
    within[ok] = (capture_us[ok] - indexed_us[position[ok]]) <= epsilon_us
    out[within] = position[within]
    return out


def _modal_offset(indexed_us: np.ndarray, capture_us: np.ndarray, wanted: np.ndarray, epsilon_us: int) -> int | None:
    """The most common (indexed - capture) delta among near matches, if one dominates."""
    probe = capture_us[wanted]
    if len(probe) == 0:
        return None
    position = np.clip(np.searchsorted(indexed_us, probe), 0, len(indexed_us) - 1)
    deltas = indexed_us[position] - probe
    deltas = deltas[np.abs(deltas) <= epsilon_us]
    if len(deltas) == 0:
        return None
    values, counts = np.unique(deltas, return_counts=True)
    best = int(values[np.argmax(counts)])
    return best if best != 0 else None


def _rate(row: np.ndarray, wanted: np.ndarray) -> float:
    if not wanted.any():
        return 0.0
    return float((row[wanted] >= 0).mean())

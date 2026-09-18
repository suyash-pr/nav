#!/usr/bin/env python3
"""Standalone evidence that the recorder's H.265 RGB groups can carry undecodable frames.

Finding: an `image_encoded` group's `.meta.json` sidecar can list N frames whose byte lengths
correctly account for every byte in the `.bin` (no truncation, no I/O loss), while several of
those "recorded" frames are permanently undecodable, because one or more *earlier* frames in the
same group's linear GOP were dropped before ever reaching the recorder -- a gap in `seq`, visible
straight from the sidecar with no decode required.

Because the group uses a single keyframe with every P-frame chained to its predecessor, one such
gap poisons the reference chain for every following frame until the group's next keyframe (there
is often just one keyframe per group). So one dropped frame does not cost one frame of data -- it
can cost the rest of the group.

This script proves the claim two independent ways per group:
  1. `seq` gaps -- pure metadata, no decode: frames the sensor produced that never reached the
     recorder at all.
  2. Actual decode failure -- packets are fed to a real HEVC decoder with an explicit PTS tag per
     sidecar position (`packet.pts = position`), so a decoded frame can be matched back to the
     exact sidecar entry it came from regardless of the decoder's internal reorder/output delay.
     A sidecar position that never appears in the decoder's output, even after a full flush, is a
     confirmed, permanent decode failure -- not a false negative from positional misalignment.

Dependencies: only `av` (PyAV) and the standard library -- no other part of this repository.

Usage:
    python scripts/diagnose_recorder_frame_loss.py /data/0918/data/camera_oakd_front_rgb/image_encoded
    python scripts/diagnose_recorder_frame_loss.py path/to/one_group.bin --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import av


@dataclass
class GroupReport:
    bin_path: Path
    n_declared: int
    n_bytes_accounted: int
    n_bytes_actual: int
    seq_gaps: list[tuple[int, int]]  # (last_seen_seq, next_seen_seq) around each gap
    undecodable_positions: list[int]  # sidecar indices that never decoded, even after flush

    @property
    def n_frames_lost_to_seq_gaps(self) -> int:
        return sum(next_seq - last_seq - 1 for last_seq, next_seq in self.seq_gaps)

    @property
    def bytes_accounted_for(self) -> bool:
        return self.n_bytes_accounted == self.n_bytes_actual


def find_seq_gaps(seqs: list[int]) -> list[tuple[int, int]]:
    """Consecutive-integer gaps in a group's recorded `seq` numbers.

    Each gap is direct evidence that the sensor produced a frame the recorder never received --
    read straight off the sidecar, independent of anything downstream being able to decode.
    """
    gaps = []
    for previous, current in zip(seqs, seqs[1:]):
        if current - previous > 1:
            gaps.append((previous, current))
    return gaps


def decode_with_position_tracking(blob: bytes, frames_meta: list[dict], *, width: int, height: int) -> list[int]:
    """Feed every packet with an explicit PTS equal to its sidecar position, and return the
    sidecar positions that were *never* recovered from the decoder, even after a full flush.

    Tagging PTS this way is what makes "never recovered" a reliable signal: without it, a
    decoder's internal reorder/output delay could make an early position merely *appear* late,
    which would look identical to genuine loss under naive positional (list-index) comparison.
    With PTS tagging, a position is only reported missing if the decoder truly never emitted it.
    """
    decoder = av.CodecContext.create("hevc", "r")
    decoder.width = width
    decoder.height = height
    decoder.pix_fmt = "rgb24"

    recovered_positions: set[int] = set()
    offset = 0
    for position, entry in enumerate(frames_meta):
        length = int(entry["length"])
        chunk = blob[offset : offset + length]
        offset += length
        packet = av.Packet(chunk)
        packet.pts = position
        packet.dts = position
        try:
            for frame in decoder.decode(packet):
                recovered_positions.add(frame.pts)
        except av.error.FFmpegError:
            pass  # a hard decode error on this packet; its position simply won't be recovered

    try:
        for frame in decoder.decode(None):  # flush anything the decoder was still holding
            recovered_positions.add(frame.pts)
    except av.error.FFmpegError:
        pass

    return sorted(set(range(len(frames_meta))) - recovered_positions)


def diagnose_group(meta_path: Path, *, verbose: bool = False) -> GroupReport | None:
    bin_path = meta_path.with_suffix("").with_suffix(".bin")
    if not bin_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    frames_meta = meta["frames"]
    if not frames_meta:
        return None

    blob = bin_path.read_bytes()
    n_bytes_accounted = sum(int(entry["length"]) for entry in frames_meta)

    seqs = [int(entry["seq"]) for entry in frames_meta]
    gaps = find_seq_gaps(seqs)

    undecodable = decode_with_position_tracking(blob, frames_meta, width=int(meta["width"]), height=int(meta["height"]))

    report = GroupReport(
        bin_path=bin_path,
        n_declared=len(frames_meta),
        n_bytes_accounted=n_bytes_accounted,
        n_bytes_actual=len(blob),
        seq_gaps=gaps,
        undecodable_positions=undecodable,
    )

    if verbose:
        print(f"  {bin_path.name}")
        print(f"    sidecar declares {report.n_declared} frames, {n_bytes_accounted} bytes "
              f"({'accounted for' if report.bytes_accounted_for else 'MISMATCH vs ' + str(len(blob)) + ' actual bytes -- truncated file'})")
        if gaps:
            print(f"    seq gaps (recorder never received these): {gaps} "
                  f"-> {report.n_frames_lost_to_seq_gaps} frame(s)")
        if undecodable:
            print(f"    sidecar positions that never decoded, even after flush: {undecodable}")
            if gaps:
                first_gap_seq = gaps[0][0] + 1
                cascades = [p for p in undecodable if seqs[p] >= first_gap_seq]
                print(f"    -> {len(cascades)} of those sit at or after the first seq gap "
                      f"(seq {first_gap_seq}): consistent with one dropped reference frame "
                      f"poisoning every P-frame chained to it")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="a single .bin/.meta.json group, or a directory of them")
    parser.add_argument("--verbose", action="store_true", help="print a line per group, not just the summary")
    args = parser.parse_args()

    if args.path.is_dir():
        meta_paths = sorted(args.path.glob("*.meta.json"))
    elif args.path.suffix == ".json" or args.path.name.endswith(".meta.json"):
        meta_paths = [args.path]
    else:
        meta_paths = [args.path.with_suffix("").with_suffix(".meta.json")]

    if not meta_paths:
        print(f"no .meta.json files found under {args.path}", file=sys.stderr)
        sys.exit(1)

    reports = [r for p in meta_paths if (r := diagnose_group(p, verbose=args.verbose)) is not None]
    if not reports:
        print("no readable groups found", file=sys.stderr)
        sys.exit(1)

    n_groups = len(reports)
    n_groups_with_gaps = sum(1 for r in reports if r.seq_gaps)
    n_groups_with_decode_loss = sum(1 for r in reports if r.undecodable_positions)
    n_declared = sum(r.n_declared for r in reports)
    n_lost_to_gaps = sum(r.n_frames_lost_to_seq_gaps for r in reports)
    n_undecodable = sum(len(r.undecodable_positions) for r in reports)
    n_truncated = sum(1 for r in reports if not r.bytes_accounted_for)

    print()
    print(f"{n_groups} group(s) examined ({args.path})")
    print(f"  {n_declared} frames declared across all sidecars")
    print(f"  {n_groups_with_gaps}/{n_groups} groups have a seq gap "
          f"(a frame the sensor produced but the recorder never received)")
    print(f"  {n_lost_to_gaps} frame(s) lost that way, never even reaching the .bin")
    print(f"  {n_groups_with_decode_loss}/{n_groups} groups have at least one sidecar-declared "
          f"frame that never decodes")
    print(f"  {n_undecodable} frame(s) present as bytes in a .bin but permanently undecodable "
          f"(reference chain broken by an earlier gap, no recovery until the group's next keyframe)")
    if n_truncated:
        print(f"  {n_truncated} group(s) additionally truncated: sidecar-declared byte length "
              f"exceeds the actual .bin size")
    if n_declared:
        print(f"  net: {(n_lost_to_gaps + n_undecodable) / (n_declared + n_lost_to_gaps) * 100:.1f}% "
              f"of frames the sensor actually produced are unusable")


if __name__ == "__main__":
    main()

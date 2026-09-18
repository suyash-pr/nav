"""Locating topics and their rotated chunk files inside a recorded karter session.

Mirrors karter/src/python/utils/recording_loader.py (pinned at 208a6fce9). A session is a
directory holding `data/<full/topic/path>/` plus `logs/`; recordings predating the `data/`
subdirectory put topic dirs at the session root instead.
"""

from __future__ import annotations

import re
from pathlib import Path

DATA_SUBDIR = "data"

PLANNING_CYCLE_TOPIC = "navigation/planning_cycle"
SENSOR_FRONTIER_TOPIC = "navigation/sensor_frontier"
NAVIGATION_COMMAND_TOPIC = "navigation/command"
EXECUTED_COMMAND_TOPIC = "ll_executed_command"
VELOCITY_ESTIMATE_TOPIC = "robot/velocity_estimate"

# Trailing 20-digit chunk start timestamp_us before ".csv" -- see karter's
# RotatingCSVWriter._open_writer. 20 digits so it never overflows int64 and always sorts.
# A name that doesn't match has unknown coverage and is never excluded by a window query.
_CHUNK_TIMESTAMP_RE = re.compile(r"_(\d{20})\.csv$")


def data_root(session_dir: Path) -> Path:
    """The session's `data/` root, falling back to the session dir for older recordings."""
    root = session_dir / DATA_SUBDIR
    return root if root.is_dir() else session_dir


def channel_dir(session_dir: Path, topic: str) -> Path:
    """The directory holding `topic`'s files.

    Falls back to a directory named after the topic's first path segment only, which is where
    every non-image dtype was written before karter's layout change. Returns the current-layout
    path when neither exists, so a caller's error message names the expected location.
    """
    root = data_root(session_dir)
    current = root / topic
    if current.is_dir():
        return current
    legacy = root / topic.split("/")[0]
    return legacy if legacy.is_dir() else current


def topic_slug(topic: str) -> str:
    """The filename stem a topic's chunks are named with (`navigation/command` -> `navigation_command`)."""
    return "_".join(topic.split("/"))


def find_topic_csvs(session_dir: Path, topic: str) -> list[Path]:
    """A topic's rotated CSV chunks in write order, or [] if the topic was never recorded.

    A configured-but-silent topic leaves no directory at all, which is not an error here -- the
    caller decides whether a missing topic is fatal.
    """
    directory = channel_dir(session_dir, topic)
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{topic_slug(topic)}_*.csv"), key=_sort_key)


def _sort_key(path: Path) -> tuple[int, str]:
    """Order chunks by their encoded start time, with unparseable names first and stable."""
    match = _CHUNK_TIMESTAMP_RE.search(path.name)
    return (int(match.group(1)) if match else -1, path.name)


def find_sessions(root: Path) -> list[Path]:
    """Every session under `root` that carries planning-cycle instrumentation.

    A session without it cannot be cycle-joined at all, so it is not a session this pipeline
    can use -- excluding it here is what keeps that failure at discovery rather than halfway
    through a preprocessing run.
    """
    root = Path(root)
    if _has_cycles(root):
        return [root]
    return sorted(child for child in root.iterdir() if child.is_dir() and _has_cycles(child))


def _has_cycles(session_dir: Path) -> bool:
    return bool(find_topic_csvs(session_dir, PLANNING_CYCLE_TOPIC))

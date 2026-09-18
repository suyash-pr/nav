"""Segmenting a session's planning cycles into navigation runs.

`cycle_id` counts ticks within *one navigation process*. A restart begins a fresh run from 1, and
karter's PlanningCycle docstring is explicit that telling two runs apart is "the session
manifest's job" -- but no session manifest exists in karter today. So the split is inferred here,
and `cycle_id` alone is never a key: `(session, run_id, cycle_id)` is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def segment_runs(cycle_id: np.ndarray) -> np.ndarray:
    """Assign each row a run index, cutting wherever `cycle_id` fails to increase.

    Rows must be in write order, which is time order -- rotated chunks sort by their encoded
    start timestamp, and the recorder appends within a chunk.

    Only a non-increasing `cycle_id` starts a run. A large *timestamp* gap with `cycle_id` still
    increasing is navigation continuing to tick across a recorder gap, not a restart, and
    splitting on it would invent a boundary that no process boundary matches.
    """
    cycle_id = np.asarray(cycle_id, dtype=np.int64)
    if len(cycle_id) == 0:
        return np.zeros(0, dtype=np.int64)
    restart = np.empty(len(cycle_id), dtype=bool)
    restart[0] = False
    restart[1:] = cycle_id[1:] <= cycle_id[:-1]
    return np.cumsum(restart, dtype=np.int64)


def run_bounds(run_id: np.ndarray, timestamp_us: np.ndarray) -> pd.DataFrame:
    """Per-run first and last tick timestamp, for assigning un-keyed rows to a run by time."""
    frame = pd.DataFrame({"run_id": np.asarray(run_id), "timestamp_us": np.asarray(timestamp_us)})
    bounds = frame.groupby("run_id")["timestamp_us"].agg(["min", "max", "count"]).reset_index()
    return bounds.rename(columns={"min": "t_start_us", "max": "t_end_us", "count": "n_cycles"})


def assign_run_by_time(bounds: pd.DataFrame, timestamp_us: np.ndarray) -> np.ndarray:
    """Map timestamps onto run ids by which run's tick span contains them, -1 if none.

    Used for channels that carry `cycle_id` but not `run_id` (navigation/command): the id alone
    is ambiguous across a restart, and the timestamp is what disambiguates it.
    """
    timestamp_us = np.asarray(timestamp_us, dtype=np.int64)
    out = np.full(len(timestamp_us), -1, dtype=np.int64)
    for row in bounds.itertuples():
        inside = (timestamp_us >= row.t_start_us) & (timestamp_us <= row.t_end_us)
        out[inside] = row.run_id
    return out

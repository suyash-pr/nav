"""The ground-truth join: which sensor frames informed which planning tick, and what it decided.

This module is what replaces the guessed per-modality latencies the old pipeline used. Navigation
publishes PlanningCycle and SensorFrontier from the same call each tick, so joining them on
cycle_id recovers exactly what the robot held when it decided -- no timestamp search, no
tolerance, and no possibility of selecting a frame recorded after the decision.

SensorFrontier is an *arrival* frontier, deliberately. A capture-time join would admit frames
whose capture timestamp precedes the tick but which had not yet reached navigation's process; a
policy trained on those learns from data it will not have at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nav.session import layout
from nav.session.actions import veto_mask
from nav.session.csv_reader import read_topic
from nav.session.runs import assign_run_by_time, run_bounds, segment_runs

# Mirrors karter dtypes/sensor_frontier.py and dtypes/planning_cycle.py (pinned at 208a6fce9).
NO_FRAME_AGE_US = -1
NO_FRAME_SEQ = -1
NO_CYCLE_ID = -1

SEQ_PREFIX = "seq."
AGE_PREFIX = "age_us."

# CycleOutcome. Only these three published a command at all; the two skipped_* outcomes still
# consume a cycle_id, so they are present-but-untrainable rather than gaps in the sequence.
OUTCOME_PLANNED = "planned"
OUTCOME_FLOATING = "floating"
OUTCOME_BRAKING = "braking"
TRAINABLE_OUTCOMES = (OUTCOME_PLANNED,)

# NavigationStatus, as carried in PlanningCycle.status when the planner ran. Replaces the old
# pipeline's scan for contiguous `navigation_status == "navigating"` stretches.
TRAINABLE_STATUS = ("navigating",)

# A role whose newest arrived frame is older than this was re-used by navigation across several
# ticks. Unlike the tolerances this replaces, it gates a measured age rather than a guess.
DEFAULT_MAX_AGE_US = 300_000


@dataclass(frozen=True)
class RoleFrames:
    """Per-cycle identity of one sensor role's frame, as navigation held it at tick time."""

    capture_us: np.ndarray  # int64, -1 where no usable frame
    seq: np.ndarray  # int64, NO_FRAME_SEQ where the wire format carries none
    age_us: np.ndarray  # int64, NO_FRAME_AGE_US where nothing ever arrived
    present: np.ndarray  # bool

    def __len__(self) -> int:
        return len(self.capture_us)


@dataclass(frozen=True)
class CycleTable:
    """One session's trainable cycles, with every sensor role resolved to a frame identity."""

    cycles: pd.DataFrame
    roles: dict[str, RoleFrames]
    bounds: pd.DataFrame
    dropped: dict[str, int]

    def __len__(self) -> int:
        return len(self.cycles)


def frontier_roles(frontier: pd.DataFrame) -> list[str]:
    """Role names present in a sensor_frontier frame, from its flattened `age_us.<role>` columns.

    Keyed off age_us rather than seq because every configured role always gets an age column,
    including roles whose wire format carries no sequence at all.
    """
    return sorted(c[len(AGE_PREFIX) :] for c in frontier.columns if c.startswith(AGE_PREFIX))


def load_cycle_table(
    session_dir: Path,
    *,
    max_age_us: int | dict[str, int] = DEFAULT_MAX_AGE_US,
    trainable_outcomes: tuple[str, ...] = TRAINABLE_OUTCOMES,
    trainable_status: tuple[str, ...] = TRAINABLE_STATUS,
    filter_vetoed: bool = True,
) -> CycleTable:
    """Join a session's planning cycles, sensor frontier and commands into one table."""
    session_dir = Path(session_dir)
    dropped: dict[str, int] = {}

    cycle_rows = read_topic(session_dir, layout.PLANNING_CYCLE_TOPIC)
    frontier_rows = read_topic(session_dir, layout.SENSOR_FRONTIER_TOPIC)
    if cycle_rows.empty or frontier_rows.empty:
        raise ValueError(
            f"{session_dir}: no planning-cycle instrumentation "
            f"({layout.PLANNING_CYCLE_TOPIC} / {layout.SENSOR_FRONTIER_TOPIC}). "
            "This session predates karter 208a6fce9 and cannot be cycle-joined."
        )

    cycle_rows = cycle_rows.copy()
    cycle_rows["run_id"] = segment_runs(cycle_rows["cycle_id"].to_numpy())
    bounds = run_bounds(cycle_rows["run_id"].to_numpy(), cycle_rows["timestamp_us"].to_numpy())

    frontier_rows = frontier_rows.copy()
    frontier_rows["run_id"] = assign_run_by_time(bounds, frontier_rows["timestamp_us"].to_numpy())

    # A cycle_id present in one channel but not the other is always a transit drop: both records
    # are published from the same CycleRecorder.record() call, so navigation never emits one
    # without the other. LCM is UDP and karter measured ~0.2% row loss on karter-03.
    before = len(cycle_rows)
    joined = cycle_rows.merge(
        frontier_rows,
        on=["run_id", "cycle_id"],
        how="inner",
        suffixes=("", "_frontier"),
        validate="one_to_one",
    )
    dropped["udp_drop"] = before - len(joined)

    joined = _attach_action(session_dir, joined, bounds, dropped)

    keep = joined["outcome"].isin(trainable_outcomes)
    dropped["untrainable_outcome"] = int((~keep).sum())
    joined = joined[keep]

    keep = joined["status"].isin(trainable_status)
    dropped["untrainable_status"] = int((~keep).sum())
    joined = joined[keep].reset_index(drop=True)

    if filter_vetoed:
        vetoed = veto_mask(session_dir, joined.rename(columns={"timestamp_us": "tick_us"}))
        dropped["vetoed"] = int(vetoed.sum())
        joined = joined[~vetoed].reset_index(drop=True)

    roles = _resolve_roles(joined, frontier_roles(frontier_rows), max_age_us)

    cycles = joined[
        [
            "run_id",
            "cycle_id",
            "timestamp_us",
            "outcome",
            "status",
            "desired_linear_velocity",
            "desired_angular_velocity",
            "velocity_timestamp_us",
            "velocity_seq",
        ]
    ].rename(columns={"timestamp_us": "tick_us"})

    return CycleTable(cycles=cycles.reset_index(drop=True), roles=roles, bounds=bounds, dropped=dropped)


def _attach_action(session_dir: Path, joined: pd.DataFrame, bounds: pd.DataFrame, dropped: dict[str, int]) -> pd.DataFrame:
    """Join navigation/command on (run_id, cycle_id).

    The command carries cycle_id but not run_id, so its run is recovered from its timestamp
    first -- cycle_id alone is ambiguous across a restart. Commands published outside a tick
    (a shutdown stop, a driving tool) carry NO_CYCLE_ID and are dropped.
    """
    commands = read_topic(session_dir, layout.NAVIGATION_COMMAND_TOPIC)
    if commands.empty or "cycle_id" not in commands.columns:
        raise ValueError(
            f"{session_dir}: {layout.NAVIGATION_COMMAND_TOPIC} has no cycle_id column. "
            "This session predates karter 518d7f122 and its actions cannot be cycle-joined."
        )

    commands = commands[commands["cycle_id"] != NO_CYCLE_ID].copy()
    commands["run_id"] = assign_run_by_time(bounds, commands["timestamp_us"].to_numpy())
    commands = commands.drop_duplicates(subset=["run_id", "cycle_id"], keep="first")

    before = len(joined)
    joined = joined.merge(
        commands[["run_id", "cycle_id", "desired_linear_velocity", "desired_angular_velocity"]],
        on=["run_id", "cycle_id"],
        how="inner",
        validate="one_to_one",
    )
    dropped["no_command"] = before - len(joined)
    return joined


def _resolve_roles(joined: pd.DataFrame, roles: list[str], max_age_us: int | dict[str, int]) -> dict[str, RoleFrames]:
    """Turn each role's (seq, age_us) columns into a per-cycle frame identity.

    A role's capture timestamp is `frontier.timestamp_us - age_us[role]` -- karter
    sensor_frontier.py:67. Two sentinels are distinguishable and mean different things:
    NO_FRAME_AGE_US means nothing ever arrived on the role, while a real age with NO_FRAME_SEQ
    means the frame arrived but its wire format carries no sequence.
    """
    tick_us = joined["timestamp_us"].to_numpy(dtype=np.int64)
    resolved: dict[str, RoleFrames] = {}

    for role in roles:
        age = joined[f"{AGE_PREFIX}{role}"].to_numpy(dtype=np.int64)
        seq_column = f"{SEQ_PREFIX}{role}"
        seq = (
            joined[seq_column].to_numpy(dtype=np.int64)
            if seq_column in joined.columns
            else np.full(len(joined), NO_FRAME_SEQ, dtype=np.int64)
        )

        limit = max_age_us[role] if isinstance(max_age_us, dict) else max_age_us
        present = (age != NO_FRAME_AGE_US) & (age >= 0) & (age <= limit)

        capture_us = np.where(present, tick_us - age, -1)

        # The anti-leak invariant, stated rather than assumed: nothing selected may post-date
        # the decision it informed. Holds by construction for age >= 0, so a violation means
        # the age column itself is corrupt.
        if np.any(capture_us[present] > tick_us[present]):
            raise ValueError(f"role {role}: capture timestamp after its own tick")

        resolved[role] = RoleFrames(capture_us=capture_us, seq=seq, age_us=age, present=present)

    return resolved

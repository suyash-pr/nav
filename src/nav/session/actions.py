"""Action and proprioception sources for a cycle table.

Three velocity signals exist and are not interchangeable (see plan discussion):

- `navigation/command` -- the planner's *intent*. This is the action label: TD-MPC proposes
  commands, so the model's action space has to match what it will actually search over.
- `robot/velocity_estimate` -- the *measured* outcome. Loaded as proprioception alongside the
  observation, not as the label, so the model can explain tracking error (slip, a slope) instead
  of absorbing it as unexplained noise. It is exactly cycle-joined too: it is a planner input, so
  PlanningCycle records `velocity_timestamp_us`/`velocity_seq` for it directly.
- `ll_executed_command` -- what actually reached the motors after safety vetoes. It carries no
  `cycle_id` at all, so it is never a label; it is used here purely to *drop* cycles where a
  safety override diverged from what navigation asked for, since training the dynamics model on
  a command that was overridden would teach it a transition that didn't happen. A timestamp join
  is fine for this because a mismatch only costs a dropped sample, not a leaked one.

  `source != motor_target` is *not* itself a veto signal -- measured on a real session, `hold`
  (llcontrol's own loop ran but no new command had arrived that iteration -- it runs faster than
  navigation ticks, so this is the modal value between real updates, not a divergence) and
  `floating` (frequently just navigation's own chosen state, matching that cycle's own outcome)
  together outnumbered `motor_target` by more than 10:1 and, filtered that way, dropped all but
  one cycle of a session. Only the sources that represent the safety system actually overriding
  navigation's intent -- `hw_fault`, `protective_stop`, `stale_timeout` -- are treated as vetoes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nav.session import layout
from nav.session.csv_reader import read_topic

MOTOR_TARGET_SOURCE = "motor_target"
# ExecutedCommandSource values where the safety system actually overrode navigation's command,
# as opposed to a source that just reflects navigation's own chosen state (floating, stopping)
# or llcontrol's own loop cadence (hold -- see the module docstring).
VETO_SOURCES = frozenset({"hw_fault", "protective_stop", "stale_timeout"})
DEFAULT_VETO_EPSILON_US = 100_000


def commanded_action(cycles: pd.DataFrame) -> np.ndarray:
    """(n, 2) float32 [linear, angular] from navigation/command, already joined onto `cycles`."""
    return cycles[["desired_linear_velocity", "desired_angular_velocity"]].to_numpy(dtype=np.float32)


def measured_velocity(session_dir: Path, cycles: pd.DataFrame) -> np.ndarray:
    """(n, 2) float32 [linear_x, angular_z] measured body velocity, exactly cycle-joined.

    PlanningCycle.velocity_seq is the identity of the Velocity row the planner held; matching by
    seq is exact and avoids re-deriving a timestamp tolerance for a value that already has an
    authoritative join key.
    """
    velocity_rows = read_topic(session_dir, layout.VELOCITY_ESTIMATE_TOPIC)
    if velocity_rows.empty:
        return np.zeros((len(cycles), 2), dtype=np.float32)

    linear_x = velocity_rows["linear_velocity"].apply(lambda cell: float(cell.split(",")[0]))
    angular_z = velocity_rows["angular_velocity"].apply(lambda cell: float(cell.split(",")[2]))
    by_seq = pd.DataFrame(
        {"seq": velocity_rows["seq"].to_numpy(dtype=np.int64), "linear_x": linear_x, "angular_z": angular_z}
    ).drop_duplicates(subset="seq", keep="first")

    matched = pd.DataFrame({"velocity_seq": cycles["velocity_seq"].to_numpy(dtype=np.int64)}).merge(
        by_seq, left_on="velocity_seq", right_on="seq", how="left"
    )
    out = matched[["linear_x", "angular_z"]].fillna(0.0).to_numpy(dtype=np.float32)
    return out


def veto_mask(session_dir: Path, cycles: pd.DataFrame, *, epsilon_us: int = DEFAULT_VETO_EPSILON_US) -> np.ndarray:
    """True where a cycle's command should be dropped because the wheels didn't run it as issued.

    `ll_executed_command` has no cycle_id, so this is a nearest-timestamp match rather than an
    exact join -- acceptable here because the only consequence of a mis-match is an extra dropped
    sample, never a leaked one.
    """
    executed_rows = read_topic(session_dir, layout.EXECUTED_COMMAND_TOPIC)
    if executed_rows.empty:
        return np.zeros(len(cycles), dtype=bool)

    executed_ts = executed_rows["timestamp_us"].to_numpy(dtype=np.int64)
    order = np.argsort(executed_ts, kind="stable")
    executed_ts = executed_ts[order]
    executed_source = executed_rows["source"].to_numpy()[order]

    tick_us = cycles["tick_us"].to_numpy(dtype=np.int64)
    position = np.clip(np.searchsorted(executed_ts, tick_us), 1, len(executed_ts) - 1)
    left, right = position - 1, position
    use_left = np.abs(executed_ts[left] - tick_us) <= np.abs(executed_ts[right] - tick_us)
    nearest = np.where(use_left, left, right)
    within = np.abs(executed_ts[nearest] - tick_us) <= epsilon_us

    vetoed = np.isin(executed_source[nearest], list(VETO_SOURCES))
    return vetoed & within  # no nearby executed-command row at all is not itself evidence of a veto

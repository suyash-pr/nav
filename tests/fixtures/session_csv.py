"""Minimal hand-built session fixtures: real ;-CSV files at the real path/naming contract,
constructed directly with pandas rather than through karter's recorder. Enough to pin the join
logic in nav.session.cycles without needing karter importable or real video groups.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CHUNK_START = "0" * 20  # any 20-digit chunk-start name is fine; these tests never window-select


def _write(session_dir: Path, topic: str, frame: pd.DataFrame) -> None:
    directory = session_dir / "data" / topic
    directory.mkdir(parents=True, exist_ok=True)
    slug = "_".join(topic.split("/"))
    frame.to_csv(directory / f"{slug}_{CHUNK_START}.csv", sep=";", index=False)


def write_planning_cycle(session_dir: Path, rows: list[dict]) -> None:
    columns = [
        "timestamp_us",
        "cycle_id",
        "outcome",
        "status",
        "skip_reason",
        "grid_timestamp_us",
        "grid_seq",
        "pose_timestamp_us",
        "pose_seq",
        "velocity_timestamp_us",
        "velocity_seq",
        "torque_left_timestamp_us",
        "torque_right_timestamp_us",
        "goal_timestamp_us",
        "goal_type",
        "goal_x_m",
        "goal_y_m",
        "goal_yaw_rad",
    ]
    defaults = {c: 0 for c in columns} | {"outcome": "planned", "status": "navigating", "skip_reason": "", "goal_type": ""}
    frame = pd.DataFrame([{**defaults, **row} for row in rows])[columns]
    _write(session_dir, "navigation/planning_cycle", frame)


def write_sensor_frontier(session_dir: Path, rows: list[dict], roles: list[str]) -> None:
    columns = ["timestamp_us", "cycle_id", "provenance"]
    columns += [f"seq.{role}" for role in roles] + [f"age_us.{role}" for role in roles]
    defaults = {c: -1 for c in columns} | {"provenance": "observed"}
    frame = pd.DataFrame([{**defaults, **row} for row in rows])[columns]
    _write(session_dir, "navigation/sensor_frontier", frame)


def write_command(session_dir: Path, rows: list[dict]) -> None:
    columns = [
        "timestamp_us",
        "navigation_status",
        "pid_interpolation_duration_s",
        "torque_left",
        "torque_right",
        "desired_linear_velocity",
        "desired_angular_velocity",
        "cycle_id",
    ]
    defaults = {c: 0 for c in columns} | {"navigation_status": "navigating", "pid_interpolation_duration_s": 0.1}
    frame = pd.DataFrame([{**defaults, **row} for row in rows])[columns]
    _write(session_dir, "navigation/command", frame)


def write_executed_command(session_dir: Path, rows: list[dict]) -> None:
    columns = [
        "timestamp_us",
        "source",
        "braking",
        "floating",
        "brake_current",
        "desired_linear_velocity",
        "desired_angular_velocity",
        "torque_left",
        "torque_right",
    ]
    defaults = {c: 0 for c in columns} | {"source": "motor_target", "braking": False, "floating": False}
    frame = pd.DataFrame([{**defaults, **row} for row in rows])[columns]
    _write(session_dir, "ll_executed_command", frame)


def write_velocity_estimate(session_dir: Path, rows: list[dict]) -> None:
    columns = ["timestamp_us", "frame_id", "seq", "linear_velocity", "angular_velocity"]
    defaults = {"frame_id": "robot", "linear_velocity": "0.0,0.0,0.0", "angular_velocity": "0.0,0.0,0.0"}
    frame = pd.DataFrame([{**defaults, **row} for row in rows])[columns]
    _write(session_dir, "robot/velocity_estimate", frame)

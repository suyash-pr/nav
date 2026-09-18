import pytest
from fixtures.session_csv import (
    write_command,
    write_executed_command,
    write_planning_cycle,
    write_sensor_frontier,
    write_velocity_estimate,
)

from nav.session.cycles import NO_FRAME_AGE_US, load_cycle_table


def _basic_session(tmp_path, n=5, roles=("rgb_front",), start_tick_us=0, cycle_period_us=100_000):
    cycles = [
        {"timestamp_us": start_tick_us + i * cycle_period_us, "cycle_id": i + 1, "velocity_seq": i}
        for i in range(n)
    ]
    frontier = [
        {
            "timestamp_us": start_tick_us + i * cycle_period_us,
            "cycle_id": i + 1,
            **{f"seq.{role}": i for role in roles},
            **{f"age_us.{role}": 10_000 for role in roles},
        }
        for i in range(n)
    ]
    commands = [
        {
            "timestamp_us": start_tick_us + i * cycle_period_us,
            "cycle_id": i + 1,
            "desired_linear_velocity": 1.0,
            "desired_angular_velocity": 0.0,
        }
        for i in range(n)
    ]
    write_planning_cycle(tmp_path, cycles)
    write_sensor_frontier(tmp_path, frontier, list(roles))
    write_command(tmp_path, commands)
    return tmp_path


def test_load_cycle_table_basic_join(tmp_path):
    session = _basic_session(tmp_path, n=5)
    table = load_cycle_table(session, filter_vetoed=False)

    assert len(table) == 5
    assert table.cycles["cycle_id"].tolist() == [1, 2, 3, 4, 5]
    assert table.cycles["run_id"].tolist() == [0, 0, 0, 0, 0]
    role = table.roles["rgb_front"]
    assert role.present.all()
    # capture_us = tick_us - age_us, and must never post-date the tick (the anti-leak invariant)
    assert (role.capture_us <= table.cycles["tick_us"].to_numpy()).all()


def test_udp_drop_in_one_channel_only_is_dropped_and_counted(tmp_path):
    cycles = [{"timestamp_us": i * 100_000, "cycle_id": i + 1, "velocity_seq": i} for i in range(5)]
    frontier = [
        {"timestamp_us": i * 100_000, "cycle_id": i + 1, "seq.rgb_front": i, "age_us.rgb_front": 10_000}
        for i in range(5)
        if i != 2  # row 3 lost on the frontier channel only -- a transit drop, not a real gap
    ]
    commands = [
        {"timestamp_us": i * 100_000, "cycle_id": i + 1, "desired_linear_velocity": 1.0, "desired_angular_velocity": 0.0}
        for i in range(5)
    ]
    write_planning_cycle(tmp_path, cycles)
    write_sensor_frontier(tmp_path, frontier, ["rgb_front"])
    write_command(tmp_path, commands)

    table = load_cycle_table(tmp_path, filter_vetoed=False)

    assert len(table) == 4
    assert table.cycles["cycle_id"].tolist() == [1, 2, 4, 5]
    assert table.dropped["udp_drop"] == 1


def test_navigation_restart_produces_two_runs(tmp_path):
    # cycle_id resets to 1 after a restart; both runs otherwise look identical.
    first_run = [{"timestamp_us": i * 100_000, "cycle_id": i + 1, "velocity_seq": i} for i in range(3)]
    second_run = [{"timestamp_us": 10_000_000 + i * 100_000, "cycle_id": i + 1, "velocity_seq": i} for i in range(3)]
    cycles = first_run + second_run
    frontier = [
        {"timestamp_us": row["timestamp_us"], "cycle_id": row["cycle_id"], "seq.rgb_front": i, "age_us.rgb_front": 10_000}
        for i, row in enumerate(cycles)
    ]
    commands = [
        {
            "timestamp_us": row["timestamp_us"],
            "cycle_id": row["cycle_id"],
            "desired_linear_velocity": 1.0,
            "desired_angular_velocity": 0.0,
        }
        for row in cycles
    ]
    write_planning_cycle(tmp_path, cycles)
    write_sensor_frontier(tmp_path, frontier, ["rgb_front"])
    write_command(tmp_path, commands)

    table = load_cycle_table(tmp_path, filter_vetoed=False)

    assert table.cycles["run_id"].tolist() == [0, 0, 0, 1, 1, 1]
    # cycle_id alone is ambiguous across the restart -- both runs have a cycle_id == 1 -- but the
    # (run_id, cycle_id) pairs must be unique.
    keys = list(zip(table.cycles["run_id"], table.cycles["cycle_id"]))
    assert len(keys) == len(set(keys))


def test_skipped_and_non_navigating_cycles_are_dropped(tmp_path):
    cycles = [
        {"timestamp_us": 0, "cycle_id": 1, "outcome": "planned", "status": "navigating", "velocity_seq": 0},
        {"timestamp_us": 100_000, "cycle_id": 2, "outcome": "skipped_no_data", "status": "", "velocity_seq": 1},
        {"timestamp_us": 200_000, "cycle_id": 3, "outcome": "skipped_behind_schedule", "status": "", "velocity_seq": 2},
        {"timestamp_us": 300_000, "cycle_id": 4, "outcome": "planned", "status": "floating", "velocity_seq": 3},
        {"timestamp_us": 400_000, "cycle_id": 5, "outcome": "planned", "status": "navigating", "velocity_seq": 4},
    ]
    frontier = [
        {"timestamp_us": row["timestamp_us"], "cycle_id": row["cycle_id"], "seq.rgb_front": i, "age_us.rgb_front": 10_000}
        for i, row in enumerate(cycles)
    ]
    # skipped_* cycles publish no command at all -- they never reached the point that would.
    commands = [
        {
            "timestamp_us": row["timestamp_us"],
            "cycle_id": row["cycle_id"],
            "desired_linear_velocity": 1.0,
            "desired_angular_velocity": 0.0,
        }
        for row in cycles
        if row["outcome"] != "skipped_no_data" and row["outcome"] != "skipped_behind_schedule"
    ]
    write_planning_cycle(tmp_path, cycles)
    write_sensor_frontier(tmp_path, frontier, ["rgb_front"])
    write_command(tmp_path, commands)

    table = load_cycle_table(tmp_path, filter_vetoed=False)

    # Only cycle 1 and 5 are outcome=planned AND status=navigating.
    assert table.cycles["cycle_id"].tolist() == [1, 5]


def test_missing_role_is_reported_as_not_present(tmp_path):
    cycles = [{"timestamp_us": i * 100_000, "cycle_id": i + 1, "velocity_seq": i} for i in range(3)]
    frontier = [
        {
            "timestamp_us": i * 100_000,
            "cycle_id": i + 1,
            "seq.rgb_front": i if i != 1 else -1,
            "age_us.rgb_front": 10_000 if i != 1 else NO_FRAME_AGE_US,
        }
        for i in range(3)
    ]
    commands = [
        {"timestamp_us": i * 100_000, "cycle_id": i + 1, "desired_linear_velocity": 1.0, "desired_angular_velocity": 0.0}
        for i in range(3)
    ]
    write_planning_cycle(tmp_path, cycles)
    write_sensor_frontier(tmp_path, frontier, ["rgb_front"])
    write_command(tmp_path, commands)

    table = load_cycle_table(tmp_path, filter_vetoed=False)

    role = table.roles["rgb_front"]
    assert role.present.tolist() == [True, False, True]
    assert role.capture_us[1] == -1


def test_stale_frame_beyond_max_age_is_treated_as_absent(tmp_path):
    cycles = [{"timestamp_us": 0, "cycle_id": 1, "velocity_seq": 0}]
    frontier = [{"timestamp_us": 0, "cycle_id": 1, "seq.rgb_front": 0, "age_us.rgb_front": 500_000}]
    commands = [{"timestamp_us": 0, "cycle_id": 1, "desired_linear_velocity": 1.0, "desired_angular_velocity": 0.0}]
    write_planning_cycle(tmp_path, cycles)
    write_sensor_frontier(tmp_path, frontier, ["rgb_front"])
    write_command(tmp_path, commands)

    table = load_cycle_table(tmp_path, max_age_us=300_000, filter_vetoed=False)
    assert not table.roles["rgb_front"].present[0]


def test_vetoed_cycle_is_dropped_when_filter_enabled(tmp_path):
    session = _basic_session(tmp_path, n=3)
    write_executed_command(
        session,
        [
            {"timestamp_us": 0, "source": "motor_target"},
            {"timestamp_us": 100_000, "source": "protective_stop"},  # vetoed
            {"timestamp_us": 200_000, "source": "motor_target"},
        ],
    )

    table = load_cycle_table(session, filter_vetoed=True)

    assert table.cycles["cycle_id"].tolist() == [1, 3]
    assert table.dropped["vetoed"] == 1


def test_no_planning_cycle_instrumentation_raises(tmp_path):
    with pytest.raises(ValueError):
        load_cycle_table(tmp_path)


def test_no_cycle_id_column_on_command_raises(tmp_path):
    cycles = [{"timestamp_us": 0, "cycle_id": 1, "velocity_seq": 0}]
    frontier = [{"timestamp_us": 0, "cycle_id": 1, "seq.rgb_front": 0, "age_us.rgb_front": 10_000}]
    write_planning_cycle(tmp_path, cycles)
    write_sensor_frontier(tmp_path, frontier, ["rgb_front"])
    # Write a pre-518d7f122-style command CSV with no cycle_id column at all.
    import pandas as pd

    directory = tmp_path / "data" / "navigation" / "command"
    directory.mkdir(parents=True)
    pd.DataFrame([{"timestamp_us": 0, "desired_linear_velocity": 1.0, "desired_angular_velocity": 0.0}]).to_csv(
        directory / f"navigation_command_{'0' * 20}.csv", sep=";", index=False
    )

    with pytest.raises(ValueError):
        load_cycle_table(tmp_path)


def test_measured_velocity_resolves_by_exact_seq(tmp_path):
    from nav.session.actions import measured_velocity

    session = _basic_session(tmp_path, n=2)
    write_velocity_estimate(
        session,
        [
            {"timestamp_us": 0, "seq": 0, "linear_velocity": "1.5,0.0,0.0", "angular_velocity": "0.0,0.0,0.25"},
            {"timestamp_us": 100_000, "seq": 1, "linear_velocity": "2.5,0.0,0.0", "angular_velocity": "0.0,0.0,-0.1"},
        ],
    )
    table = load_cycle_table(session, filter_vetoed=False)

    proprio = measured_velocity(session, table.cycles)

    import numpy as np

    np.testing.assert_allclose(proprio, [[1.5, 0.25], [2.5, -0.1]], atol=1e-5)

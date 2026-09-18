import numpy as np

from nav.session.runs import assign_run_by_time, run_bounds, segment_runs


def test_segment_runs_single_continuous_run():
    cycle_id = np.arange(1, 51)
    run_id = segment_runs(cycle_id)
    assert np.all(run_id == 0)


def test_segment_runs_restart_starts_new_run():
    cycle_id = np.concatenate([np.arange(1, 51), np.arange(1, 51)])
    run_id = segment_runs(cycle_id)
    assert np.all(run_id[:50] == 0)
    assert np.all(run_id[50:] == 1)


def test_segment_runs_timestamp_gap_with_monotone_id_is_not_a_restart():
    # cycle_id keeps increasing even though (in a real session) a large wall-clock gap sits
    # between rows -- segment_runs only looks at cycle_id, so this must stay one run.
    cycle_id = np.array([1, 2, 3, 50, 51, 52])
    run_id = segment_runs(cycle_id)
    assert np.all(run_id == 0)


def test_segment_runs_non_increasing_mid_sequence():
    cycle_id = np.array([48, 49, 3, 4, 5])
    run_id = segment_runs(cycle_id)
    assert run_id.tolist() == [0, 0, 1, 1, 1]


def test_segment_runs_empty():
    assert segment_runs(np.array([], dtype=np.int64)).shape == (0,)


def test_run_bounds_and_assign_by_time():
    run_id = np.array([0, 0, 0, 1, 1])
    timestamp_us = np.array([100, 200, 300, 1000, 1100])
    bounds = run_bounds(run_id, timestamp_us)

    assert bounds["run_id"].tolist() == [0, 1]
    assert bounds["t_start_us"].tolist() == [100, 1000]
    assert bounds["t_end_us"].tolist() == [300, 1100]
    assert bounds["n_cycles"].tolist() == [3, 2]

    assigned = assign_run_by_time(bounds, np.array([150, 1050, 5000]))
    assert assigned.tolist() == [0, 1, -1]  # 5000 falls outside every run's span

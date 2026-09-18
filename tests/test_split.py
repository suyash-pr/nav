
from fixtures.minimal_store import write_minimal_store

from nav.dataset import CycleDataset
from nav.utils.split import split_by_run, split_by_session


def _contiguous_store(tmp_path, name, n, start_tick_us=0):
    cycle_id = list(range(1, n + 1))
    tick_us = [start_tick_us + i * 100_000 for i in range(n)]
    return write_minimal_store(tmp_path / name, run_id=[0] * n, cycle_id=cycle_id, tick_us=tick_us)


def test_split_by_session_holds_out_whole_trailing_sessions(tmp_path):
    stores = [_contiguous_store(tmp_path, f"session_{k}", n=20) for k in range(5)]
    datasets = [CycleDataset(store) for store in stores]

    train_idx, val_idx = split_by_session(datasets, val_fraction=0.2)

    # ~20% of 5 sessions' worth of samples -> should hold out exactly one trailing session, and
    # the split must partition every dataset index between the two.
    assert set(train_idx) | set(val_idx) == set(range(5))
    assert set(train_idx) & set(val_idx) == set()
    assert val_idx == sorted(val_idx)
    assert max(train_idx) < min(val_idx)  # held-out sessions are the trailing ones


def test_split_by_session_never_empties_train_with_multiple_sessions(tmp_path):
    stores = [_contiguous_store(tmp_path, f"session_{k}", n=5) for k in range(2)]
    datasets = [CycleDataset(store) for store in stores]

    train_idx, val_idx = split_by_session(datasets, val_fraction=0.9)

    assert len(train_idx) >= 1
    assert len(val_idx) >= 1


def test_split_by_run_guard_band_prevents_adjacent_leakage(tmp_path):
    store = _contiguous_store(tmp_path, "session", n=100)
    dataset = CycleDataset(store)  # step_cycles=2 by default -> pairs (i, i+2) for i in [0, 97]

    train_idx, val_idx = split_by_run(dataset, val_fraction=0.2)

    train_pair_starts = dataset.pairs[train_idx, 0]
    val_pair_starts = dataset.pairs[val_idx, 0]
    assert val_pair_starts.min() > train_pair_starts.max()
    # No train pair's second endpoint may equal a val pair's first endpoint or vice versa.
    train_ends = set(dataset.pairs[train_idx, 1].tolist())
    val_starts = set(val_pair_starts.tolist())
    assert train_ends.isdisjoint(val_starts)


def test_split_by_run_holds_out_whole_trailing_runs_when_multiple_exist(tmp_path):
    run_id = [0] * 10 + [1] * 10
    cycle_id = list(range(1, 11)) + list(range(1, 11))
    tick_us = [i * 100_000 for i in range(10)] + [10_000_000 + i * 100_000 for i in range(10)]
    store = write_minimal_store(tmp_path / "session", run_id=run_id, cycle_id=cycle_id, tick_us=tick_us)
    dataset = CycleDataset(store)

    train_idx, val_idx = split_by_run(dataset, val_fraction=0.3)

    val_runs = set(dataset.run_id[dataset.pairs[val_idx, 0]].tolist())
    train_runs = set(dataset.run_id[dataset.pairs[train_idx, 0]].tolist())
    assert val_runs.isdisjoint(train_runs)

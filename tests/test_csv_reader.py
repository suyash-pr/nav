import pytest

from nav.session.csv_reader import RaggedChunkError, check_rectangular, read_chunk


def test_check_rectangular_passes_on_well_formed_csv(tmp_path):
    path = tmp_path / "chunk.csv"
    path.write_text("timestamp_us;seq.a\n0;1\n100;2\n")
    check_rectangular(path)  # must not raise


def test_check_rectangular_rejects_ragged_row(tmp_path):
    # Reproduces karter's pre-c3ee1d702 sensor_frontier corruption: the header locked in with
    # one column from the first row, and a later row carrying more columns misaligns under it.
    path = tmp_path / "chunk.csv"
    path.write_text("timestamp_us;seq.rgb_left\n0;1\n100;2;3\n")

    with pytest.raises(RaggedChunkError):
        check_rectangular(path)


def test_read_chunk_raises_on_ragged_file_by_default(tmp_path):
    path = tmp_path / "chunk.csv"
    path.write_text("timestamp_us;seq.rgb_left\n0;1\n100;2;3\n")

    with pytest.raises(RaggedChunkError):
        read_chunk(path)


def test_check_rectangular_empty_file_is_fine(tmp_path):
    path = tmp_path / "chunk.csv"
    path.write_text("")
    check_rectangular(path)  # header-only-or-nothing must not raise

"""Reading karter's `;`-delimited rotated CSV chunks.

karter's CSVWriter fixes a chunk's header from the first row it writes and only *warns* if a
later row carries a different column set, writing it anyway (csv_storage.py:92-102). That is not
a hypothetical: it is exactly how `sensor_frontier` was corrupted before karter c3ee1d702, where
the `seq` dict's key set grew as streams came online and every row after the first misaligned
under a header locked to a single column. Such a file cannot be repaired after the fact, so this
module detects ragged chunks and refuses them rather than handing back silently shifted data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from nav.session.layout import find_topic_csvs

SEPARATOR = ";"


class RaggedChunkError(ValueError):
    """A chunk holds rows whose field count disagrees with its header.

    Recorded before karter c3ee1d702 -- the data is unrecoverable, because which column a value
    belongs to is exactly what was lost.
    """


def check_rectangular(path: Path) -> None:
    """Raise RaggedChunkError if any row's field count differs from the header's."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter=SEPARATOR)
        try:
            header = next(reader)
        except StopIteration:
            return
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise RaggedChunkError(
                    f"{path}: header has {len(header)} fields but line {line_number} has "
                    f"{len(row)}. Recorded before karter c3ee1d702; the column alignment is "
                    f"unrecoverable, so this session cannot be used."
                )


def read_chunk(path: Path, *, validate: bool = True) -> pd.DataFrame:
    if validate:
        check_rectangular(path)
    # karter's own reader treats an empty cell as the literal empty string (key_value_data.py:
    # "elif actual_type is str: kv[column] = ''"), e.g. PlanningCycle.status/skip_reason before
    # a planner ran. pandas' default NA sniffing would turn that into float NaN instead -- and
    # would also mis-fire on any free-text field (skip_reason) that happens to contain a token
    # from pandas' default NA list (e.g. "NA", "N/A", "None"). keep_default_na=False matches
    # karter's own empty-cell semantics exactly rather than pandas' generic CSV convention.
    return pd.read_csv(path, sep=SEPARATOR, keep_default_na=False, na_values=[])


def read_topic(session_dir: Path, topic: str, *, validate: bool = True) -> pd.DataFrame:
    """Every chunk of `topic` concatenated in write order.

    Returns an empty frame when the topic was never recorded, so a caller can distinguish
    "absent" from "present but empty" only by checking `find_topic_csvs` itself -- for every
    consumer here the two are handled the same way.
    """
    frames = [read_chunk(path, validate=validate) for path in find_topic_csvs(session_dir, topic)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

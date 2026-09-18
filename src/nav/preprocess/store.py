"""Writing one session's preprocessed cycle store: memmapped arrays + a self-describing manifest.

Random access into H.265/FFV1 groups is infeasible for a shuffled dataloader, so this pass
decodes everything a session's trainable cycles reference exactly once and writes it out flat.
`__getitem__` at training time becomes pure memmap slicing -- no decode, no join, no search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


@dataclass
class SessionManifest:
    schema_version: int = SCHEMA_VERSION
    sensor_version: str = ""
    session: str = ""
    n_cycles: int = 0
    rgb_cameras: list[str] = field(default_factory=list)
    depth_cameras: list[str] = field(default_factory=list)
    rgb_size: tuple[int, int] = (0, 0)
    depth_size: tuple[int, int] = (0, 0)
    bev_shape: tuple[int, int] = (0, 0)
    action_source: str = "commanded_cycle"
    rgb_join_strategy: dict[str, str] = field(default_factory=dict)
    rgb_match_rate: dict[str, float] = field(default_factory=dict)
    rgb_decode_rate: dict[str, float] = field(default_factory=dict)
    depth_join_strategy: dict[str, str] = field(default_factory=dict)
    depth_match_rate: dict[str, float] = field(default_factory=dict)
    depth_decode_rate: dict[str, float] = field(default_factory=dict)
    lidar_match_rate: dict[str, float] = field(default_factory=dict)
    lidar_roles_configured: list[str] = field(default_factory=list)
    lidar_valid_rate: float = 0.0
    dropped: dict[str, int] = field(default_factory=dict)

    def write(self, out_dir: Path) -> None:
        payload = {**self.__dict__}
        (out_dir / MANIFEST_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def read(cls, out_dir: Path) -> SessionManifest:
        payload = json.loads((Path(out_dir) / MANIFEST_NAME).read_text())
        return cls(**payload)


def write_array(out_dir: Path, relative_path: str, array: np.ndarray) -> None:
    path = Path(out_dir) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def read_array(store_dir: Path, relative_path: str, *, mmap: bool = True) -> np.ndarray:
    path = Path(store_dir) / relative_path
    return np.load(path, mmap_mode="r" if mmap else None)

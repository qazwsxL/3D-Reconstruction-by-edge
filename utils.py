from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

    path.write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

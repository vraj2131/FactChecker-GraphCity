import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Iterable

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    """
    Create a directory if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file into a list of dictionaries.
    """
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e

    return rows

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Stream a JSONL file row by row.
    Useful for large FEVER wiki-pages files.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write dictionaries to a JSONL file.
    """
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """
    Append one dictionary row to a JSONL file.
    """
    ensure_dir(path.parent)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_jsonl_files(directory: Path) -> List[Path]:
    """
    Return all JSONL files in a directory, sorted by name.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.jsonl"))
    return files

def read_json(path: Path) -> Any:
    """
    Read a JSON file from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, indent: int = 2) -> None:
    """
    Write JSON payload to disk.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame to parquet.
    """
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to save parquet to {path}. "
            f"If this is a parquet engine issue, install pyarrow:\n"
            f"    pip install pyarrow\n\n"
            f"Original error: {e}"
        ) from e


def load_parquet(path: Path) -> pd.DataFrame:
    """
    Load a parquet file into a DataFrame.
    """
    return pd.read_parquet(path)

def save_numpy(array: np.ndarray, path: Path) -> None:
    """
    Save a numpy array to disk.
    """
    ensure_dir(path.parent)
    np.save(path, array)


def load_numpy(path: Path) -> np.ndarray:
    """
    Load a numpy array from disk.
    """
    return np.load(path, allow_pickle=False)


def save_pickle(obj: Any, path: Path) -> None:
    """
    Save a Python object with pickle.
    """
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """
    Load a pickled Python object.
    """
    with path.open("rb") as f:
        return pickle.load(f)
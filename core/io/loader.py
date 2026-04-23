# core/io/loader.py

"""
Data and configuration loading utilities.

This module provides a small unified loading layer for:
- checking file existence
- loading JSON configs
- expanding grid configs into multiple combinations
- loading tabular datasets from different file formats
- loading raw and processed project data from standard directories
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from itertools import product
from copy import deepcopy

import pandas as pd

import paths  # central paths module: CONFIGS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR


# ============================================================
# 1) File existence helper
# ============================================================

def check_file(path: Path) -> Path:
    """
    Validates that a path exists and points to a file.

    Args:
        path (Path):
            File path to validate.

    Returns:
        Path:
            The same validated path, useful for chaining.

    Raises:
        FileNotFoundError:
            If the path does not exist or is not a file.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")
    return path


# ============================================================
# 2) JSON loading
# ============================================================

def load_json(file_path: Path | str) -> Dict[str, Any]:
    """
    Loads a JSON file from disk.

    Args:
        file_path (Path | str):
            Absolute or relative path to a JSON file.

    Returns:
        Dict[str, Any]:
            Parsed JSON content.
    """
    p = check_file(Path(file_path))

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 3) Single config loading
# ============================================================

def load_config(config: str | Path, is_path: bool = False) -> Dict[str, Any]:
    """
    Loads a configuration dictionary.

    Supported usage:
    - by config name inside paths.CONFIGS_DIR
    - by explicit filesystem path

    Args:
        config (str | Path):
            Either a config filename (for example "model_train_basic")
            or a direct path to a JSON file.

        is_path (bool):
            If True, `config` is treated as a direct path.
            If False, the file is searched inside paths.CONFIGS_DIR.

    Returns:
        Dict[str, Any]:
            Loaded configuration dictionary.
    """
    if is_path:
        return load_json(config)

    # Treat config as a file name inside CONFIGS_DIR.
    name = str(config)
    if not name.endswith(".json"):
        name += ".json"

    cfg_path = Path(paths.CONFIGS_DIR) / name
    return load_json(cfg_path)


# ------------------------------------------------------------
# Helper: determines whether a value looks like a grid axis
# ------------------------------------------------------------
def _is_iterable_grid(v):
    """
    Checks whether a value looks iterable enough to be interpreted as a grid-like object.

    Notes:
    - lists and tuples are treated as iterable
    - strings are explicitly excluded
    - this helper is currently kept for compatibility / future use
    """
    if isinstance(v, (list, tuple)):
        return True
    if isinstance(v, str):
        return False

    try:
        iter(v)
        return True
    except TypeError:
        return False


# ------------------------------------------------------------
# Recursive grid expansion
# ------------------------------------------------------------
def _expand_grid_node(node):
    """
    Recursively expands a config node into all grid combinations.

    Rules:
    1. {"__grid__": [...]} defines an explicit grid axis
    2. normal dict -> recursive cartesian product across keys
    3. normal list -> treated as a plain value, not as a grid axis
    4. scalar -> returned as a single option

    This allows configs like:
        {
            "model": {
                "hidden": {"__grid__": [64, 128]},
                "dropout": {"__grid__": [0.1, 0.2]}
            }
        }

    to be expanded into multiple concrete configs.
    """
    # =========================================================
    # Explicit grid axis
    # =========================================================
    if isinstance(node, dict) and set(node.keys()) == {"__grid__"}:
        values = node["__grid__"]

        if not isinstance(values, list):
            raise TypeError("__grid__ value must be a list")

        result = []
        for item in values:
            expanded_items = _expand_grid_node(item)

            if isinstance(expanded_items, list):
                result.extend(deepcopy(expanded_items))
            else:
                result.append(deepcopy(expanded_items))

        return result

    # =========================================================
    # Normal dict -> recursive cartesian product
    # =========================================================
    if isinstance(node, dict):
        expanded = {k: _expand_grid_node(v) for k, v in node.items()}
        keys = list(expanded.keys())

        result = []
        for combo in product(*(expanded[k] for k in keys)):
            result.append({k: deepcopy(v) for k, v in zip(keys, combo)})

        return result

    # =========================================================
    # Normal list = plain value, not a grid axis
    # =========================================================
    if isinstance(node, list):
        return [deepcopy(node)]

    # =========================================================
    # Scalar
    # =========================================================
    return [deepcopy(node)]


def load_config_grid(name: str):
    """
    Loads a config and yields all expanded grid combinations.

    Args:
        name (str):
            Config name inside CONFIGS_DIR.

    Yields:
        dict:
            One concrete config combination at a time.
    """
    cfg = load_config(name)
    yield from _expand_grid_node(cfg)


# ============================================================
# 5) DataFrame loading
# ============================================================

def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    """
    Loads a tabular file into a pandas DataFrame.

    Supported formats:
    - .csv
    - .parquet / .pq
    - .json
    - .xlsx / .xls

    If the extension is unknown, CSV is used as a fallback.

    Args:
        file_path (str | Path):
            Path to the dataset file.

    Returns:
        pd.DataFrame:
            Loaded dataframe.
    """
    p = check_file(Path(file_path))
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(p)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    elif suffix == ".json":
        return pd.read_json(p)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    else:
        # Fallback to CSV for unknown file extensions.
        return pd.read_csv(p)


# ============================================================
# 6) Raw / processed data loading
# ============================================================

def load_raw_data(file_name: str) -> Dict[str, Any]:
    """
    Loads raw project data from paths.RAW_DATA_DIR.

    Expected use case:
    raw files are typically stored as JSON,
    for example raw Binance candle dumps.

    Args:
        file_name (str):
            File name inside RAW_DATA_DIR.

    Returns:
        Dict[str, Any]:
            Parsed raw JSON content.
    """
    p = Path(paths.RAW_DATA_DIR) / file_name
    return load_json(p)


def load_processed_data(file_name: str) -> pd.DataFrame:
    """
    Loads processed feature data from paths.PROCESSED_DATA_DIR.

    Typical use case:
    processed datasets are stored as CSV or Parquet files.

    Args:
        file_name (str):
            File name inside PROCESSED_DATA_DIR.

    Returns:
        pd.DataFrame:
            Loaded processed dataframe.
    """
    p = Path(paths.PROCESSED_DATA_DIR) / file_name
    return load_dataframe(p)
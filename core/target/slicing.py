# target/slicing.py

import pandas as pd
from typing import List, Tuple, Optional, Dict, Union, Any, Iterable


def ensure_datetime(df: pd.DataFrame, cols: Union[str, Iterable[str]]):
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], utc=True)

    return df


def cut_by_date(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_col: str = "open_time_eth",
) -> pd.DataFrame:
    """
    Returns a dataframe restricted to the inclusive date interval
    [start_date, end_date].

    Args:
        df (pd.DataFrame):
            Input dataframe.

        start_date (Optional[str]):
            Left boundary of the interval.

        end_date (Optional[str]):
            Right boundary of the interval.

        time_col (str):
            Name of the datetime column used for filtering.

    Returns:
        pd.DataFrame:
            Filtered dataframe with reset index.
    """
    out = df.copy()
    out = ensure_datetime(out, time_col)

    if start_date is not None:
        out = out[out[time_col] >= pd.to_datetime(start_date, utc=True)]

    if end_date is not None:
        out = out[out[time_col] <= pd.to_datetime(end_date, utc=True)]

    return out.reset_index(drop=True)


def cut_by_date_ranges(
    df: pd.DataFrame,
    ranges: List[Tuple[str, str]],
    time_col: str = "open_time_eth",
    duplicates: bool = False,
) -> pd.DataFrame:
    """
    Combines multiple date ranges into a single dataframe.

    Each range is processed with cut_by_date(...), then all parts are concatenated.

    Args:
        df (pd.DataFrame):
            Input dataframe.

        ranges (List[Tuple[str, str]]):
            List of (start_date, end_date) intervals.

        time_col (str):
            Datetime column used for filtering.

        duplicates (bool):
            Whether to keep duplicate rows by time_col after concatenation.

    Returns:
        pd.DataFrame:
            Concatenated dataframe sorted by time_col.
    """
    ensure_datetime(df, time_col)

    parts = [cut_by_date(df, start, end, time_col) for (start, end) in ranges]

    if not parts:
        return df.iloc[0:0].copy()

    out = pd.concat(parts, axis=0)

    if not duplicates:
        out = out.drop_duplicates(subset=[time_col])

    return out.sort_values(time_col).reset_index(drop=True)


def split_by_date_ranges(
    df: pd.DataFrame,
    *ranges_groups: List[Tuple[str, str]],
    time_col: str = "open_time_eth",
    duplicates: bool = False,
) -> Tuple[pd.DataFrame, ...]:
    """
    Splits a dataframe into multiple windows defined by separate groups of date ranges.

    Example:
        split_by_date_ranges(df, train_ranges, val_ranges, test_ranges)

    Args:
        df (pd.DataFrame):
            Input dataframe.

        *ranges_groups:
            Multiple lists of (start_date, end_date) tuples.
            Each list becomes one output dataframe.

        time_col (str):
            Datetime column used for filtering.

        duplicates (bool):
            Whether to keep duplicate rows inside each combined window.

    Returns:
        Tuple[pd.DataFrame, ...]:
            One dataframe per date-range group.
    """
    ensure_datetime(df, time_col)

    out_dfs = []

    for group_range in ranges_groups:
        df_sub = cut_by_date_ranges(df, group_range, time_col, duplicates)
        out_dfs.append(df_sub)

    return tuple(out_dfs)


def split_by_candles(
    df: pd.DataFrame,
    *candles: int,
    from_tail: bool = True,
    end_date: Optional[str] = None,
    time_col: str = "open_time_eth",
) -> Tuple[pd.DataFrame, ...]:
    """
    Splits a dataframe into multiple windows by candle counts.

    Example:
        split_by_candles(df, 10000, 2000, 3000)

    If from_tail=True, slicing starts from the most recent rows and moves backward.

    Args:
        df (pd.DataFrame):
            Input dataframe.

        *candles (int):
            Window sizes in rows / candles.

        from_tail (bool):
            Whether to start slicing from the end of the dataframe.

        end_date (Optional[str]):
            Optional upper time boundary before slicing.

        time_col (str):
            Datetime column used for sorting and optional truncation.

    Returns:
        Tuple[pd.DataFrame, ...]:
            One dataframe per requested candle window.
    """
    base = ensure_datetime(df.copy(), time_col)
    base = base.sort_values(time_col).reset_index(drop=True)

    if end_date is not None:
        base = base[base[time_col] <= pd.to_datetime(end_date, utc=True)]

    if from_tail:
        base = base.iloc[::-1].reset_index(drop=True)

    out = []
    idx = 0

    for n in candles:
        part = base.iloc[idx: idx + n].copy()
        idx += n

        if from_tail:
            part = part.iloc[::-1].reset_index(drop=True)

        out.append(part)

    return tuple(out)


def slice_data(
    df: pd.DataFrame,
    slicing_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, ...]:
    """
    Unified entry point for dataset slicing.

    Supported slicing modes:
    - "candles": split by row counts
    - "ranges": split by date ranges

    Expected config structure:
        {
            "type": "candles" | "ranges",
            "candles": [...]
            ...
        }

    Args:
        df (pd.DataFrame):
            Path or filename of the processed dataset.

        slicing_config (Dict[str, Any]):
            Slicing configuration dictionary.

    Returns:
        Tuple[pd.DataFrame, ...]:
            Tuple of sliced dataframes.
    """

    slice_type = slicing_config.pop("type")
    candles = slicing_config.pop("candles")

    if slice_type == "candles":
        windows = split_by_candles(df, *candles, **slicing_config)
    elif slice_type == "ranges":
        windows = split_by_date_ranges(df, *candles, **slicing_config)
    else:
        raise ValueError(f"Unknown slicing type: {slice_type}")

    return tuple(windows)
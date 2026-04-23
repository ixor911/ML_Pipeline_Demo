# target/filter.py

from functools import lru_cache
from typing import Optional, Tuple, List

import pandas as pd

from core.io.loader import load_config


# =========================================================
# Feature groups loader
# =========================================================

@lru_cache(maxsize=1)
def load_feature_groups() -> dict:
    """
    Loads feature groups configuration once and caches it.

    Returns:
        dict:
            Mapping from feature group name to list of column names.
    """
    return load_config("feature_groups")


def ordered_unique(items: List[str]) -> List[str]:
    """
    Returns a list of unique items while preserving the original order.

    Args:
        items (List[str]):
            Input list.

    Returns:
        List[str]:
            Ordered list without duplicates.
    """
    seen = set()
    out = []

    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)

    return out


def all_feature_group_columns() -> List[str]:
    """
    Collects all columns from all configured feature groups.

    Returns:
        List[str]:
            Ordered unique list of all feature columns.
    """
    groups = load_feature_groups()
    cols: List[str] = []

    for group_cols in groups.values():
        cols.extend(group_cols)

    return ordered_unique(cols)


# =========================================================
# Strict whitelist feature selection
# =========================================================

def resolve_feature_columns(
    df: pd.DataFrame,
    *,
    feature_groups: Optional[List[str]] = None,
    features_include: Optional[List[str]] = None,
    extra_drop: Optional[List[str]] = None,
) -> List[str]:
    """
    Resolves the final feature whitelist.

    Rules:
    1. feature_groups and features_include together define the whitelist
    2. if the whitelist is empty -> raise an error
    3. extra_drop removes columns from the whitelist
    4. only columns that actually exist in the dataframe are kept

    Args:
        df (pd.DataFrame):
            Input dataframe.

        feature_groups (Optional[List[str]]):
            List of feature group names.

        features_include (Optional[List[str]]):
            Additional explicitly included feature columns.

        extra_drop (Optional[List[str]]):
            Columns to remove after whitelist construction.

    Returns:
        List[str]:
            Final ordered list of feature columns.
    """
    groups_cfg = load_feature_groups()

    feature_groups = feature_groups or []
    features_include = features_include or []
    extra_drop = extra_drop or []

    selected: List[str] = []

    # Expand group-based feature lists.
    if feature_groups:
        if "all" in feature_groups:
            selected.extend(all_feature_group_columns())

        for group_name in feature_groups:
            if group_name == "all":
                continue

            if group_name not in groups_cfg:
                raise ValueError(
                    f"Unknown feature group: {group_name}. "
                    f"Available groups: {list(groups_cfg.keys()) + ['all']}"
                )

            selected.extend(groups_cfg[group_name])

    # Add manually included columns.
    selected.extend(features_include)

    # Remove duplicates while preserving order.
    selected = ordered_unique(selected)

    if not selected:
        raise ValueError(
            "No features selected. "
            "Provide at least one feature_groups or features_include."
        )

    # Apply explicit exclusions.
    if extra_drop:
        drop_set = set(extra_drop)
        selected = [c for c in selected if c not in drop_set]

    if not selected:
        raise ValueError(
            "All selected features were removed by extra_drop. "
            "Resulting feature set is empty."
        )

    # Keep only columns that actually exist in the dataframe.
    selected = [c for c in selected if c in df.columns]

    if not selected:
        raise ValueError(
            "None of selected features exist in dataframe columns. "
            "Check feature_groups / features_include names."
        )

    return selected


def keep_only_selected_features(
    df: pd.DataFrame,
    *,
    selected_features: List[str],
    future_ret_col: str,
) -> pd.DataFrame:
    """
    Keeps only the selected feature columns plus the target column.

    Args:
        df (pd.DataFrame):
            Input dataframe.

        selected_features (List[str]):
            Feature whitelist.

        future_ret_col (str):
            Target column name.

    Returns:
        pd.DataFrame:
            Reduced dataframe containing only selected features and target.
    """
    keep_cols = [c for c in selected_features if c in df.columns]

    if future_ret_col in df.columns and future_ret_col not in keep_cols:
        keep_cols.append(future_ret_col)

    return df[keep_cols].copy()


# =========================================================
# tau / deadzone
# =========================================================

def normalize_tau_list(
    tau_pct,
    n: int,
    param_name: str = "tau_pct"
):
    """
    Normalizes tau_pct into a list of length n.

    If a scalar value is provided, it is repeated n times.
    If a list is provided, its length must match n.

    Args:
        tau_pct:
            Scalar or list-like tau value(s).

        n (int):
            Number of windows.

        param_name (str):
            Parameter name used in error messages.

    Returns:
        list:
            Tau values normalized to length n.
    """
    if isinstance(tau_pct, list):
        if len(tau_pct) != n:
            raise ValueError(
                f"{param_name} list must match number of windows: "
                f"{len(tau_pct)} given, {n} required"
            )
        return tau_pct

    return [tau_pct for _ in range(n)]


def apply_deadzone(
    df: pd.DataFrame,
    future_ret_col: str,
    tau_pct: Optional[float]
) -> pd.DataFrame:
    """
    Applies a deadzone filter to the target.

    Rows are kept only if the absolute future return is greater than
    or equal to the specified tau threshold.

    Args:
        df (pd.DataFrame):
            Input dataframe.

        future_ret_col (str):
            Future return column used for filtering.

        tau_pct (Optional[float]):
            Threshold in percent. If None, no filtering is applied.

    Returns:
        pd.DataFrame:
            Filtered dataframe.
    """
    if tau_pct is None:
        return df.copy()

    tau = float(tau_pct) / 100.0
    mask = df[future_ret_col].abs() >= tau
    return df.loc[mask].copy()


def apply_deadzone_windows(
    *dfs: pd.DataFrame,
    future_ret_col: str,
    tau_list=None,
) -> Tuple[pd.DataFrame, ...]:
    """
    Applies deadzone filtering independently to multiple dataframes.

    Args:
        *dfs (pd.DataFrame):
            Input windows.

        future_ret_col (str):
            Future return column used for filtering.

        tau_list:
            Scalar, list, or None. If scalar, it is broadcast to all windows.
            If None, deadzone is disabled for all windows.

    Returns:
        Tuple[pd.DataFrame, ...]:
            Tuple of filtered dataframes.
    """
    if tau_list is None:
        tau_list = []

    n = len(dfs)

    if tau_list == []:
        tau_list = [None] * n
    else:
        tau_list = normalize_tau_list(tau_list, n)

    cleaned = []

    for df, tau_pct in zip(dfs, tau_list):
        df_clean = apply_deadzone(df, future_ret_col, tau_pct)
        cleaned.append(df_clean)

    return tuple(cleaned)


# =========================================================
# regime
# =========================================================

def apply_regime_filter(
    df: pd.DataFrame,
    allowed: Optional[List[str]],
    col: str = "regime"
) -> pd.DataFrame:
    """
    Filters dataframe rows by allowed regime values.

    Args:
        df (pd.DataFrame):
            Input dataframe.

        allowed (Optional[List[str]]):
            List of allowed regime values. If None, no filtering is applied.

        col (str):
            Regime column name.

    Returns:
        pd.DataFrame:
            Filtered dataframe.
    """
    if allowed is None:
        return df.copy()

    return df[df[col].isin(allowed)].copy()


def apply_regime_filter_windows(
    *dfs: pd.DataFrame,
    allowed: Optional[List[str]],
    col: str = "regime",
) -> Tuple[pd.DataFrame, ...]:
    """
    Applies regime filtering independently to multiple dataframes.

    Args:
        *dfs (pd.DataFrame):
            Input windows.

        allowed (Optional[List[str]]):
            Allowed regime values.

        col (str):
            Regime column name.

    Returns:
        Tuple[pd.DataFrame, ...]:
            Tuple of filtered dataframes.
    """
    cleaned = []

    for df in dfs:
        df_out = apply_regime_filter(df, allowed, col)
        cleaned.append(df_out)

    return tuple(cleaned)
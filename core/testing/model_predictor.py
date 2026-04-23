# core/testing/models_predictor.py

from __future__ import annotations

from typing import Dict
import pandas as pd

from core.models import ModelState


# ============================================================
# Predictions: single category
# ============================================================

def category_predictions(
    *,
    category_ram: Dict[str, ModelState],
    candles: pd.DataFrame,
    include_target: bool = False,
) -> pd.DataFrame:
    """
    Builds a prediction dataframe for one category of models.

    For each model in the category:
    - calls state.predict_frame(...)
    - appends service columns:
        model_id
        category
        thr

    Returns one combined dataframe for the whole category.
    """
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("category_predictions expects candles as pandas.DataFrame")

    rows = []

    for model_id, state in category_ram.items():
        part = state.predict_frame(
            candles=candles,
            include_target=include_target,
        )

        if part is None or part.empty:
            continue

        part = part.copy()
        part["model_id"] = state.model_id or model_id
        part["category"] = state.category
        part["thr"] = state.thr

        rows.append(part)

    if not rows:
        base_cols = ["row_idx", "proba", "pred", "model_id", "category", "thr"]
        if include_target:
            base_cols.insert(3, "target")
        return pd.DataFrame(columns=base_cols)

    out = pd.concat(rows, axis=0, ignore_index=True)

    # Keep the most important columns first for readability.
    preferred_order = ["row_idx", "proba", "pred"]
    if include_target and "target" in out.columns:
        preferred_order.append("target")
    preferred_order += ["model_id", "category", "thr"]

    existing = [c for c in preferred_order if c in out.columns]
    rest = [c for c in out.columns if c not in existing]

    return out[existing + rest]


# ============================================================
# Predictions: all categories as dict
# ============================================================

def categories_predictions(
    *,
    models_ram: Dict[str, Dict[str, ModelState]],
    candles: pd.DataFrame,
    include_target: bool = False,
    drop_empty: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Builds prediction dataframes for all categories.

    Returns:
        {
            "mcc": df,
            "signals10": df,
            ...
        }

    Parameters:
    - include_target:
        forwarded to state.predict_frame(...)
    - drop_empty:
        if True, categories with empty prediction dataframes are skipped
    """
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("categories_predictions expects candles as pandas.DataFrame")

    out: Dict[str, pd.DataFrame] = {}

    for category, category_ram in models_ram.items():
        df = category_predictions(
            category_ram=category_ram,
            candles=candles,
            include_target=include_target,
        )

        if drop_empty and df.empty:
            continue

        out[category] = df

    return out


# ============================================================
# Predictions: all categories as one flat dataframe
# ============================================================

def all_models_predictions(
    *,
    models_ram: Dict[str, Dict[str, ModelState]],
    candles: pd.DataFrame,
    include_target: bool = False,
) -> pd.DataFrame:
    """
    Builds one flat prediction dataframe across all models.

    Internally:
    - calls categories_predictions(...)
    - concatenates all category-level dataframes into one table

    This is useful when the next step needs a single unified view
    over the entire model pool.
    """
    predictions_by_category = categories_predictions(
        models_ram=models_ram,
        candles=candles,
        include_target=include_target,
        drop_empty=True,
    )

    if not predictions_by_category:
        base_cols = ["row_idx", "proba", "pred", "model_id", "category", "thr"]
        if include_target:
            base_cols.insert(3, "target")
        return pd.DataFrame(columns=base_cols)

    parts = list(predictions_by_category.values())
    out = pd.concat(parts, axis=0, ignore_index=True)

    preferred_order = ["row_idx", "proba", "pred"]
    if include_target and "target" in out.columns:
        preferred_order.append("target")
    preferred_order += ["model_id", "category", "thr"]

    existing = [c for c in preferred_order if c in out.columns]
    rest = [c for c in out.columns if c not in existing]

    return out[existing + rest]


# ============================================================
# Small helpers
# ============================================================

def count_active_predictions(predictions_df: pd.DataFrame) -> int:
    """
    Counts positive predictions (pred == 1) in a prediction dataframe.
    """
    if predictions_df.empty or "pred" not in predictions_df.columns:
        return 0

    return int((predictions_df["pred"] == 1).sum())


def build_row_predictions(
    *,
    predictions_df: pd.DataFrame,
    row_idx: int,
) -> pd.DataFrame:
    """
    Returns predictions for one specific candle row.

    Useful for future step-by-step simulation or backtest logic,
    where predictions need to be inspected one candle at a time.
    """
    if "row_idx" not in predictions_df.columns:
        raise KeyError("build_row_predictions: predictions_df must contain 'row_idx' column")

    return predictions_df[predictions_df["row_idx"] == row_idx].copy()
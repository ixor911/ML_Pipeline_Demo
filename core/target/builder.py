# target/Builder.py

import pandas as pd
from typing import Optional, List, Tuple

from core.target import slicing
from core.target import filter as ft


class Builder:
    """
    Main interface for building X and y from processed candle data.

    This class applies the target-building pipeline on top of an already
    prepared dataframe. Its design follows a strict whitelist approach:

    - features are selected only from feature_groups / features_include
    - extra_drop removes columns from the already selected whitelist
    - future_ret_col is used only for target construction,
      not for feature selection itself
    """

    slicing = slicing
    filter = ft

    @classmethod
    def build(
        cls,
        df: pd.DataFrame,
        *,
        horizon: int,
        tau_pct: Optional[float],
        extra_drop: Optional[List[str]] = None,
        future_ret_col: str = "future_ret",
        regime_filter: Optional[List[str]] = None,
        feature_groups: Optional[List[str]] = None,
        features_include: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Builds X and y for a single dataframe.
        """
        df2 = df.copy()
        df2[future_ret_col] = df2["close_eth"].shift(-horizon) / df2["close_eth"] - 1

        if regime_filter is not None:
            df2, = ft.apply_regime_filter_windows(
                df2,
                allowed=regime_filter,
            )

        df2, = ft.apply_deadzone_windows(
            df2,
            future_ret_col=future_ret_col,
            tau_list=tau_pct,
        )

        selected_features = ft.resolve_feature_columns(
            df2,
            feature_groups=feature_groups,
            features_include=features_include,
            extra_drop=extra_drop,
        )

        df2 = ft.keep_only_selected_features(
            df2,
            selected_features=selected_features,
            future_ret_col=future_ret_col,
        )

        df2 = df2.dropna().copy()

        if future_ret_col not in df2.columns:
            raise RuntimeError("future_ret_col is missing after filtering.")

        y = (df2[future_ret_col] > 0).astype(int)
        X = df2.drop(columns=[future_ret_col], errors="ignore")

        if X.empty:
            raise ValueError("Feature set is empty after whitelist selection.")

        return X, y

    @classmethod
    def build_windows(
        cls,
        *dfs: pd.DataFrame,
        horizon: int,
        tau_pct: Optional[List[Optional[float]]] = None,
        extra_drop: Optional[List[str]] = None,
        future_ret_col: str = "future_ret",
        regime_filter: Optional[List[str]] = None,
        feature_groups: Optional[List[str]] = None,
        features_include: Optional[List[str]] = None,
    ):
        """
        Batch version of build() for multiple windows.

        This method simply applies build() to each input dataframe and returns
        a flat tuple:
            (X1, y1, X2, y2, X3, y3, ...)
        """
        n = len(dfs)
        tau_list = ft.normalize_tau_list(tau_pct, n) if tau_pct is not None else [None] * n

        out = []

        for df, tau_one in zip(dfs, tau_list):
            X, y = cls.build(
                df,
                horizon=horizon,
                tau_pct=tau_one,
                extra_drop=extra_drop,
                future_ret_col=future_ret_col,
                regime_filter=regime_filter,
                feature_groups=feature_groups,
                features_include=features_include,
            )
            out.append(X)
            out.append(y)

        return tuple(out)
# core/features/preprocessor/Preprocessor.py

from __future__ import annotations

import os
from typing import Optional, Literal

import pandas as pd

# Import internal preprocessors.
from core.features.preprocessor.ETH import PreprocessorETH
from core.features.preprocessor.BTC import PreprocessorBTC
import paths


class Preprocessor:
    """
    Unified interface for ETH, BTC, and ETHBTC feature preprocessing.

    This class orchestrates individual asset-level preprocessors and merges
    their outputs into a single feature dataframe suitable for modeling.

    Exposed submodules:
    - ETH: ETHUSDT feature engineering
    - BTC: BTCUSDT + ETHBTC feature engineering
    """

    # Expose sub-preprocessors directly through the main interface.
    ETH = PreprocessorETH
    BTC = PreprocessorBTC

    @staticmethod
    def preprocess(
        eth_df: pd.DataFrame,
        btc_df: pd.DataFrame,
        ethbtc_df: Optional[pd.DataFrame] = None,
        *,
        window_anchor: int = 60,
        shock_k: float = 1.5,
        lead_lag: bool = True,
        join: Literal["inner", "left", "right"] = "inner",
        prefer_time: Literal["eth", "btc", "both"] = "eth",
        build_ethbtc_if_missing: bool = True,
        add_cross_features: bool = True,
        add_volume_pack: bool = True,
    ) -> pd.DataFrame:
        """
        Main preprocessing entry point for ETH + BTC (+ ETHBTC).

        This method runs the complete multi-asset preprocessing pipeline:
        - prepares ETH features
        - prepares BTC and ETHBTC features
        - optionally builds ETHBTC synthetically if it is missing
        - merges the feature sets in time
        - optionally adds simple ETH-vs-BTC cross features

        Pipeline overview:
        1. Build ETHBTC synthetically from ETH/BTC ratio if needed
        2. Run ETH-specific preprocessing
        3. Run BTC + ETHBTC preprocessing
        4. Merge asset-level features by time
        5. Optionally add cross-market features
        6. Remove any leftover duplicate merge artifacts

        Args:
            eth_df (pd.DataFrame):
                Raw ETHUSDT hourly OHLCV dataframe.

            btc_df (pd.DataFrame):
                Raw BTCUSDT hourly OHLCV dataframe.

            ethbtc_df (Optional[pd.DataFrame]):
                Raw ETHBTC hourly OHLCV dataframe.
                If not provided and build_ethbtc_if_missing=True,
                a synthetic ETHBTC series will be built from ETH/BTC close ratios.

            window_anchor (int):
                Number of initial rows to discard because rolling indicators
                need a warm-up period.

            shock_k (float):
                Multiplier used in BTC volatility shock detection.

            lead_lag (bool):
                Whether to include lagged BTC return features.

            join (Literal["inner", "left", "right"]):
                Merge strategy used when combining ETH and BTC feature sets.

            prefer_time (Literal["eth", "btc", "both"]):
                Controls which open_time column is preferred after merging.

            build_ethbtc_if_missing (bool):
                Whether to synthesize ETHBTC from ETHUSDT/BTCUSDT if ETHBTC is not provided.

            add_cross_features (bool):
                Whether to create additional ETH-vs-BTC relative features.

            add_volume_pack (bool):
                Whether to enable additional volume-related features
                inside the ETH and BTC preprocessors.

        Returns:
            pd.DataFrame:
                Final merged multi-asset feature dataframe.
        """

        # Build ETHBTC synthetically if the raw pair is not available.
        if ethbtc_df is None and build_ethbtc_if_missing:
            ethbtc_df = Preprocessor._build_ethbtc_from_ratio(eth_df, btc_df)

        # Run ETH-only preprocessing.
        eth_fx = Preprocessor.ETH.preprocess(
            eth_df,
            window_anchor=window_anchor,
            add_volume_pack=add_volume_pack,
        )

        # Run BTC preprocessing together with ETHBTC-derived relative-strength features.
        btc_fx = Preprocessor.BTC.preprocess(
            btc_df,
            ethbtc_df,
            window_anchor=window_anchor,
            shock_k=shock_k,
            lead_lag=lead_lag,
            add_volume_pack=add_volume_pack,
        )

        # Merge asset-level feature sets into a single dataframe.
        merged = Preprocessor._merge(eth_fx, btc_fx, how=join, prefer_time=prefer_time)

        # Optionally add simple cross-market features between ETH and BTC.
        if add_cross_features:
            merged = Preprocessor._add_cross_features(merged)

        # Drop any leftover duplicate columns from earlier ETHBTC joins.
        merged = merged.drop(
            columns=[c for c in merged.columns if c.endswith("ethbtcdup")],
            errors="ignore"
        )

        return merged

    @staticmethod
    def preprocess_and_save(
        eth_df: pd.DataFrame,
        btc_df: pd.DataFrame,
        ethbtc_df: Optional[pd.DataFrame] = None,
        *,
        months: int,
        interval: str = "1h",
        symbol: str = "ETHUSDT",  # target asset name used in the output filename
        out_dir: str = "data/processed",
        fmt: str = "parquet",  # 'parquet' | 'csv'
        window_anchor: int = 60,
        shock_k: float = 1.5,
        lead_lag: bool = True,
        join: Literal["inner", "left", "right"] = "inner",
        prefer_time: Literal["eth", "btc", "both"] = "eth",
        build_ethbtc_if_missing: bool = True,
        add_cross_features: bool = True,
        add_volume_pack: bool = True,
    ) -> pd.DataFrame:
        """
        Runs the full preprocessing pipeline and saves the result to disk.

        The output filename follows the pattern:
            {symbol}_{months}_{interval}.parquet
        or falls back to CSV if parquet saving fails.

        Args:
            eth_df (pd.DataFrame):
                Raw ETHUSDT dataframe.

            btc_df (pd.DataFrame):
                Raw BTCUSDT dataframe.

            ethbtc_df (Optional[pd.DataFrame]):
                Optional raw ETHBTC dataframe.

            months (int):
                Dataset span used for naming the output file.

            interval (str):
                Candle interval used for naming the output file.

            symbol (str):
                Main target symbol used in the output filename.

            out_dir (str):
                Directory where the processed dataset will be saved.

            fmt (str):
                Preferred save format: "parquet" or "csv".

        Returns:
            str:
                Path to the saved processed dataset.
        """
        # 1) Build the final feature dataframe.
        df = Preprocessor.preprocess(
            eth_df,
            btc_df,
            ethbtc_df,
            window_anchor=window_anchor,
            shock_k=shock_k,
            lead_lag=lead_lag,
            join=join,
            prefer_time=prefer_time,
            build_ethbtc_if_missing=build_ethbtc_if_missing,
            add_cross_features=add_cross_features,
            add_volume_pack=add_volume_pack,
        )

        # 2) Prepare output directory and filename.
        os.makedirs(out_dir, exist_ok=True)
        symbol = (symbol or "ETHUSDT").upper()
        base_name = f"{symbol}_{months}_{interval}"

        # 3) Save to parquet if requested, otherwise to CSV.
        save_path = os.path.join(
            out_dir,
            f"{base_name}.parquet" if fmt == "parquet" else f"{base_name}.csv"
        )

        try:
            if fmt == "parquet":
                df.to_parquet(save_path, index=False)
            else:
                df.to_csv(save_path, index=False)
        except Exception:
            # Fallback to CSV if parquet support is unavailable or fails.
            save_path = os.path.join(out_dir, f"{base_name}.csv")
            df.to_csv(save_path, index=False)

        return df

    # ---------- helpers ----------

    @staticmethod
    def _merge(
        eth: pd.DataFrame,
        btc: pd.DataFrame,
        how: str = "inner",
        prefer_time: str = "eth",
    ) -> pd.DataFrame:
        """
        Merges ETH and BTC feature dataframes on close_time.

        Args:
            eth (pd.DataFrame):
                ETH feature dataframe.

            btc (pd.DataFrame):
                BTC feature dataframe.

            how (str):
                Merge strategy passed to pandas.merge.

            prefer_time (str):
                Controls whether ETH or BTC open_time column should be kept
                when both are present.

        Returns:
            pd.DataFrame:
                Time-aligned merged dataframe.
        """
        merged = pd.merge(
            eth,
            btc,
            on="close_time",
            how=how,
            suffixes=("_eth", "_btc"),
        )

        if prefer_time == "eth" and "open_time_btc" in merged.columns:
            merged = merged.drop(columns=["open_time_btc"])
        if prefer_time == "btc" and "open_time_eth" in merged.columns:
            merged = merged.drop(columns=["open_time_eth"])

        return merged.sort_values("close_time").reset_index(drop=True)

    @staticmethod
    def _build_ethbtc_from_ratio(eth_df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds a synthetic ETHBTC series from ETHUSDT / BTCUSDT close prices.

        This helper is useful when raw ETHBTC candles are not available,
        but cross-market relative-strength features are still needed.

        Returns:
            pd.DataFrame:
                Minimal ETHBTC-like dataframe with open_time, close_time, and close.
        """
        eth = eth_df[["close_time", "close"]].rename(columns={"close": "eth_close"})
        btc = btc_df[["close_time", "close"]].rename(columns={"close": "btc_close"})
        df = pd.merge(eth, btc, on="close_time", how="inner")
        df["close"] = df["eth_close"] / df["btc_close"]
        df["open_time"] = df["close_time"]  # placeholder to satisfy downstream interface
        return df[["open_time", "close_time", "close"]]

    @staticmethod
    def _add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds simple cross-market ETH vs BTC features.

        Current cross features:
        - eth_ret_1h_pct
        - rs_1h_pct   : ETH 1h return minus BTC 1h return
        - rs_24h_pct  : ETH 24h return minus BTC 24h return

        Returns:
            pd.DataFrame:
                Dataframe with additional relative-strength features.
        """
        x = df.copy()

        if "close_eth" in x and "btc_ret_1h_pct" in x:
            x["eth_ret_1h_pct"] = (x["close_eth"] / x["close_eth"].shift(1) - 1.0) * 100
            x["rs_1h_pct"] = x["eth_ret_1h_pct"] - x["btc_ret_1h_pct"]

        if "ret_24h_pct" in x and "btc_ret_24h_pct" in x:
            x["rs_24h_pct"] = x["ret_24h_pct"] - x["btc_ret_24h_pct"]

        return x


if __name__ == "__main__":
    from core.features.DataProvider import DataProvider

    # ==========================================
    # CONFIG
    # ==========================================
    MONTHS = 6
    INTERVAL = "1h"

    USE_ETHBTC = True              # if False, ETHBTC will be built synthetically from ETH/BTC
    SAVE_DIR = paths.PROCESSED_DATA_DIR
    SAVE_FMT = "csv"               # "csv" | "parquet"

    WINDOW_ANCHOR = 60
    SHOCK_K = 1.5
    LEAD_LAG = True
    JOIN = "inner"
    PREFER_TIME = "eth"
    BUILD_ETHBTC_IF_MISSING = True
    ADD_CROSS_FEATURES = True
    ADD_VOLUME_PACK = True

    print("\n" + "=" * 80)
    print("START PREPROCESSOR PIPELINE")
    print("=" * 80)

    # ==========================================
    # 1) LOAD RAW DATA
    # ==========================================
    print("\n[1/3] Load raw candles...")

    eth_df = DataProvider.read_raw_symbol(
        symbol="ETHUSDT",
        months=MONTHS,
        interval=INTERVAL,
    )

    btc_df = DataProvider.read_raw_symbol(
        symbol="BTCUSDT",
        months=MONTHS,
        interval=INTERVAL,
    )

    ethbtc_df = None
    if USE_ETHBTC:
        ethbtc_df = DataProvider.read_raw_symbol(
            symbol="ETHBTC",
            months=MONTHS,
            interval=INTERVAL,
        )

    print(f"ETH rows   : {len(eth_df)}")
    print(f"BTC rows   : {len(btc_df)}")
    print(f"ETHBTC rows: {len(ethbtc_df) if ethbtc_df is not None else 'None (will build from ratio)'}")

    # ==========================================
    # 2) PREPROCESS + SAVE
    # ==========================================
    print("\n[2/3] Run Preprocessor.preprocess_and_save(...)...")

    save_path = Preprocessor.preprocess_and_save(
        eth_df=eth_df,
        btc_df=btc_df,
        ethbtc_df=ethbtc_df,
        months=MONTHS,
        interval=INTERVAL,
        symbol="ETHUSDT",
        out_dir=SAVE_DIR,
        fmt=SAVE_FMT,
        window_anchor=WINDOW_ANCHOR,
        shock_k=SHOCK_K,
        lead_lag=LEAD_LAG,
        join=JOIN,
        prefer_time=PREFER_TIME,
        build_ethbtc_if_missing=BUILD_ETHBTC_IF_MISSING,
        add_cross_features=ADD_CROSS_FEATURES,
        add_volume_pack=ADD_VOLUME_PACK,
    )

    # ==========================================
    # 3) DONE
    # ==========================================
    print("\n[3/3] Done.")
    print(f"Saved processed dataset to: {save_path}")

    print("\n" + "-" * 80)
    print("FINISH PREPROCESSOR PIPELINE")
    print("-" * 80)
    print("=" * 80 + "\n")
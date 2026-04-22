# core/features/preprocessor/ETH.py

from __future__ import annotations

import numpy as np
import pandas as pd


class PreprocessorETH:
    """
    Feature engineering pipeline for ETHUSDT 1H candles.

    This preprocessor converts raw ETH OHLCV market data into a richer feature set
    suitable for downstream ML models.

    Expected input columns:
        ['open_time', 'open', 'high', 'low', 'close', 'volume',
         'close_time', 'quote', 'trades', 'taker_buy_base', 'taker_buy_quote']

    Output:
        Original columns plus engineered technical, volume, candle structure,
        and regime-related features.

    By default, the first 59 rows are removed to ensure rolling indicators
    are fully formed when using a 60-bar anchor window.
    """

    @staticmethod
    def preprocess(
        df: pd.DataFrame,
        window_anchor: int = 60,
        add_volume_pack: bool = True,
    ) -> pd.DataFrame:
        """
        Main preprocessing entry point.

        This method runs the full ETH feature engineering pipeline in a fixed,
        sequential order. The goal is to transform raw hourly ETHUSDT candles
        into a clean, model-ready dataframe with trend, momentum, volatility,
        volume, candle microstructure, and market regime features.

        Pipeline overview:
        1. Validate and sort input data chronologically
        2. Cast base numeric columns to stable numeric types
        3. Compute smoothing and volatility indicators
        4. Compute return and volume-derived features
        5. Compute candle microstructure and range features
        6. Compute regime / market state features
        7. Trim warm-up rows required by rolling indicators
        8. Drop temporary helper columns
        9. Optionally add an extended volume feature pack

        Args:
            df (pd.DataFrame):
                Raw ETHUSDT hourly OHLCV dataframe.

            window_anchor (int):
                Number of rows required before the output starts.
                This is mainly used to remove early rows where rolling
                indicators are still warming up.

            add_volume_pack (bool):
                Whether to add additional volume-related features such as
                volume z-score, CMF, MFI, and volume/volatility interaction features.

        Returns:
            pd.DataFrame:
                Final engineered dataframe ready for downstream modeling.
        """
        if df.empty:
            return df.copy()

        # Ensure stable row order and consistent numeric types before feature generation.
        x = PreprocessorETH._ensure_sorted(df)
        x = PreprocessorETH._cast_types(x)

        # ============================================================
        # Core trend / smoothing / volatility indicators
        # ============================================================
        # These features describe broad market structure:
        # moving averages, MACD, RSI, ATR-based volatility, and Bollinger width.
        x = PreprocessorETH._add_sma_ema(x)
        x = PreprocessorETH._add_macd(x)
        x = PreprocessorETH._add_rsi14(x)
        x = PreprocessorETH._add_atr14_pct(x)
        x = PreprocessorETH._add_bollinger_width_pct(x)

        # ============================================================
        # Return and volume features
        # ============================================================
        # These features capture recent price movement and basic volume behavior.
        x = PreprocessorETH._add_returns(x)
        x = PreprocessorETH._add_volume_features(x)

        # ============================================================
        # Candle microstructure
        # ============================================================
        # These features describe candle anatomy, taker flow imbalance,
        # wick behavior, intrabar range, and relative price position.
        x = PreprocessorETH._add_microstructure(x)
        x = PreprocessorETH._add_wick_imbalance(x)
        x = PreprocessorETH._add_range_features(x)
        x = PreprocessorETH._add_pctB(x)

        # ============================================================
        # Regime / market state features
        # ============================================================
        # These features provide a simplified market context such as
        # trend direction, elevated volatility, and a legacy string regime label.
        x = PreprocessorETH._add_trend_sign(x)
        x = PreprocessorETH._add_is_high_vol(x)
        x = PreprocessorETH._add_regime_legacy(x)

        # Remove early rows where rolling features are not yet reliable,
        # then drop intermediate helper columns.
        x = PreprocessorETH._trim_window(x, window_anchor)
        x = PreprocessorETH._finalize_columns(x)

        # Optional extended volume pack with more advanced volume diagnostics.
        if add_volume_pack:
            x = PreprocessorETH._add_volume_pack(x)

        return x

    # ========= helpers =========

    @staticmethod
    def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values("open_time").reset_index(drop=True)

    @staticmethod
    def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        num_cols = [
            "open", "high", "low", "close", "volume",
            "quote", "taker_buy_base", "taker_buy_quote"
        ]
        for c in num_cols:
            x[c] = pd.to_numeric(x[c], errors="coerce")

        # "trades" may arrive as Int64/object, so cast to float for safe divisions.
        x["trades"] = pd.to_numeric(x["trades"], errors="coerce").astype("float64")
        return x

    @staticmethod
    def _rolling_std_pop(s: pd.Series, window: int, min_periods: int) -> pd.Series:
        # Population standard deviation (ddof=0).
        return s.rolling(window, min_periods=min_periods).std(ddof=0)

    @staticmethod
    def _add_sma_ema(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["sma20"] = x["close"].rolling(20, min_periods=20).mean()
        x["sma50"] = x["close"].rolling(50, min_periods=50).mean()
        x["sd20"] = PreprocessorETH._rolling_std_pop(x["close"], 20, 20)

        # EMA with adjust=False to match standard technical analysis behavior.
        x["ema12"] = x["close"].ewm(span=12, adjust=False).mean()
        x["ema26"] = x["close"].ewm(span=26, adjust=False).mean()
        return x

    @staticmethod
    def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["macd_line"] = x["ema12"] - x["ema26"]
        x["signal_line"] = x["macd_line"].ewm(span=9, adjust=False).mean()
        x["macd_hist"] = x["macd_line"] - x["signal_line"]
        return x

    @staticmethod
    def _add_rsi14(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        delta = x["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss
        x["rsi14"] = 100 - (100 / (1 + rs))
        x.loc[avg_loss == 0, "rsi14"] = 100.0
        return x

    @staticmethod
    def _add_atr14_pct(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        prev_close = x["close"].shift(1)
        tr = pd.concat([
            (x["high"] - x["low"]),
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs()
        ], axis=1).max(axis=1)

        x["atr14"] = tr.rolling(14, min_periods=14).mean()
        x["atr14_proxy_pct"] = (x["atr14"] / x["close"]) * 100.0
        return x

    @staticmethod
    def _add_bollinger_width_pct(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        # width% = 2 * sigma20 / SMA20 * 100
        x["bb_width_pct"] = (2 * x["sd20"] / x["sma20"]) * 100.0
        return x

    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes ETH return features in one place:
        - ret_1h_pct
        - ret_3h_pct
        - ret_6h_pct
        - ret_24h_pct
        - ret_acceleration
        """
        x = df.copy()

        x["ret_1h_pct"] = ((x["close"] / x["close"].shift(1)) - 1.0) * 100.0
        x["ret_3h_pct"] = ((x["close"] / x["close"].shift(3)) - 1.0) * 100.0
        x["ret_6h_pct"] = ((x["close"] / x["close"].shift(6)) - 1.0) * 100.0
        x["ret_24h_pct"] = ((x["close"] / x["close"].shift(24)) - 1.0) * 100.0

        # Short-term return acceleration.
        x["ret_acceleration"] = x["ret_1h_pct"] - x["ret_1h_pct"].shift(1)

        return x

    @staticmethod
    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den2 = den.replace(0, np.nan)
        return num.where(den2.notna(), np.nan) / den2

    @staticmethod
    def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["vol_ma20"] = x["volume"].rolling(20, min_periods=20).mean()
        x["vol_spike"] = x["volume"] > (1.5 * x["vol_ma20"])
        x["rvol20"] = x["volume"] / x["vol_ma20"]
        return x

    @staticmethod
    def _add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()

        # Taker buy ratio (0..1) and normalized imbalance (-1..1).
        x["taker_buy_ratio"] = PreprocessorETH._safe_div(x["taker_buy_quote"], x["quote"])
        x["taker_delta_norm"] = 2.0 * x["taker_buy_ratio"] - 1.0

        # Average trade size in quote currency.
        x["quote_per_trade"] = PreprocessorETH._safe_div(x["quote"], x["trades"])

        # Candle profile features.
        rng = (x["high"] - x["low"]).replace(0, np.nan)
        body = (x["close"] - x["open"]).abs()

        x["clv"] = (x["close"] - x["low"]) / (x["high"] - x["low"]).replace(0, np.nan)
        x["body_ratio"] = body / rng
        x["upper_wick_ratio"] = (x["high"] - x[["open", "close"]].max(axis=1)) / rng
        x["lower_wick_ratio"] = (x[["open", "close"]].min(axis=1) - x["low"]) / rng
        return x

    @staticmethod
    def _add_wick_imbalance(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["wick_imbalance"] = x["upper_wick_ratio"] - x["lower_wick_ratio"]
        return x

    @staticmethod
    def _add_range_features(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()

        # Absolute intrabar range in percentage terms.
        x["range_pct"] = ((x["high"] - x["low"]) / x["close"].replace(0, np.nan)) * 100.0

        # Average candle range over the last 10 bars.
        x["range_ma10"] = x["range_pct"].rolling(10, min_periods=10).mean()

        # Compression / expansion relative to recent norm.
        x["range_compression"] = x["range_pct"] / x["range_ma10"].replace(0, np.nan)

        return x

    @staticmethod
    def _add_pctB(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        # %B = (close - (SMA20 - 2σ)) / (4σ)
        denom = (4 * x["sd20"]).replace(0, np.nan)
        x["pctB"] = (x["close"] - (x["sma20"] - 2 * x["sd20"])) / denom
        return x

    @staticmethod
    def _add_trend_sign(df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes simplified trend direction:
            +1: bullish trend
            -1: bearish trend
             0: neutral / mixed
        """
        x = df.copy()

        up = (
            x["sma20"].notna() & x["sma50"].notna() & x["ret_24h_pct"].notna() &
            (x["sma20"] > x["sma50"]) & (x["ret_24h_pct"] > 0.5)
        )
        down = (
            x["sma20"].notna() & x["sma50"].notna() & x["ret_24h_pct"].notna() &
            (x["sma20"] < x["sma50"]) & (x["ret_24h_pct"] < -0.5)
        )

        x["trend_sign"] = np.select([up, down], [1, -1], default=0).astype("int8")
        return x

    @staticmethod
    def _add_is_high_vol(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flags elevated volatility regime:
            1 if atr14_proxy_pct > 1.2 else 0
        """
        x = df.copy()
        hv = x["atr14_proxy_pct"].notna() & (x["atr14_proxy_pct"] > 1.2)
        x["is_high_vol"] = hv.astype("int8")
        return x

    @staticmethod
    def _add_regime_legacy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds a legacy string-based market regime label for compatibility.
        """
        x = df.copy()
        conds = [
            (
                x["sma20"].notna() & x["sma50"].notna() & x["ret_24h_pct"].notna() &
                (x["sma20"] > x["sma50"]) & (x["ret_24h_pct"] > 0.5)
            ),
            (
                x["sma20"].notna() & x["sma50"].notna() & x["ret_24h_pct"].notna() &
                (x["sma20"] < x["sma50"]) & (x["ret_24h_pct"] < -0.5)
            ),
            (
                x["atr14_proxy_pct"].notna() & (x["atr14_proxy_pct"] > 1.2)
            ),
        ]
        choices = ["trend_up", "trend_down", "high_volatility"]
        x["regime"] = np.select(conds, choices, default="range").astype("object")
        return x

    @staticmethod
    def _trim_window(df: pd.DataFrame, window_anchor: int) -> pd.DataFrame:
        x = df.copy()
        if len(x) >= window_anchor:
            x = x.iloc[window_anchor - 1:].reset_index(drop=True)
        return x

    @staticmethod
    def _finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()

        # Drop intermediate helper columns that are no longer needed downstream.
        drop_cols = ["sd20", "atr14", "vol_ma20", "range_ma10"]
        x = x.drop(columns=[c for c in drop_cols if c in x.columns], errors="ignore")
        return x

    # ===== helpers: volume pack =====

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int = 24, lag: int = 1) -> pd.Series:
        """
        Rolling z-score without future leakage.

        Mean and standard deviation are computed only from past values,
        excluding the current one when lag=1.
        """
        s = series.astype(float)
        mu = s.shift(lag).rolling(window=window, min_periods=window).mean()
        sd = s.shift(lag).rolling(window=window, min_periods=window).std(ddof=0)
        z = (s - mu) / sd
        return z

    @staticmethod
    def _cmf20(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Chaikin Money Flow (CMF, 20).

        CMF = sum(CLV * Volume, N) / sum(Volume, N)
        """
        h = high.astype(float)
        l = low.astype(float)
        c = close.astype(float)
        v = volume.astype(float)

        hl_range = (h - l).replace(0, np.nan)
        clv = ((c - l) - (h - c)) / hl_range
        mfv = clv * v

        num = mfv.rolling(window=window, min_periods=window).sum()
        den = v.rolling(window=window, min_periods=window).sum()
        cmf = num / den
        return cmf

    @staticmethod
    def _mfi14(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """
        Money Flow Index (MFI, 14).
        """
        h = high.astype(float)
        l = low.astype(float)
        c = close.astype(float)
        v = volume.astype(float)

        tp = (h + l + c) / 3.0
        rmf = tp * v

        up = (tp > tp.shift(1)).astype(float)
        dn = (tp < tp.shift(1)).astype(float)

        pos_mf = (rmf * up).rolling(window=window, min_periods=window).sum()
        neg_mf = (rmf * dn).rolling(window=window, min_periods=window).sum()

        mfr = pos_mf / neg_mf.replace(0, np.nan)
        mfi = 100.0 - 100.0 / (1.0 + mfr)
        mfi = mfi.clip(0, 100)
        return mfi

    @staticmethod
    def _add_volume_pack(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds an extended set of volume-related features on top of
        already computed ETH features.

        Output features:
            - vol_ret_1h_pct
            - vol_zscore_24
            - cmf20
            - mfi14
            - vol_vola_coupling
            - rvol_anom
            - rvol_anom_signed
        """
        if not {"volume", "high", "low", "close"}.issubset(df.columns):
            return df

        # 1) Hourly volume change in percentage.
        df["vol_ret_1h_pct"] = df["volume"].pct_change() * 100.0

        # 2) Rolling volume z-score (24h, no leakage).
        df["vol_zscore_24"] = PreprocessorETH._rolling_zscore(df["volume"], window=24, lag=1)

        # 3) CMF(20).
        df["cmf20"] = PreprocessorETH._cmf20(
            df["high"], df["low"], df["close"], df["volume"], window=20
        )

        # 4) MFI(14).
        df["mfi14"] = PreprocessorETH._mfi14(
            df["high"], df["low"], df["close"], df["volume"], window=14
        )

        # 5) Volume-volatility interaction: rvol20 * atr14_proxy_pct.
        if {"rvol20", "atr14_proxy_pct"}.issubset(df.columns):
            df["vol_vola_coupling"] = df["rvol20"] * (df["atr14_proxy_pct"] / 100.0)

        # 6) Relative volume anomaly and signed anomaly.
        if "rvol20" in df.columns:
            df["rvol_anom"] = df["rvol20"] - 1.0
        else:
            df["rvol_anom"] = np.nan

        # Signed anomaly using return direction.
        if "eth_ret_1h_pct" in df.columns:
            sign_ret = np.sign(df["eth_ret_1h_pct"])
        else:
            sign_ret = np.sign(np.log(df["close"] / df["close"].shift(1)) * 100.0)

        df["rvol_anom_signed"] = sign_ret * df["rvol_anom"]

        return df
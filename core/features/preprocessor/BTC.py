# core/features/preprocessor/BTC.py

from __future__ import annotations

import pandas as pd


class PreprocessorBTC:
    """
    Feature engineering pipeline for BTCUSDT and ETHBTC (1H timeframe).

    This preprocessor transforms raw BTCUSDT and ETHBTC OHLCV data into a
    unified feature set suitable for downstream ML models.

    Main responsibilities:
    - compute BTC-based technical features
    - compute ETHBTC relative-strength features
    - align both datasets by time
    - optionally add BTC volume-derived features
    """

    @staticmethod
    def preprocess(
        btc_df: pd.DataFrame,
        ethbtc_df: pd.DataFrame,
        window_anchor: int = 60,
        shock_k: float = 1.5,
        lead_lag: bool = True,
        add_volume_pack: bool = True,
    ) -> pd.DataFrame:
        """
        Main preprocessing entry point.

        This method runs the full BTC + ETHBTC feature engineering pipeline and
        returns a single merged dataframe ready for modeling.

        Pipeline overview:
        1. Validate both input datasets
        2. Sort rows chronologically
        3. Cast numeric columns to stable numeric types
        4. Build BTC trend / momentum / volatility / return features
        5. Build ETHBTC relative-strength features
        6. Remove warm-up rows required by rolling indicators
        7. Merge BTC and ETHBTC by close_time
        8. Drop helper / duplicate columns and reorder output
        9. Optionally add BTC volume-pack features

        Args:
            btc_df (pd.DataFrame):
                Raw BTCUSDT hourly OHLCV dataframe.

            ethbtc_df (pd.DataFrame):
                Raw ETHBTC hourly OHLCV dataframe used for cross-market context.

            window_anchor (int):
                Number of rows required before the output starts.
                Used to drop early rows where rolling indicators are still warming up.

            shock_k (float):
                Multiplier used for volatility-based shock detection.

            lead_lag (bool):
                Whether to include lagged BTC return features.

            add_volume_pack (bool):
                Whether to add additional BTC volume-derived features.

        Returns:
            pd.DataFrame:
                Final merged feature dataframe aligned by time.
        """

        # Validate inputs before any processing starts.
        if btc_df is None or btc_df.empty:
            raise ValueError("btc_df is empty")
        if ethbtc_df is None or ethbtc_df.empty:
            raise ValueError("ethbtc_df is empty")

        # Ensure both datasets are ordered consistently in time.
        b = PreprocessorBTC._ensure_sorted(btc_df)
        e = PreprocessorBTC._ensure_sorted(ethbtc_df)

        # Normalize base numeric columns for stable downstream calculations.
        b = PreprocessorBTC._cast_base_types(b)
        e = PreprocessorBTC._cast_base_types(e)

        # ============================================================
        # BTC feature engineering
        # ============================================================
        # These features describe BTC market structure directly:
        # moving averages, MACD, RSI, volatility proxies, returns, and shocks.
        b = PreprocessorBTC._add_btc_sma_ema(b)
        b = PreprocessorBTC._add_btc_macd(b)
        b = PreprocessorBTC._add_btc_rsi14(b)
        b = PreprocessorBTC._add_btc_atr14_pct(b)
        b = PreprocessorBTC._add_btc_bb_width_pct(b)
        b = PreprocessorBTC._add_btc_returns_and_shock(
            b,
            shock_k=shock_k,
            lead_lag=lead_lag
        )

        # ============================================================
        # ETHBTC feature engineering
        # ============================================================
        # These features provide relative-strength context between ETH and BTC,
        # which can be useful as cross-market input for ETH-related modeling.
        e = PreprocessorBTC._add_ethbtc_signals(e)

        # Remove early rows where rolling indicators are not yet reliable.
        b = PreprocessorBTC._trim_window(b, window_anchor)
        e = PreprocessorBTC._trim_window(e, window_anchor)

        # Merge both datasets using close_time alignment.
        out = PreprocessorBTC._merge_on_close_time(b, e)

        # Drop intermediate helper columns and reorder final output columns.
        out = PreprocessorBTC._finalize_columns(out)

        # Optionally add BTC volume-based diagnostics after the main merge.
        if add_volume_pack:
            out = PreprocessorBTC._add_volume_pack(out)

        return out

    @staticmethod
    def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values('open_time').reset_index(drop=True)

    @staticmethod
    def _cast_base_types(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        num_cols = ['open', 'high', 'low', 'close', 'volume', 'quote', 'taker_buy_base', 'taker_buy_quote']
        for c in num_cols:
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors='coerce')
        if 'trades' in x.columns:
            x['trades'] = pd.to_numeric(x['trades'], errors='coerce').astype('float64')
        return x

    @staticmethod
    def _rolling_std_pop(s: pd.Series, window: int, min_periods: int) -> pd.Series:
        return s.rolling(window, min_periods=min_periods).std(ddof=0)

    @staticmethod
    def _add_btc_sma_ema(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x['btc_sma20'] = x['close'].rolling(20, min_periods=20).mean()
        x['btc_sma50'] = x['close'].rolling(50, min_periods=50).mean()
        x['btc_sd20'] = PreprocessorBTC._rolling_std_pop(x['close'], 20, 20)
        x['btc_ema12'] = x['close'].ewm(span=12, adjust=False).mean()
        x['btc_ema26'] = x['close'].ewm(span=26, adjust=False).mean()
        return x

    @staticmethod
    def _add_btc_macd(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x['btc_macd_line'] = x['btc_ema12'] - x['btc_ema26']
        x['btc_signal_line'] = x['btc_macd_line'].ewm(span=9, adjust=False).mean()
        x['btc_macd_hist'] = x['btc_macd_line'] - x['btc_signal_line']
        return x

    @staticmethod
    def _add_btc_rsi14(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        delta = x['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss
        x['btc_rsi14'] = 100 - (100 / (1 + rs))
        x.loc[avg_loss == 0, 'btc_rsi14'] = 100.0
        return x

    @staticmethod
    def _add_btc_atr14_pct(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        prev_close = x['close'].shift(1)
        tr = pd.concat([
            (x['high'] - x['low']),
            (x['high'] - prev_close).abs(),
            (x['low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        x['btc_atr14'] = tr.rolling(14, min_periods=14).mean()
        x['btc_atr14_proxy_pct'] = (x['btc_atr14'] / x['close']) * 100.0
        return x

    @staticmethod
    def _add_btc_bb_width_pct(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x['btc_bb_width_pct'] = (2 * x['btc_sd20'] / x['btc_sma20']) * 100.0
        return x

    @staticmethod
    def _add_btc_returns_and_shock(
        df: pd.DataFrame,
        shock_k: float = 1.5,
        lead_lag: bool = True
    ) -> pd.DataFrame:
        x = df.copy()
        x['btc_ret_1h_pct'] = (x['close'] / x['close'].shift(1) - 1.0) * 100.0
        x['btc_ret_24h_pct'] = (x['close'] / x['close'].shift(24) - 1.0) * 100.0

        sigma24 = PreprocessorBTC._rolling_std_pop(x['btc_ret_1h_pct'], 24, 24)
        x['btc_shock_24h'] = (x['btc_ret_1h_pct'].abs() > (shock_k * sigma24))

        if lead_lag:
            x['btc_ret_1h_lag1'] = x['btc_ret_1h_pct'].shift(1)
            x['btc_ret_1h_lag2'] = x['btc_ret_1h_pct'].shift(2)
            x['btc_ret_1h_lag3'] = x['btc_ret_1h_pct'].shift(3)

        return x

    @staticmethod
    def _add_ethbtc_signals(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x['ethbtc_ema12'] = x['close'].ewm(span=12, adjust=False).mean()
        x['ethbtc_ema26'] = x['close'].ewm(span=26, adjust=False).mean()
        x['ethbtc_macd_line'] = x['ethbtc_ema12'] - x['ethbtc_ema26']
        x['ethbtc_signal_line'] = x['ethbtc_macd_line'].ewm(span=9, adjust=False).mean()
        x['ethbtc_macd_hist'] = x['ethbtc_macd_line'] - x['ethbtc_signal_line']

        delta = x['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss
        x['ethbtc_rsi14'] = 100 - (100 / (1 + rs))
        x.loc[avg_loss == 0, 'ethbtc_rsi14'] = 100.0

        x['ethbtc_ret_1h_pct'] = (x['close'] / x['close'].shift(1) - 1.0) * 100.0
        x['ethbtc_ret_24h_pct'] = (x['close'] / x['close'].shift(24) - 1.0) * 100.0
        return x

    @staticmethod
    def _trim_window(df: pd.DataFrame, window_anchor: int) -> pd.DataFrame:
        x = df.copy()
        if len(x) >= window_anchor:
            x = x.iloc[window_anchor - 1:].reset_index(drop=True)
        return x

    @staticmethod
    def _merge_on_close_time(btc: pd.DataFrame, ethbtc: pd.DataFrame) -> pd.DataFrame:
        e = ethbtc.copy()
        e = e.rename(columns={'open_time': 'ethbtc_open_time', 'close_time': 'ethbtc_close_time'})

        out = pd.merge(
            btc, e,
            left_on='close_time',
            right_on='ethbtc_close_time',
            how='inner',
            suffixes=('', '_ethbtcdup')
        )

        drop_cols = ['ethbtc_close_time']
        out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors='ignore')
        return out

    @staticmethod
    def _finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()

        # Remove temporary helper columns and duplicated merge artifacts.
        drop_cols = ['btc_sd20', 'btc_atr14']
        dup_cols = ['close_ethbtcdup', 'open_ethbtcdup', 'high_ethbtcdup', 'low_ethbtcdup']
        x = x.drop(columns=[c for c in drop_cols + dup_cols if c in x.columns], errors='ignore')

        # Keep time columns first, then sort the rest for a stable output layout.
        time_cols = [c for c in ['open_time', 'close_time', 'ethbtc_open_time'] if c in x.columns]
        other_cols = [c for c in x.columns if c not in time_cols]
        other_cols = sorted(other_cols)

        ordered = time_cols + other_cols
        x = x[ordered]

        return x

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int = 24, lag: int = 1) -> pd.Series:
        """
        Rolling z-score without future leakage.

        Mean and standard deviation are computed only from past values,
        excluding the current bar when lag=1.
        """
        s = series.astype(float)
        mu = s.shift(lag).rolling(window=window, min_periods=window).mean()
        sd = s.shift(lag).rolling(window=window, min_periods=window).std(ddof=0)
        return (s - mu) / sd

    @staticmethod
    def _add_volume_pack(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a compact BTC volume feature pack after the BTC/ETHBTC merge.

        Output features:
            - btc_rvol20
            - btc_vol_zscore_24
            - btc_vol_vola_coupling
        """
        cols = set(df.columns)
        need = {"volume_btc"}
        if not need.issubset(cols):
            return df

        v = df["volume_btc"].astype(float)

        sma20_prev = v.shift(1).rolling(window=20, min_periods=20).mean()
        df["btc_rvol20"] = v / sma20_prev
        df["btc_vol_zscore_24"] = PreprocessorBTC._rolling_zscore(v, window=24, lag=1)

        if {"btc_atr14_proxy_pct"}.issubset(cols):
            df["btc_vol_vola_coupling"] = df["btc_rvol20"] * (df["btc_atr14_proxy_pct"] / 100.0)

        return df
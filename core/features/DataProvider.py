# core/features/DataProvider.py

from __future__ import annotations

import time
import requests
import pandas as pd
from typing import Union, Optional
import os
import json
import paths


class DataProvider:
    """
    Static Binance kline data provider for spot market candles.

    This class is designed as a lightweight utility for downloading,
    reading, and converting Binance OHLCV data into a consistent
    pandas DataFrame format used throughout the project.

    Supported public methods:
    - get_ethusdt_1h(...)
    - get_btcusdt_1h(...)
    - get_ethbtc_1h(...)
    - download_klines_months(...)

    Standard output format:
        ['open_time', 'open', 'high', 'low', 'close', 'volume',
         'close_time', 'quote', 'trades', 'taker_buy_base', 'taker_buy_quote']

    Column conventions:
    - time columns: UTC pandas.Timestamp
    - price / volume columns: float
    - trades: nullable integer (Int64)
    """

    BASE_URL = "https://api.binance.com"
    ENDPOINT = "/api/v3/klines"
    INTERVAL = "1h"
    MAX_LIMIT = 1000

    # ---------- Public methods ----------

    @staticmethod
    def get_ethusdt_1h(start: Union[str, int, pd.Timestamp],
                       end: Union[str, int, pd.Timestamp],
                       drop_incomplete: bool = True,
                       tz: str = "Europe/Berlin",
                       session: Optional[requests.Session] = None) -> pd.DataFrame:
        """
        Downloads ETHUSDT 1-hour candles for the requested time range.
        """
        return DataProvider._get_symbol_1h("ETHUSDT", start, end, drop_incomplete, tz, session)

    @staticmethod
    def get_btcusdt_1h(start: Union[str, int, pd.Timestamp],
                       end: Union[str, int, pd.Timestamp],
                       drop_incomplete: bool = True,
                       tz: str = "Europe/Berlin",
                       session: Optional[requests.Session] = None) -> pd.DataFrame:
        """
        Downloads BTCUSDT 1-hour candles for the requested time range.
        """
        return DataProvider._get_symbol_1h("BTCUSDT", start, end, drop_incomplete, tz, session)

    @staticmethod
    def get_ethbtc_1h(start: Union[str, int, pd.Timestamp],
                      end: Union[str, int, pd.Timestamp],
                      drop_incomplete: bool = True,
                      tz: str = "Europe/Berlin",
                      session: Optional[requests.Session] = None) -> pd.DataFrame:
        """
        Downloads ETHBTC 1-hour candles for the requested time range.
        """
        # Binance uses the symbol ETHBTC for the ETH/BTC pair.
        return DataProvider._get_symbol_1h("ETHBTC", start, end, drop_incomplete, tz, session)

    @staticmethod
    def download_klines_months(
        symbol: str,
        months: int,
        interval: str = "1h",
        tz: str = "Europe/Berlin",
        drop_incomplete: bool = True,
        save_dir: str = "data/raw",
        session: Optional[requests.Session] = None
    ) -> pd.DataFrame:
        """
        Downloads historical klines for the last N months, saves the raw Binance
        response to JSON, and returns a converted DataFrame.

        The request is paginated internally in chunks of up to 1000 candles.

        Saved file format:
            data/raw/{symbol}_{months}_{interval}.json

        Example:
            data/raw/ETHUSDT_6_1h.json

        Args:
            symbol (str):
                Trading pair symbol, e.g. "ETHUSDT".

            months (int):
                Number of months of historical data to download.

            interval (str):
                Binance kline interval, e.g. "1h".

            tz (str):
                Local timezone used to compute the start timestamp.

            drop_incomplete (bool):
                Whether to remove a potentially unfinished last candle.

            save_dir (str):
                Directory where raw JSON should be stored.

            session (Optional[requests.Session]):
                Optional reusable requests session.

        Returns:
            pd.DataFrame:
                Converted and time-sorted OHLCV dataframe.
        """
        if months <= 0:
            raise ValueError("months must be positive integer")
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be non-empty string")

        symbol = symbol.upper()

        # End boundary: current UTC time.
        now_utc = pd.Timestamp.now('UTC')
        end_ms = int(now_utc.value // 10 ** 6)

        # Start boundary: "now - N months" computed in local timezone, then converted to UTC.
        start_local = pd.Timestamp.now(tz=tz) - pd.DateOffset(months=months)
        start_ms = int(start_local.tz_convert("UTC").value // 10 ** 6)

        if end_ms <= start_ms:
            raise ValueError("Computed end <= start; check months / system time")

        # Download all candles through the internal paginator.
        klines = DataProvider._fetch_klines_all(symbol, interval, start_ms, end_ms, session)

        # Optionally remove a potentially incomplete last candle.
        if drop_incomplete and klines:
            now_ms = int(time.time() * 1000)
            klines = [row for row in klines if isinstance(row, list) and len(row) >= 7 and int(row[6]) <= now_ms]

        # Save raw Binance klines exactly as returned by the API.
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{symbol}_{months}_{interval}.json"
        fpath = os.path.join(save_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(klines, f, ensure_ascii=False)

        # Convert raw klines to the internal DataFrame format.
        df = DataProvider._klines_to_df(klines)
        return df.sort_values("open_time").reset_index(drop=True)

    @staticmethod
    def read_raw_json_to_df(path: str, drop_incomplete: bool = True, sort_by_time: bool = True) -> pd.DataFrame:
        """
        Reads a raw JSON file containing Binance klines (list-of-lists format),
        optionally removes unfinished candles, and converts the result into the
        project's standard OHLCV DataFrame format.

        Args:
            path (str):
                Path to a raw JSON file, for example:
                data/raw/ETHUSDT_24_1h.json

            drop_incomplete (bool):
                Whether to remove candles whose close_time is still in the future.

            sort_by_time (bool):
                Whether to sort the final dataframe by open_time.

        Returns:
            pd.DataFrame:
                Standardized OHLCV dataframe with columns:
                open_time, open, high, low, close, volume,
                close_time, quote, trades, taker_buy_base, taker_buy_quote
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw JSON file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            klines = json.load(f)

        if not isinstance(klines, list):
            raise ValueError("Raw JSON must be a list (Binance klines format).")

        # Optionally remove unfinished candles.
        if drop_incomplete and klines:
            now_ms = int(time.time() * 1000)
            klines = [
                row for row in klines
                if isinstance(row, list) and len(row) >= 7 and int(row[6]) <= now_ms
            ]

        df = DataProvider._klines_to_df(klines)

        if sort_by_time and not df.empty:
            df = df.sort_values("open_time").reset_index(drop=True)

        return df

    @staticmethod
    def read_raw_symbol(symbol: str, months: int, interval: str = "1h", **kwargs) -> pd.DataFrame:
        """
        Convenience wrapper that builds the expected raw JSON path and
        reads it through read_raw_json_to_df().

        Expected file location:
            data/raw/{symbol}_{months}_{interval}.json

        Args:
            symbol (str):
                Trading pair symbol, e.g. "ETHUSDT".

            months (int):
                Number of months used in the file naming convention.

            interval (str):
                Kline interval, e.g. "1h".

            **kwargs:
                Forwarded to read_raw_json_to_df(),
                for example drop_incomplete=True, sort_by_time=True.

        Returns:
            pd.DataFrame:
                Standardized OHLCV dataframe.
        """
        symbol = symbol.upper()
        path = paths.RAW_DATA_DIR / f"{symbol}_{months}_{interval}.json"

        return DataProvider.read_raw_json_to_df(path, **kwargs)

    # ---------- Internal utilities ----------

    @staticmethod
    def _get_symbol_1h(
        symbol: str,
        start: Union[str, int, pd.Timestamp],
        end: Union[str, int, pd.Timestamp],
        drop_incomplete: bool,
        tz: str,
        session: Optional[requests.Session]
    ) -> pd.DataFrame:
        """
        Internal helper for downloading a specific symbol on the default 1H interval.
        """
        start_ms = DataProvider._to_ms_utc(start, tz, floor_to_hour=True)
        end_ms = DataProvider._to_ms_utc(end, tz, ceil_to_hour=True)

        if end_ms <= start_ms:
            raise ValueError("end must be greater than start")

        klines = DataProvider._fetch_klines_all(symbol, DataProvider.INTERVAL, start_ms, end_ms, session)
        df = DataProvider._klines_to_df(klines)

        if df.empty:
            return df

        # Remove a potentially incomplete candle if the range touches "now".
        if drop_incomplete:
            now_ms = int(time.time() * 1000)
            df = df[df["close_time"].astype("int64") // 10**6 <= now_ms]
            df = df[df["close_time"] <= pd.to_datetime(end_ms, unit="ms", utc=True)]

        # Enforce the requested time range strictly.
        df = df[(df["open_time"] >= pd.to_datetime(start_ms, unit="ms", utc=True)) &
                (df["close_time"] <= pd.to_datetime(end_ms, unit="ms", utc=True))]

        return df.sort_values("open_time").reset_index(drop=True)

    @staticmethod
    def _fetch_klines_all(symbol: str, interval: str, start_ms: int, end_ms: int,
                          session: Optional[requests.Session]) -> list:
        """
        Downloads all klines for a symbol/interval/time range using Binance pagination.

        Binance returns at most MAX_LIMIT candles per request, so this method
        repeatedly requests new batches until the full range is covered.
        """
        sess = session or requests.Session()
        out = []
        cur_start = start_ms

        while cur_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cur_start,
                "endTime": end_ms,
                "limit": DataProvider.MAX_LIMIT,
            }
            r = sess.get(DataProvider.BASE_URL + DataProvider.ENDPOINT, params=params, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"Binance API error {r.status_code}: {r.text}")
            batch = r.json()
            if not isinstance(batch, list):
                raise RuntimeError(f"Unexpected response: {batch}")

            if not batch:
                break

            out.extend(batch)

            last_open = batch[-1][0]
            cur_start = last_open + DataProvider._interval_ms(interval)

            # If fewer than MAX_LIMIT candles were returned, there is nothing left to fetch.
            if len(batch) < DataProvider.MAX_LIMIT:
                break

        return out

    @staticmethod
    def _interval_ms(interval: str) -> int:
        """
        Converts a Binance interval string into milliseconds.
        """
        unit = interval[-1]
        val = int(interval[:-1])
        if unit == "m":
            return val * 60 * 1000
        if unit == "h":
            return val * 60 * 60 * 1000
        if unit == "d":
            return val * 24 * 60 * 60 * 1000
        raise ValueError(f"Unsupported interval: {interval}")

    @staticmethod
    def _klines_to_df(klines: list) -> pd.DataFrame:
        """
        Converts raw Binance klines (list-of-lists format) into the project's
        standard pandas DataFrame schema.
        """
        if not klines:
            return pd.DataFrame(columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote", "trades", "taker_buy_base", "taker_buy_quote"
            ])

        df = pd.DataFrame(klines, columns=[
            "open_time_ms", "open", "high", "low", "close", "volume",
            "close_time_ms", "quote", "trades", "taker_buy_base", "taker_buy_quote", "_ignore"
        ])

        # Cast numeric columns to stable numeric types.
        num_cols = ["open", "high", "low", "close", "volume", "quote", "taker_buy_base", "taker_buy_quote"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
        df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)

        # Remove raw timestamp columns and Binance's unused trailing field.
        df = df.drop(columns=["open_time_ms", "close_time_ms", "_ignore"])

        # Reorder into the standard project-wide schema.
        df = df[[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote", "trades", "taker_buy_base", "taker_buy_quote"
        ]]
        return df

    @staticmethod
    def _to_ms_utc(
        t: Union[str, int, pd.Timestamp],
        tz: str,
        floor_to_hour: bool = False,
        ceil_to_hour: bool = False
    ) -> int:
        """
        Converts a timestamp-like input into UTC milliseconds.

        If a naive timestamp is provided, it is assumed to belong to the given local timezone.
        Optional floor/ceil alignment can be applied for hourly boundaries.
        """
        if isinstance(t, int):
            return t  # already in milliseconds

        ts = pd.to_datetime(t, errors="raise")
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz).tz_convert("UTC")
        else:
            ts = ts.tz_convert("UTC")

        if floor_to_hour:
            ts = ts.floor("h")
        if ceil_to_hour:
            ts = ts.ceil("h")

        return int(ts.value // 10**6)


if __name__ == "__main__":
    SYMBOL = "ETHUSDT"
    MONTHS = 6
    INTERVAL = "1h"
    TZ = "Europe/Berlin"
    DROP_INCOMPLETE = True
    SAVE_DIR = paths.RAW_DATA_DIR

    print("\n" + "=" * 80)
    print("START DATA DOWNLOAD")
    print("=" * 80)

    print("\n[1/1] Download klines...")
    df = DataProvider.download_klines_months(
        symbol=SYMBOL,
        months=MONTHS,
        interval=INTERVAL,
        tz=TZ,
        drop_incomplete=DROP_INCOMPLETE,
        save_dir=str(SAVE_DIR),
    )

    print(f"Downloaded rows: {len(df)}")
    print(f"Saved raw JSON to: {SAVE_DIR / f'{SYMBOL}_{MONTHS}_{INTERVAL}.json'}")

    print("\n" + "-" * 80)
    print("FINISH DATA DOWNLOAD")
    print("-" * 80)
    print("=" * 80 + "\n")
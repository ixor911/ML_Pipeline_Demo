from .preprocessor import Preprocessor
from .DataProvider import DataProvider

"""
Features module.

This module contains the data preparation and feature engineering layer of the project.
Its purpose is to take raw market candles and transform them into structured,
model-ready datasets that can be used for training, validation, testing,
and later inference.

Main components
---------------
preprocessor/
    Feature engineering submodule.

    This submodule is responsible for converting raw OHLCV candles into enriched
    datasets with technical indicators, microstructure features, volatility signals,
    volume features, regime labels, and cross-market context.

    Included files:
    - ETH.py
        ETHUSDT-specific feature engineering pipeline.
        Main entry point:
        - PreprocessorETH.preprocess(...)

    - BTC.py
        BTCUSDT + ETHBTC feature engineering pipeline.
        Main entry point:
        - PreprocessorBTC.preprocess(...)

    - Preprocessor.py
        Unified high-level interface that orchestrates ETH and BTC preprocessors,
        merges feature sets, optionally builds synthetic ETHBTC, and can save
        processed datasets to disk.

        Main entry points:
        - Preprocessor.preprocess(...)
        - Preprocessor.preprocess_and_save(...)

DataProvider.py
    Raw data access layer.

    This module is responsible for downloading, loading, and converting Binance
    kline data into the standard internal dataframe format used by the feature
    engineering pipelines.

    Main responsibilities:
    - download raw candle data from Binance
    - save raw klines to JSON
    - read saved raw data back into pandas DataFrames
    - normalize schema and time handling

    Main entry points:
    - DataProvider.download_klines_months(...)
    - DataProvider.read_raw_json_to_df(...)
    - DataProvider.read_raw_symbol(...)
    - DataProvider.get_ethusdt_1h(...)
    - DataProvider.get_btcusdt_1h(...)
    - DataProvider.get_ethbtc_1h(...)

Module purpose
--------------
The features module acts as the bridge between raw exchange data and the ML pipeline.

Typical flow:
1. DataProvider downloads or reads raw market candles
2. preprocessor transforms raw candles into engineered features
3. the resulting processed dataset is used by downstream training and evaluation code

Design notes
------------
- Raw data access and feature engineering are intentionally separated.
- DataProvider handles acquisition and normalization of candles.
- preprocessor handles transformation into predictive features.
- This separation makes the pipeline easier to test, maintain, and extend.
"""
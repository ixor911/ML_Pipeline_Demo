from .ETH import PreprocessorETH
from .BTC import PreprocessorBTC
from .Preprocessor import Preprocessor

"""
Feature preprocessing package.

This package contains the project's feature engineering layer for raw market data.
Its main purpose is to transform raw OHLCV candles into structured, model-ready
datasets with technical indicators, microstructure signals, volume features,
and cross-market context.

Modules
-------
ETH.py
    ETH-specific feature engineering for ETHUSDT candles.

    Main responsibilities:
    - build trend, momentum, and volatility indicators
    - create return-based and volume-based features
    - create candle microstructure features
    - assign simplified market regime labels

    Main entry point:
    - PreprocessorETH.preprocess(...)

BTC.py
    BTC and ETHBTC feature engineering.

    Main responsibilities:
    - build BTC trend, momentum, volatility, and shock features
    - build ETHBTC relative-strength features
    - merge BTC and ETHBTC into a single feature block
    - optionally add BTC volume-related features

    Main entry point:
    - PreprocessorBTC.preprocess(...)

Preprocessor.py
    Unified high-level interface for the full preprocessing pipeline.

    Main responsibilities:
    - orchestrate ETH and BTC preprocessors
    - optionally build synthetic ETHBTC from ETH/BTC ratio
    - merge all feature blocks into one final dataframe
    - add optional cross-market ETH-vs-BTC features
    - save processed datasets to disk

    Main entry points:
    - Preprocessor.preprocess(...)
    - Preprocessor.preprocess_and_save(...)

Design notes
------------
- Asset-specific logic is separated into dedicated modules (ETH.py, BTC.py)
  to keep feature engineering easier to maintain and extend.
- Preprocessor.py acts as the integration layer for building the final dataset.
- All preprocessors are implemented as static utility-style classes,
  so they can be used without instance creation.
"""


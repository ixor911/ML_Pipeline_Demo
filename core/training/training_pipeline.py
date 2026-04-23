# core/training/training_pipeline.py

from typing import Dict, Any, Union, List, Tuple
from pathlib import Path
import pandas as pd

from core.target.slicing import slice_data
from core.target.builder import Builder
from core.models.TorchModel import TorchModel
from core.eval import evaluator
from core.models import ModelState
from core.io import loader
from core.training.training_engine import create_states


# ============================================================
# Data preparation
# ============================================================

def prepare_data(
    *,
    datapath: Union[str, Path],
    slicing_config: Dict[str, Any],
    builder_config: Dict[str, Any],
) -> Tuple:
    """
    Prepares train / validation data for model training.

    Steps:
    1. slice the processed dataset into windows
    2. build features and targets for each window
    3. return train/validation splits plus the final feature list

    If at least two windows are available, the first two are treated as
    train and validation. Otherwise, a simple 80/20 split is applied.
    """

    df = loader.load_processed_data(datapath)

    windows = slice_data(
        df=df,
        slicing_config=slicing_config,
    )

    bcfg = dict(builder_config)
    out = Builder.build_windows(*windows, **bcfg)

    # Expected common case:
    # (X_train, y_train, X_val, y_val, ...)
    if len(out) >= 4:
        X_train, y_train, X_val, y_val = out[:4]
    else:
        # Fallback for single-window setups.
        X, y = out
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]

    features = list(X_train.columns)
    return X_train, y_train, X_val, y_val, features


# ============================================================
# Training pipeline
# ============================================================

def training_pipeline(
    *,
    datapath: Union[str, Path],
    slicing_config: Dict[str, Any],
    builder_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, ModelState]:
    """
    Runs the full training pipeline and returns one ModelState per threshold category.

    Flow:
    1. prepare train / validation data
    2. train the model on train data
    3. compute validation probabilities
    4. search best thresholds per evaluation category
    5. wrap the trained model into category-specific ModelState objects
    """
    X_train, y_train, X_val, y_val, features = prepare_data(
        datapath=datapath,
        slicing_config=slicing_config,
        builder_config=builder_config,
    )

    # Used for absolute coverage metrics during threshold evaluation.
    abs_candles = len(X_train) + len(X_val)

    # Train the base model.
    model = TorchModel(**model_config)
    model.fit(X_train, y_train, X_val=X_val, y_cls_val=y_val)

    # Select thresholds on validation probabilities.
    probs = model.predict_proba(X_val)
    thrs = evaluator.compute_thresholds(probs, y_val, abs_candles=abs_candles)

    # Store enough metadata to reconstruct the training setup later.
    meta = {
        "datapath": datapath,
        "model_config": model_config,
        "slicing_config": slicing_config,
        "builder_config": builder_config,
        "features": features,
    }

    # The same trained model can be reused across multiple categories,
    # each with its own decision threshold.
    states = create_states(
        model=model,
        meta=meta,
        thrs=thrs,
    )

    return states


def training_pipeline_list(
    *,
    datapath: Union[str, Path],
    slicing_config: Dict[str, Any],
    builder_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> List[ModelState]:
    """
    Convenience wrapper that returns training results as a list instead of a dict.
    """
    states = training_pipeline(
        datapath=datapath,
        slicing_config=slicing_config,
        builder_config=builder_config,
        model_config=model_config
    )

    return list(states.values())


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example configs for local testing.

    datapath = "ETHUSDT_6_1h.csv"

    slicing_config = {
        "type": "candles",        # "candles" or "ranges"
        "candles": [3000, 1000],
        "from_tail": True,
        "end_date": None,
        "time_col": "open_time_eth",
    }

    builder_config = {
        "horizon": 1,
        "tau_pct": [0.03, None],        # deadzone only for train
        "extra_drop": ['regime', 'future_ret'],
        "future_ret_col": "future_ret",
        "regime_filter": ["trend_up"],
        "feature_groups": ["all"],
    }

    model_config = {
        "task": "classification",
        "hidden": 32,
        "depth": 3,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "epochs": 40,
        "patience": 6,
        "val_share": 0.2,
        "seed": 1,
        "verbose": False,
    }

    model_data = training_pipeline_list(
        datapath=datapath,
        slicing_config=slicing_config,
        builder_config=builder_config,
        model_config=model_config,
    )

    print("Training pipeline finished. Model is trained.")
    print(model_data)
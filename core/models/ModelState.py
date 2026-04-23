# core/models/ModelState.py

import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from configs import EVALUATOR_CONFIG
from . import TorchModel


class ModelState:
    """
    Lightweight wrapper around a fitted model and its runtime metadata.

    This object represents one concrete model instance together with:
    - category / optimization bucket
    - threshold used for binary decisions
    - saved path / model_id
    - feature-building metadata
    - evaluation results
    - structural signature for deduplication

    In practice, ModelState acts as the bridge between:
    - raw candles
    - feature preparation
    - model inference
    - metric evaluation
    - model persistence
    """

    def __init__(
        self,
        *,
        category: str,
        model: TorchModel,
        meta: dict,
        thr: float,
        path: Path = None,
        eval: dict = None,
        model_id: str = None,
        signature: str = None,
        is_inactive: bool = False
    ):
        self.model_id = model_id      # None -> candidate model not yet persisted
        self.category = category
        self.path = path
        self.model: TorchModel = model
        self.meta = meta if meta is not None else {}
        self.thr = thr
        self.eval = eval if eval is not None else {}
        self._signature = signature or self.meta.get("signature")

        # Inactive models are kept in RAM / storage but intentionally return no signals.
        self.is_inactive = bool(is_inactive or self.meta.get("is_inactive", False))

    @property
    def json(self):
        """
        Returns a simple dictionary snapshot of the current model state.
        """
        return {
            "model_id": self.model_id,
            "model": self.model,
            "category": self.category,
            "path": self.path,
            "thr": self.thr,
            "meta": self.meta,
            "eval": self.eval,
            "signature": self.signature,
        }

    @property
    def score(self) -> float | None:
        """
        Returns the main optimization score for this category.

        The target metric is resolved from evaluator_metrics config.
        """
        field = EVALUATOR_CONFIG.get(self.category, {}).get("optimize")
        return self.eval.get("metrics", {}).get(field)

    @staticmethod
    def make_signature(meta: dict) -> str:
        """
        Builds a compact structural signature for the model configuration.

        The signature is intended for deduplication of models that have the same
        effective structure and feature setup.

        Included in the signature:
        - category
        - model_config
        - slicing_config
        - builder_config
        - ordered feature list

        Excluded from the signature:
        - model_id
        - threshold
        - evaluation results
        - filesystem path
        """
        payload = {
            "category": meta.get("category"),
            "model_config": meta.get("model_config", {}),
            "slicing_config": meta.get("slicing_config", {}),
            "builder_config": meta.get("builder_config", {}),
            "features": meta.get("features", []),
        }

        raw = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def ensure_signature(self) -> str:
        """
        Returns an existing signature if available, otherwise computes and stores it.

        The signature is cached both in the instance and inside meta.
        """
        if self._signature:
            return self._signature

        meta_signature = self.meta.get("signature")
        if meta_signature:
            self._signature = meta_signature
            return self._signature

        self._signature = self.make_signature(self.meta)
        self.meta["signature"] = self._signature
        return self._signature

    @property
    def signature(self) -> str:
        """
        Read-only access to the ensured structural signature.
        """
        return self.ensure_signature()

    @signature.setter
    def signature(self, value: str | None):
        """
        Allows explicit signature override and keeps meta in sync.
        """
        self._signature = value
        if value is not None:
            self.meta["signature"] = value

    def get_builder_config_for_inference(self) -> dict:
        """
        Returns builder configuration adapted for inference.

        Important detail:
        - tau_pct is disabled during inference to avoid dropping rows via deadzone
        """
        bcfg = dict(self.meta.get("builder_config", {}) or {})
        bcfg["tau_pct"] = [None]
        return bcfg

    def prepare_data(
            self,
            candles: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Builds model-ready features and targets from raw candles.

        This method uses the Builder config stored in meta, which makes inference
        consistent with the original training pipeline.

        Returns:
            Tuple[pd.DataFrame, pd.Series]:
                X features and y target
        """
        from core.target import Builder

        if not isinstance(candles, pd.DataFrame):
            raise TypeError("ModelState.prepare_data expects candles as pandas.DataFrame")

        bcfg = self.get_builder_config_for_inference()

        out = Builder.build_windows(candles, **bcfg)

        if len(out) >= 2:
            X, y = out[:2]
        else:
            raise ValueError("Builder.build_windows returned invalid output")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Builder.build_windows must return X as pandas.DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("Builder.build_windows must return y as pandas.Series")

        return X, y

    def prepare_features(
            self,
            candles: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convenience shortcut that returns only X features without the target.
        """
        X, _ = self.prepare_data(candles)
        return X

    def predict_proba(
            self,
            candles: pd.DataFrame,
    ) -> np.ndarray:
        """
        Returns class-1 probabilities for the given candle set.

        If the model is inactive:
        - returns a zero array with the same length as X
        """
        X, _ = self.prepare_data(candles)

        if X.empty:
            return np.array([], dtype=float)

        if self.is_inactive:
            return np.zeros(len(X), dtype=float)

        probs = self.model.predict_proba(X)
        return np.asarray(probs, dtype=float)

    def predict(
            self,
            candles: pd.DataFrame,
    ) -> np.ndarray:
        """
        Returns binary predictions using the model threshold.

        Logic:
        - call predict_proba(...)
        - if empty -> return empty integer array
        - otherwise threshold probabilities with self.thr
        """
        probs = self.predict_proba(candles)

        if probs.size == 0:
            return np.array([], dtype=int)

        return (probs >= self.thr).astype(int)

    def predict_proba_one(
            self,
            candles: pd.DataFrame,
    ) -> Optional[float]:
        """
        Returns a single probability if exactly one row is produced.
        """
        probs = self.predict_proba(candles)

        if probs.size == 0:
            return None

        if probs.size > 1:
            raise ValueError(
                f"ModelState.predict_proba_one expected 0 or 1 prediction, got {probs.size}"
            )

        return float(probs[0])

    def predict_one(
            self,
            candles: pd.DataFrame,
    ) -> Optional[int]:
        """
        Returns a single binary prediction if exactly one row is produced.
        """
        preds = self.predict(candles)

        if preds.size == 0:
            return None

        if preds.size > 1:
            raise ValueError(
                f"ModelState.predict_one expected 0 or 1 prediction, got {preds.size}"
            )

        return int(preds[0])

    def predict_frame(
            self,
            candles: pd.DataFrame,
            include_target: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a flat prediction dataframe for the given candles.

        Output columns:
        - row_idx
        - proba
        - pred
        - target (optional)

        If the model is inactive:
        - probabilities are all zeros
        - predictions are all zeros
        """
        X, y = self.prepare_data(candles)

        if X.empty:
            cols = ["row_idx", "proba", "pred"]
            if include_target:
                cols.append("target")
            return pd.DataFrame(columns=cols)

        if self.is_inactive:
            probs = np.zeros(len(X), dtype=float)
            preds = np.zeros(len(X), dtype=int)
        else:
            probs = np.asarray(self.model.predict_proba(X), dtype=float)
            preds = (probs >= self.thr).astype(int)

        out = pd.DataFrame({
            "row_idx": X.index,
            "proba": probs,
            "pred": preds,
        })

        if include_target:
            out["target"] = y.to_numpy()

        return out

    def evaluate_metrics(
            self,
            candles: pd.DataFrame,
            abs_candles: Optional[int] = None,
    ) -> dict:
        """
        Evaluates the model on a candle set using the stored threshold.

        Metrics are computed through core.eval.evaluator and cached into self.eval["metrics"].
        """
        from core.eval import evaluator

        X, y = self.prepare_data(candles)

        if abs_candles is None:
            abs_candles = len(candles)

        if X.empty:
            return {
                "thr": self.thr,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mcc": 0.0,
                "total": 0,
                "signals": 0,
                "signals_percent": 0.0,
                "abs_candles": abs_candles,
                "abs_percent": 0.0 if abs_candles else None,
                "abs_signals": 0.0 if abs_candles else None,
            }

        probs = np.asarray(self.model.predict_proba(X), dtype=float)

        metrics = evaluator.evaluate_threshold(
            probs=probs,
            y=y,
            thr=self.thr,
            abs_candles=abs_candles,
        )

        self.eval['metrics'] = metrics

        return metrics

    def save(self):
        """
        Saves the model state and metadata.

        Behavior:
        - if model_id is missing -> create a new saved model
        - if model_id already exists -> overwrite weights and meta
        """
        from core.io.model_saver import (
            create_model,
            save_model,
            save_meta,
        )

        # New model -> create full persisted record.
        if getattr(self, "model_id", None) is None:
            state = create_model(self)
            self.model_id = state.model_id
            return self

        # Existing model -> overwrite model file and metadata.
        save_model(
            model=self.model,
            folder=self.path
        )

        save_meta(
            folder=self.path,
            **self.meta
        )

        return self
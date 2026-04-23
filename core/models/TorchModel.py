# core/models/TorchModel.py

from __future__ import annotations

import math
import random
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import pretty_print


def set_seed(seed: int = 42):
    """
    Sets random seed across Python, NumPy, and PyTorch.

    This helps keep training runs more reproducible, although exact determinism
    may still depend on backend-specific operations and hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _MLP(nn.Module):
    """
    Backbone MLP with two output heads:
    - classification head returning logits
    - regression head returning continuous values

    The hidden representation is built as a simple pyramidal stack,
    where hidden size is gradually reduced with depth.
    """

    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 2, dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim

        for i in range(depth):
            layers += [
                nn.Linear(d, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d = hidden
            hidden = max(32, hidden // 2)  # light pyramidal shrinkage

        self.backbone = nn.Sequential(*layers)

        # Classification head returns raw logits (without sigmoid).
        self.head_cls = nn.Linear(d, 1)

        # Regression head returns a single continuous value.
        self.head_reg = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the shared backbone and both output heads.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - classification logits
                - regression outputs
        """
        z = self.backbone(x)
        logit = self.head_cls(z).squeeze(1)
        reg = self.head_reg(z).squeeze(1)
        return logit, reg


class TorchModel:
    """
    General-purpose tabular model built on top of PyTorch.

    Key features:
    - accepts pandas DataFrame / Series inputs
    - automatically handles preprocessing:
        * one-hot encoding for categorical features
        * boolean casting
        * robust scaling using median and IQR
    - supports three modes:
        * classification
        * regression
        * both (multi-task)
    - includes:
        * time-aware validation split
        * early stopping
        * optional class imbalance handling

    The class is designed as a compact end-to-end wrapper:
    preprocessing, training, inference, and persistence are all handled here.
    """

    def __init__(
        self,
        *,
        task: str = "classification",      # 'classification' | 'regression' | 'both'
        hidden: int = 128,
        depth: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        cls_weight: Optional[float] = None,   # if None, pos_weight is computed automatically
        loss_weights: Tuple[float, float] = (1.0, 0.3),  # (w_cls, w_reg)
        batch_size: int = 128,
        epochs: int = 50,
        patience: int = 6,
        val_share: float = 0.2,               # use the latest val_share fraction as validation
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
    ):
        assert task in {"classification", "regression", "both"}

        self.task = task
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.cls_weight = cls_weight
        self.loss_weights = loss_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_share = val_share
        self.verbose = verbose

        set_seed(seed)

        # Resolve device automatically unless provided explicitly.
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Preprocessing artifacts fitted on the training data.
        self.cat_cols_: List[str] = []
        self.bool_cols_: List[str] = []
        self.num_cols_: List[str] = []
        self.feature_columns_: List[str] = []  # final columns after get_dummies
        self.medians_: Optional[np.ndarray] = None
        self.iqrs_: Optional[np.ndarray] = None

        # Model and optimizer become available only after fit().
        self.model_: Optional[_MLP] = None
        self.optim_: Optional[torch.optim.Optimizer] = None

        # Training history and best checkpoint metadata.
        self.history_: List[Dict[str, Any]] = []
        self.best_state_: Optional[Dict[str, Any]] = None

    # ===========================
    # public API
    # ===========================

    def fit(
        self,
        X: pd.DataFrame,
        y_cls: Optional[pd.Series] = None,
        y_reg: Optional[pd.Series] = None,
        *,
        X_val: Optional[pd.DataFrame] = None,
        y_cls_val: Optional[pd.Series] = None,
        y_reg_val: Optional[pd.Series] = None,
    ) -> "TorchModel":
        """
        Fits the model on tabular data.

        Training flow:
        1. Split train / validation if validation data is not provided
        2. Fit preprocessing on train data
        3. Transform validation data using train-fitted preprocessing
        4. Build the neural network dynamically based on input dimension
        5. Train with mini-batches
        6. Evaluate on validation set
        7. Apply early stopping based on validation loss

        Notes:
        - If validation data is not provided, the most recent val_share fraction
          of the dataset is used as validation, which is more appropriate for
          time-ordered data.
        - Classification uses BCEWithLogitsLoss.
        - Regression uses SmoothL1Loss.
        - In multi-task mode, both losses are combined with user-defined weights.

        Args:
            X (pd.DataFrame):
                Input feature matrix.

            y_cls (Optional[pd.Series]):
                Classification target.

            y_reg (Optional[pd.Series]):
                Regression target.

            X_val (Optional[pd.DataFrame]):
                Optional explicit validation features.

            y_cls_val (Optional[pd.Series]):
                Optional explicit classification validation target.

            y_reg_val (Optional[pd.Series]):
                Optional explicit regression validation target.

        Returns:
            TorchModel:
                Fitted model instance.
        """
        # 1) Time-based train / validation split.
        if X_val is None:
            split = int(len(X) * (1.0 - self.val_share))
            X_tr, X_va = X.iloc[:split], X.iloc[split:]
            y_cls_tr = y_cls.iloc[:split] if y_cls is not None else None
            y_cls_va = y_cls.iloc[split:] if y_cls is not None else None
            y_reg_tr = y_reg.iloc[:split] if y_reg is not None else None
            y_reg_va = y_reg.iloc[split:] if y_reg is not None else None
        else:
            X_tr, X_va = X, X_val
            y_cls_tr, y_cls_va = y_cls, y_cls_val
            y_reg_tr, y_reg_va = y_reg, y_reg_val

        # 2) Fit preprocessing on train and apply the same transform to validation.
        Xtr_np = self._fit_transform_X(X_tr)
        Xva_np = self._transform_X(X_va)

        # 3) Build the model dynamically from the final transformed feature dimension.
        in_dim = Xtr_np.shape[1]
        self.model_ = _MLP(
            in_dim,
            hidden=self.hidden,
            depth=self.depth,
            dropout=self.dropout
        ).to(self.device)

        self.optim_ = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 4) Configure losses.
        w_cls, w_reg = self.loss_weights
        bce = None

        if self.task in {"classification", "both"}:
            pos_weight = None

            if self.cls_weight is not None:
                pos_weight = torch.tensor([self.cls_weight], dtype=torch.float32, device=self.device)
            else:
                # Auto-compute positive class weight as neg / pos.
                y_arr = y_cls_tr.to_numpy().astype(np.float32)
                pos = y_arr.sum()
                neg = max(1.0, len(y_arr) - pos)
                pw = float(neg / max(1.0, pos))
                pos_weight = torch.tensor([pw], dtype=torch.float32, device=self.device)

            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        huber = nn.SmoothL1Loss()

        # 5) Convert arrays / targets to tensors on the selected device.
        def to_tensor(a, dtype=torch.float32):
            return torch.tensor(a, dtype=dtype, device=self.device)

        yc_tr = to_tensor(y_cls_tr.to_numpy().astype(np.float32)) if y_cls_tr is not None else None
        yr_tr = to_tensor(y_reg_tr.to_numpy().astype(np.float32)) if y_reg_tr is not None else None
        yc_va = to_tensor(y_cls_va.to_numpy().astype(np.float32)) if y_cls_va is not None else None
        yr_va = to_tensor(y_reg_va.to_numpy().astype(np.float32)) if y_reg_va is not None else None

        Xtr = to_tensor(Xtr_np)
        Xva = to_tensor(Xva_np)

        # 6) Simple mini-batch iterator over time-ordered indices.
        train_idx = np.arange(len(Xtr))
        batch = self.batch_size

        def batches(idx):
            for i in range(0, len(idx), batch):
                yield idx[i:i + batch]

        # 7) Training loop with early stopping based on validation loss.
        best_val = math.inf
        no_improve = 0
        self.history_.clear()
        self.best_state_ = None

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            perm = train_idx  # keep original time order for tabular time-series data
            tr_loss = 0.0

            for bi in batches(perm):
                xb = Xtr[bi]
                self.optim_.zero_grad()

                logit, yhat_reg = self.model_(xb)

                loss = torch.tensor(0.0, device=self.device)

                if self.task in {"classification", "both"} and yc_tr is not None:
                    yb = yc_tr[bi]
                    loss = loss + w_cls * bce(logit, yb)

                if self.task in {"regression", "both"} and yr_tr is not None:
                    ybr = yr_tr[bi]
                    loss = loss + w_reg * huber(yhat_reg, ybr)

                loss.backward()
                self.optim_.step()
                tr_loss += float(loss.item()) * len(bi)

            tr_loss /= len(Xtr)

            # ---- validation ----
            self.model_.eval()
            with torch.no_grad():
                logit_v, yhat_reg_v = self.model_(Xva)
                val_loss = 0.0

                if self.task in {"classification", "both"} and yc_va is not None:
                    val_loss += w_cls * float(bce(logit_v, yc_va).item())

                if self.task in {"regression", "both"} and yr_va is not None:
                    val_loss += w_reg * float(huber(yhat_reg_v, yr_va).item())

                # Additional lightweight classification metrics for monitoring.
                acc = f1 = mcc = np.nan
                if self.task in {"classification", "both"} and yc_va is not None:
                    p = torch.sigmoid(logit_v).detach().cpu().numpy()
                    yhat = (p >= 0.5).astype(np.int32)
                    ytrue = yc_va.detach().cpu().numpy().astype(np.int32)

                    acc = (yhat == ytrue).mean()
                    f1 = self._f1_binary(ytrue, yhat)
                    mcc = self._mcc_binary(ytrue, yhat)

            self.history_.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_acc": acc,
                "val_f1": f1,
                "val_mcc": mcc
            })

            if self.verbose:
                msg = f"[{epoch:03d}] train {tr_loss:.4f} | val {val_loss:.4f}"
                if not np.isnan(mcc):
                    msg += f" | MCC {mcc:.3f} F1 {f1:.3f} ACC {acc:.3f}"
                print(msg)

            # Early stopping checkpointing logic.
            if val_loss + 1e-12 < best_val:
                best_val = val_loss
                no_improve = 0
                self.best_state_ = {
                    "model": self.model_.state_dict(),
                    "optim": self.optim_.state_dict(),
                    "feature_columns": list(self.feature_columns_),
                    "medians": None if self.medians_ is None else self.medians_.copy(),
                    "iqrs": None if self.iqrs_ is None else self.iqrs_.copy(),
                    "cat_cols": list(self.cat_cols_),
                    "bool_cols": list(self.bool_cols_),
                    "num_cols": list(self.num_cols_),
                }
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Load the best checkpoint after training finishes.
        if self.best_state_ is not None:
            self.model_.load_state_dict(self.best_state_["model"])

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns class-1 probabilities for classification tasks.

        Args:
            X (pd.DataFrame):
                Input features.

        Returns:
            np.ndarray:
                Probability vector of shape [N].
        """
        self._ensure_fitted()

        if self.task not in {"classification", "both"}:
            raise RuntimeError("Model task is not classification.")

        X_np = self._transform_X(X)
        self.model_.eval()

        with torch.no_grad():
            logits, _ = self.model_(torch.tensor(X_np, dtype=torch.float32, device=self.device))
            p = torch.sigmoid(logits).cpu().numpy()

        return p

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Returns binary predictions using a probability threshold.

        Args:
            X (pd.DataFrame):
                Input features.

            threshold (float):
                Decision threshold.

        Returns:
            np.ndarray:
                Binary predictions of shape [N].
        """
        p = self.predict_proba(X)
        return (p >= threshold).astype(np.int32)

    def predict_reg(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns regression outputs.

        Args:
            X (pd.DataFrame):
                Input features.

        Returns:
            np.ndarray:
                Continuous regression predictions.
        """
        self._ensure_fitted()

        if self.task not in {"regression", "both"}:
            raise RuntimeError("Model task is not regression.")

        X_np = self._transform_X(X)
        self.model_.eval()

        with torch.no_grad():
            _, yhat = self.model_(torch.tensor(X_np, dtype=torch.float32, device=self.device))
            out = yhat.cpu().numpy()

        return out

    def predict_both(self, X: pd.DataFrame, threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Returns both classification and regression outputs in a single call.

        Returns:
            Dict[str, np.ndarray]:
                {
                    "proba_up": ...,
                    "pred_up": ...,
                    "pred_ret": ...
                }
        """
        self._ensure_fitted()

        X_np = self._transform_X(X)
        self.model_.eval()

        with torch.no_grad():
            logits, yhat = self.model_(torch.tensor(X_np, dtype=torch.float32, device=self.device))
            p = torch.sigmoid(logits).cpu().numpy()
            ret = yhat.cpu().numpy()

        return {
            "proba_up": p,
            "pred_up": (p >= threshold).astype(np.int32),
            "pred_ret": ret,
        }

    def save(self, path: str):
        """
        Saves the best fitted model state together with preprocessing artifacts.
        """
        self._ensure_fitted()
        torch.save(self.best_state_, path)

    def load(self, path: str):
        """
        Loads a previously saved model state and restores preprocessing metadata.

        Note:
            The model architecture is rebuilt from current hyperparameters, so
            loading should be done with a compatible configuration.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.best_state_ = state
        self.feature_columns_ = state["feature_columns"]
        self.medians_ = state["medians"]
        self.iqrs_ = state["iqrs"]
        self.cat_cols_ = state["cat_cols"]
        self.bool_cols_ = state["bool_cols"]
        self.num_cols_ = state["num_cols"]

        # Rebuild the model using the fitted feature dimension.
        in_dim = len(self.feature_columns_)
        self.model_ = _MLP(
            in_dim,
            hidden=self.hidden,
            depth=self.depth,
            dropout=self.dropout
        ).to(self.device)

        self.model_.load_state_dict(state["model"])
        self.optim_ = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    # ===========================
    # preprocessing
    # ===========================

    def _fit_transform_X(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fits preprocessing on the training dataset and returns a transformed array.

        Steps:
        - detect categorical / boolean / numeric columns
        - apply one-hot encoding to categorical features
        - cast booleans to integers
        - fit robust scaling using median and IQR
        - replace NaN / inf values with zeros

        Returns:
            np.ndarray:
                Transformed float32 feature matrix.
        """
        df = X.copy()

        # Detect feature types from the training data.
        self.cat_cols_ = [
            c for c in df.columns
            if (
                    pd.api.types.is_object_dtype(df[c])
                    or pd.api.types.is_categorical_dtype(df[c])
                    or pd.api.types.is_string_dtype(df[c])
            )
        ]
        self.bool_cols_ = [c for c in df.columns if str(df[c].dtype) == "bool"]
        self.num_cols_ = [c for c in df.columns if c not in self.cat_cols_ + self.bool_cols_]

        # One-hot encode categorical columns.
        if self.cat_cols_:
            df = pd.get_dummies(df, columns=self.cat_cols_, drop_first=False)

        # Cast boolean columns to integer representation.
        for c in self.bool_cols_:
            df[c] = df[c].astype(np.int32)

        # Store final feature order after preprocessing.
        self.feature_columns_ = df.columns.tolist()
        X_np = df.to_numpy(dtype=np.float32)

        # Robust scaling: (x - median) / IQR.
        self.medians_ = np.nanmedian(X_np, axis=0)
        q75 = np.nanpercentile(X_np, 75, axis=0)
        q25 = np.nanpercentile(X_np, 25, axis=0)
        self.iqrs_ = q75 - q25

        # Protect against division by zero for constant columns.
        self.iqrs_[np.abs(self.iqrs_) < 1e-12] = 1.0

        X_np = (X_np - self.medians_) / self.iqrs_
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
        return X_np

    def _transform_X(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applies already fitted preprocessing to any new dataset.

        This method:
        - reproduces the same one-hot encoding structure
        - aligns columns to the train-time schema
        - applies stored robust scaling parameters

        Returns:
            np.ndarray:
                Transformed float32 feature matrix.
        """
        df = X.copy()

        # Apply the same one-hot encoding logic as in training.
        if self.cat_cols_:
            df = pd.get_dummies(df, columns=self.cat_cols_, drop_first=False)

        # Cast boolean columns to integers when present.
        for c in self.bool_cols_:
            if c in df.columns:
                df[c] = df[c].astype(np.int32)

        # Add any missing columns from the train schema.
        for c in self.feature_columns_:
            if c not in df.columns:
                df[c] = 0

        # Remove unexpected extra columns not seen during training.
        extra = [c for c in df.columns if c not in self.feature_columns_]
        if extra:
            df = df.drop(columns=extra)

        # Reorder strictly to the stored training schema.
        df = df[self.feature_columns_]

        X_np = df.to_numpy(dtype=np.float32)

        # Apply stored robust scaling.
        X_np = (X_np - self.medians_) / self.iqrs_
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
        return X_np

    # ===========================
    # metrics helpers
    # ===========================

    @staticmethod
    def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes binary F1 score from predicted labels.
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        denom = (2 * tp + fp + fn)
        return 0.0 if denom == 0 else (2 * tp) / denom

    @staticmethod
    def _mcc_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes Matthews Correlation Coefficient for binary classification.
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        num = tp * tn - fp * fn
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return 0.0 if den == 0 else (num / den)

    def _ensure_fitted(self):
        """
        Ensures the model has already been fitted before inference or saving.
        """
        if self.model_ is None or self.feature_columns_ is None:
            raise RuntimeError("Model is not fitted yet.")
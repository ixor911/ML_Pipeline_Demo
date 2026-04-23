# core/io/model_saver.py

from __future__ import annotations

import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch

import paths
from . import model_test_saver
from ..models import ModelState


def _ensure_thr(category: str, thr: float = None, thrs: dict = None, **kwargs):
    """Resolves threshold for a category from explicit value or mapping."""
    if not thr and not thrs:
        return None

    thrs = thrs if thrs else {}
    return thr or thrs.get(category, None)


def generate_model_id() -> str:
    """Generates a unique model ID like: model-YYYYMMDD_HHMMSS_xxxxxx."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"model-{now}_{rand}"


def _get_category_dir(category: str) -> Path:
    """Returns the save directory for a model category."""
    cat = category.lower()
    base = paths.MODEL_SAVES_DIR / cat
    # base.mkdir(parents=True, exist_ok=True)
    return base


def _ensure_dir(path: Path) -> None:
    """Creates a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, folder: Path, filename: str = "model.pt", *args, **kwargs) -> Path:
    """
    Saves model weights.

    Strategy:
    - use model.save(path) if available
    - otherwise save state_dict()
    - otherwise save the full object
    """
    # _ensure_dir(folder)
    path = folder / filename

    # if hasattr(model, "save") and callable(getattr(model, "save")):
    #     model.save(str(path))
    # else:
    #     state_dict = getattr(model, "state_dict", None)
    #     if callable(state_dict):
    #         torch.save(model.state_dict(), path)
    #     else:
    #         torch.save(model, path)

    return path


def save_meta(
    folder: Path,
    model_config: Dict[str, Any],
    slicing_config: Dict[str, Any],
    builder_config: Dict[str, Any],
    datapath: str,
    model_id: str,
    category: str,
    features: list,
    thr: float,
    signature: str = None,
    filename: str = "meta.json",
    *args, **kwargs
) -> dict:
    """Saves model metadata to meta.json."""
    _ensure_dir(folder)

    meta = {
        "model_id": model_id,
        "category": category,
        "thr": thr,
        "datapath": datapath,
        "model_config": model_config,
        "slicing_config": slicing_config,
        "builder_config": builder_config,
        "features": features,
        "signature": signature,
    }

    path = folder / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def create_model(state: ModelState) -> ModelState:
    """
    Creates a new saved model on disk and updates the provided ModelState.
    """
    # Generate new ID and create target folder.
    model_id = generate_model_id()
    cat_dir = _get_category_dir(state.category)
    model_dir = cat_dir / model_id
    # _ensure_dir(model_dir)

    # Save weights.
    # save_model(state.model, model_dir, filename="model.pt")

    # Save metadata.
    # state.ensure_signature()
    # meta = save_meta(
    #     folder=model_dir,
    #     model_id=model_id,
    #     filename="meta.json",
    #     thr=state.thr,
    #     **state.meta
    # )

    # Create tests folder and optionally save evaluation snapshots.
    # tests_dir = model_dir / "tests"
    # tests_dir.mkdir(exist_ok=True)

    # if state.eval:
    #     model_test_saver.eval_tests(
    #         model_dir,
    #         state.eval
    #     )

    # Update in-memory state.
    state.model_id = model_id
    state.path = model_dir
    # state.meta = meta

    return state


def delete_model(model_id: str) -> None:
    """Deletes a saved model folder by model_id if it exists."""
    root = paths.MODEL_SAVES_DIR
    if not root.exists():
        return

    for cat_dir in root.iterdir():
        if not cat_dir.is_dir():
            continue

        candidate = cat_dir / model_id
        if candidate.exists() and candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)


def replace_model(
    *,
    old_model_id: str,
    new_state: ModelState,
) -> ModelState:
    """
    Replaces an existing saved model with a new candidate model.
    """
    if new_state.model_id is not None:
        raise ValueError("new_state must be a candidate (model_id=None)")

    delete_model(old_model_id)
    return create_model(new_state)
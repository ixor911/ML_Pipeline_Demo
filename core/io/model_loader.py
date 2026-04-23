# core/io/model_loader.py

"""
Utilities for loading saved models from disk.

Expected folder structure:

saves/models/
  <category>/
    <model_id>/
      model.pt
      meta.json
      tests/   (optional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.models.TorchModel import TorchModel
from core.models.ModelState import ModelState
from paths import MODEL_SAVES_DIR


def _models_root() -> Path:
    """Returns the root model directory and ensures it exists."""
    root = Path(MODEL_SAVES_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def find_model_dir(model_id: str) -> Path | None:
    """Returns the absolute model directory path or None if not found."""
    category = find_model_category(model_id)
    if category is None:
        return None

    model_dir = MODEL_SAVES_DIR / category / model_id
    if model_dir.exists() and model_dir.is_dir():
        return model_dir

    return None


def get_all_categories() -> List[str]:
    """Returns all model categories except archive and hidden folders."""
    root = _models_root()
    categories: List[str] = []

    for item in root.iterdir():
        if not item.is_dir():
            continue

        name = item.name
        if name == "archive":
            continue
        if name.startswith("."):
            continue

        categories.append(name)

    return sorted(categories)


def get_category_ids(category: str) -> List[str]:
    """Returns all model IDs inside a category."""
    root = _models_root()
    cat_dir = root / category

    if not cat_dir.exists() or not cat_dir.is_dir():
        return []

    ids: List[str] = []
    for item in cat_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            ids.append(item.name)

    return sorted(ids)


def get_all_ids() -> Dict[str, List[str]]:
    """Returns model IDs grouped by category."""
    result: Dict[str, List[str]] = {}

    for cat in get_all_categories():
        result[cat] = get_category_ids(cat)

    return result


def find_model_category(model_id: str) -> Optional[str]:
    """Finds the category that contains the given model_id."""
    root = _models_root()

    for cat in get_all_categories():
        model_dir = root / cat / model_id
        if model_dir.is_dir():
            return cat

    return None


def load_model_meta_from_path(model_dir: Path) -> Dict[str, Any]:
    """Loads meta.json from a model directory. Returns {} if missing."""
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        return {}

    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_from_path(model_config: dict, model_dir: Path):
    """
    Loads a model from model.pt using the saved model_config.
    """
    model_path = model_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = TorchModel(**model_config)
    model.load(path=model_path)
    return model


def load_model(
    model_id: str,
    category: Optional[str] = None,
    strict: bool = False,
) -> Optional[ModelState]:
    """
    Loads a saved model by model_id and returns it as ModelState.

    If category is not provided, it is resolved automatically.
    """
    root = _models_root()

    # Resolve category automatically if needed.
    if category is None:
        category = find_model_category(model_id)

    if category is None:
        msg = f"Model id '{model_id}' not found in any category."
        if strict:
            raise FileNotFoundError(msg)
        return None

    model_dir = root / category / model_id
    if not model_dir.exists() or not model_dir.is_dir():
        msg = f"Model directory not found: {model_dir}"
        if strict:
            raise FileNotFoundError(msg)
        return None

    # Load metadata and weights.
    meta = load_model_meta_from_path(model_dir)
    model_obj = load_model_from_path(meta.get("model_config"), model_dir)
    thr = meta.get("thr")

    state = ModelState(
        model_id=model_id,
        category=category,
        path=model_dir,
        thr=thr,
        model=model_obj,
        meta=meta,
    )

    # Backfill signature for older saved models if needed.
    state.ensure_signature()

    return state


def load_category_models(
    category: str,
    strict: bool = False
) -> Dict[str, ModelState]:
    """Loads all models inside a category."""
    ids = get_category_ids(category)
    models = {}

    for mid in ids:
        model_state = load_model(
            model_id=mid,
            category=category,
            strict=strict,
        )
        if model_state is not None:
            models[mid] = model_state

    return models


def load_all_models(strict: bool = False):
    """Loads all saved models grouped by category."""
    result: Dict[str, Dict[str, Any]] = {}

    for cat in get_all_categories():
        result[cat] = load_category_models(category=cat, strict=strict)

    return result
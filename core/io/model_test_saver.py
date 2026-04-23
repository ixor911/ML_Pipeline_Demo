# core/io/model_test_saver.py

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import uuid


def _ensure_dir(path: Path) -> None:
    """Creates a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def create_test_name() -> str:
    """Generates a unique test file name."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"test-{ts}_{suffix}"


def create_test(
    model_dir: Path,
    test_category: str,
    result: Dict[str, Any],
) -> Path:
    """
    Saves a single test result to:

        model_dir/tests/{test_category}/{test_name}.json
    """
    tests_root = model_dir / "tests" / test_category
    tests_root.mkdir(parents=True, exist_ok=True)

    test_name = create_test_name()
    test_path = tests_root / f"{test_name}.json"

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return test_path


def eval_tests(
    model_dir: Path,
    eval: Dict[str, Any],
) -> None:
    """
    Saves evaluation results for one model.

    Expected format:
    {
        "mcc": {...},
        "tpr90": {...},
        ...
    }

    Lists are also supported and saved as multiple test files.
    """
    if not eval:
        return

    for category, result in eval.items():
        if result is None:
            continue

        if isinstance(result, list):
            for r in result:
                create_test(model_dir, category, r)
        else:
            create_test(model_dir, category, result)


def eval_models(
    models_ram: Dict[str, Dict[str, Any]],
) -> None:
    """
    Saves eval results for all models currently stored in RAM.

    Expected format:
    {
        category: {
            model_id: ModelState,
            ...
        }
    }
    """
    for _, models_by_id in models_ram.items():
        for state in models_by_id.values():
            if state.path is None:
                continue
            if not state.eval:
                continue

            eval_tests(state.path, state.eval)
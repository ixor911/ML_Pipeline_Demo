# core/pipelines/grid_search_pipeline_async.py

from __future__ import annotations

import os
import traceback
import multiprocessing as mp
from queue import Empty
from typing import Any, Dict, List
import pandas as pd

from core.io.loader import load_config, load_config_grid, load_processed_data
from core.target.slicing import slice_data
from core.training.training_pipeline import training_pipeline_list
from core.io.model_selector import eval_model
from core.models import ModelState
from pathlib import Path


# =========================
# Helpers
# =========================

def print_models_ram_metrics(
    models_ram: Dict[str, Dict[str, ModelState]],
    excel_path: str | Path | None = None,
) -> None:
    """
    Prints metrics for models_ram to the console.

    If excel_path is provided, the same per-category tables are also saved
    into an Excel workbook. If the target file already exists, a numeric
    suffix is appended automatically.
    """
    # Make pandas output easier to inspect in the terminal.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)

    excel_writer = None

    if excel_path is not None:
        excel_path = Path(excel_path)
        excel_path.parent.mkdir(parents=True, exist_ok=True)

        final_path = excel_path

        # Auto-increment file name if the target already exists.
        if final_path.exists():
            base = final_path.stem
            suffix = final_path.suffix

            i = 1
            while True:
                new_path = final_path.with_name(f"{base}_{i}{suffix}")
                if not new_path.exists():
                    final_path = new_path
                    break
                i += 1

        excel_writer = pd.ExcelWriter(final_path, engine="openpyxl")

    print("\n" + "=" * 80)
    print("MODELS RAM METRICS")
    print("=" * 80)

    for category, category_models in models_ram.items():
        rows = []

        for model_id, state in category_models.items():
            metrics = state.eval.get("metrics", {})
            selector = state.eval.get("selector", {})

            row = {
                "model_id": state.model_id,
                "category": state.category,
                "thr": state.thr,

                # Main category score.
                "score": state.score,

                # Standard evaluation metrics.
                "acc": metrics.get("acc"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "mcc": metrics.get("mcc"),

                # Coverage / signal activity metrics.
                "signals": metrics.get("signals"),
                "signals_percent": metrics.get("signals_percent"),
                "abs_signals": metrics.get("abs_signals"),
                "abs_percent": metrics.get("abs_percent"),

                # Selector-level ranking information.
                "score_raw": selector.get("score_raw"),
                "score_norm": selector.get("score_norm"),
                "constraint_fit": selector.get("constraint_fit"),
                "final_score": selector.get("final_score"),
                "constraints_passed": selector.get("passed"),
                "violations_count": selector.get("violations_count"),
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            df = df.sort_values(by="score", ascending=False).reset_index(drop=True)

        print(f"\n--- CATEGORY: {category} ---")
        if df.empty:
            print("No models")
        else:
            print(df)

        if excel_writer is not None:
            # Excel sheet names are limited to 31 characters.
            sheet_name = str(category)[:31]
            df.to_excel(excel_writer, sheet_name=sheet_name, index=False)

    if excel_writer is not None:
        excel_writer.close()
        print(f"\nExcel saved to: {final_path}")

    print("=" * 80 + "\n")


# =========================
# Worker
# =========================

def _train_test_worker(
    *,
    worker_id: int,
    config_queue: mp.Queue,
    result_queue: mp.Queue,
    test_candles: pd.DataFrame,
) -> None:
    """
    Worker process that handles one training config at a time.

    For each config:
    - trains a base model
    - expands it into candidate ModelState objects
    - evaluates every candidate on the shared test candles
    - sends tested candidates back through result_queue
    """
    while True:
        item = config_queue.get()

        if item is None:
            result_queue.put(("DONE", worker_id))
            return

        idx, total, training_config = item

        try:
            candidate_states: List[ModelState] = training_pipeline_list(
                datapath=training_config["datapath"],
                slicing_config=training_config["slicing"],
                builder_config=training_config["builder"],
                model_config=training_config["model"],
            )

            tested_count = 0
            for state in candidate_states:
                state.evaluate_metrics(candles=test_candles)
                tested_count += 1
                result_queue.put(("STATE", idx, total, state))

            result_queue.put(("CONFIG_DONE", worker_id, idx, total, tested_count))

        except Exception as exc:
            result_queue.put(
                (
                    "ERROR",
                    worker_id,
                    idx,
                    total,
                    str(exc),
                    traceback.format_exc(),
                )
            )


# =========================
# Main async pipeline
# =========================

def grid_search_pipeline_async(
    *,
    training_configs: List[Dict[str, Any]],
    test_candles: pd.DataFrame,
    models_ram: dict,
    limit: int,
    workers: int | None = None,
    queue_maxsize: int = 32,
) -> dict:
    """
    Runs asynchronous grid search over multiple training configs.

    The main process is responsible for:
    - feeding configs to workers
    - collecting tested candidates
    - passing candidates through the selector
    - maintaining the shared in-memory best-model store
    """
    total = len(training_configs)
    if total == 0:
        return models_ram

    # Keep the worker count bounded to avoid oversubscribing the machine.
    if workers is None:
        workers = max(1, min(os.cpu_count() or 2, 4, total))

    ctx = mp.get_context("spawn")
    config_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue(maxsize=queue_maxsize)

    # Enqueue all training configs.
    for idx, cfg in enumerate(training_configs, start=1):
        config_queue.put((idx, total, cfg))

    # Add stop sentinels, one per worker.
    for _ in range(workers):
        config_queue.put(None)

    processes: List[mp.Process] = []
    for worker_id in range(1, workers + 1):
        p = ctx.Process(
            target=_train_test_worker,
            kwargs={
                "worker_id": worker_id,
                "config_queue": config_queue,
                "result_queue": result_queue,
                "test_candles": test_candles,
            },
            daemon=False,
        )
        p.start()
        processes.append(p)

    done_workers = 0
    finished_configs = 0

    try:
        while done_workers < workers:
            try:
                msg = result_queue.get(timeout=1.0)
            except Empty:
                continue

            kind = msg[0]

            if kind == "STATE":
                _, idx, total_configs, state = msg

                # The selector decides whether the candidate should be kept.
                models_ram = eval_model(
                    candidate=state,
                    models_ram=models_ram,
                    limit=limit,
                )

            elif kind == "CONFIG_DONE":
                _, worker_id, idx, total_configs, tested_count = msg
                finished_configs += 1
                remaining = total - finished_configs

                print(
                    f"[{finished_configs} / {total}] "
                    f"config finished | worker={worker_id} | "
                    f"tested_states={tested_count} | remaining={remaining}"
                )

            elif kind == "ERROR":
                _, worker_id, idx, total_configs, err, tb = msg
                raise RuntimeError(
                    f"Worker {worker_id} failed on config [{idx}/{total_configs}]:\n{err}\n{tb}"
                )

            elif kind == "DONE":
                _, worker_id = msg
                done_workers += 1

    finally:
        # Try graceful shutdown first.
        for p in processes:
            p.join(timeout=5)

        # Force-terminate any stuck workers.
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

    return models_ram


# =========================
# __main__
# =========================

def __main__():
    from paths import LOGS_DIR

    LIMIT = 3
    WORKERS = 2
    QUEUE_MAXSIZE = 32

    TRAIN_GRID_NAME = "model_train_basic_grid"
    TEST_CONFIG_NAME = "model_test_basic"

    def count_models(models_ram: dict) -> int:
        return sum(len(category_models) for category_models in models_ram.values())

    def count_categories(models_ram: dict) -> int:
        return len(models_ram)

    print("\n" + "=" * 80)
    print("START GRID SEARCH ASYNC")
    print("=" * 80)

    print("\n[1/4] Load training grid configs...")
    training_configs = list(load_config_grid(TRAIN_GRID_NAME))
    total_configs = len(training_configs)
    print(f"Training configs loaded: {total_configs}")

    print("\n[2/4] Load test config...")
    test_config = load_config(TEST_CONFIG_NAME)
    test_datapath = test_config["datapath"]
    test_slicing = test_config["slicing"]

    test_data = load_processed_data(test_datapath)
    print("Test config loaded.")

    print("\n[3/4] Build test candles...")
    test_windows = slice_data(
        df=test_data,
        slicing_config=test_slicing,
    )
    test_candles = test_windows[0]
    print(f"Test candles ready: rows={len(test_candles)}")

    print("\n[4/4] Run async grid search...")
    print(
        f"Configs={total_configs} | Workers={WORKERS} | "
        f"Queue={QUEUE_MAXSIZE} | Limit={LIMIT}"
    )

    models_ram = {}
    models_ram = grid_search_pipeline_async(
        training_configs=training_configs,
        test_candles=test_candles,
        models_ram=models_ram,
        limit=LIMIT,
        workers=WORKERS,
        queue_maxsize=QUEUE_MAXSIZE,
    )

    models_after = count_models(models_ram)
    categories_after = count_categories(models_ram)

    print("\n" + "-" * 80)
    print("GRID SEARCH FINISHED")
    print("-" * 80)
    print(f"Train configs processed: {total_configs}")
    print(f"Models : {models_after}")
    print(f"Categories : {categories_after}")

    print("\nFinal RAM state:")
    for category, category_models in models_ram.items():
        print(f"  - {category}: {len(category_models)} model(s)")

    print("\n[FINAL] Print models metrics...")
    print_models_ram_metrics(
        models_ram=models_ram
    )

    print("=" * 80)
    print("END GRID SEARCH ASYNC")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    __main__()
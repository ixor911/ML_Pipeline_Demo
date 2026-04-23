from typing import Any, Dict, List
import pandas as pd

from core.pipelines.candidate_batch_pipeline import candidate_batch_pipeline
from core.io.loader import load_config, load_config_grid, load_processed_data
from core.target.slicing import slice_data
from core.testing import test_saved_models_metrics_pipeline
from core.io import model_test_saver
from core.pipelines.grid_search_pipeline_async import print_models_ram_metrics


def grid_search_pipeline(
    *,
    training_configs: List[Dict[str, Any]],
    test_candles: pd.DataFrame,
    models_ram: dict,
    limit: int,
) -> dict:
    """
    Runs grid search sequentially over a list of training configs.

    For each config:
    - trains candidate states
    - evaluates them on shared test candles
    - passes them through the selector
    - updates the in-memory model pool
    """
    total = len(training_configs)

    for i in range(total):
        training_config = training_configs[i]

        models_ram = candidate_batch_pipeline(
            training_config=training_config,
            test_candles=test_candles,
            models_ram=models_ram,
            limit=limit,
        )

        # Simple progress output for long sequential runs.
        print(f"[{i} / {total}]")

    return models_ram


def __main__():
    LIMIT = 3

    def count_models(models_ram: dict) -> int:
        return sum(len(category_models) for category_models in models_ram.values())

    def count_categories(models_ram: dict) -> int:
        return len(models_ram)

    print("\n" + "=" * 80)
    print("START grid_search_pipeline.__main__")
    print("=" * 80)

    # 1) Load all training grid combinations.
    print("\n[1/5] Load training grid configs...")
    training_configs = list(load_config_grid("model_train_basic_grid"))
    print(f"Training configs loaded: {len(training_configs)}")

    # 2) Load test config and build the shared test candle window.
    print("\n[2/5] Load test config and build test candles...")
    test_config = load_config("model_test_basic")

    test_datapath = test_config["datapath"]
    test_slicing = test_config["slicing"]
    test_data = load_processed_data(test_datapath)

    test_windows = slice_data(
        df=test_data,
        slicing_config=test_slicing,
    )
    test_candles = test_windows[0]

    print(f"Test candles loaded: rows={len(test_candles)}")

    # 3) Run sequential grid search.
    print("\n[5/5] Run grid_search_pipeline...")
    models_ram = {}
    models_ram = grid_search_pipeline(
        training_configs=training_configs,
        test_candles=test_candles,
        models_ram=models_ram,
        limit=LIMIT,
    )

    models_after = count_models(models_ram)
    categories_after = count_categories(models_ram)

    print("\n" + "-" * 80)
    print("FINISH grid_search_pipeline.__main__")
    print("-" * 80)
    print(f"Train configs processed: {len(training_configs)}")
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
    print("END GRID SEARCH")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    __main__()
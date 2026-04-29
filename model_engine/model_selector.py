"""
model_engine/model_selector.py
--------------------------------
Recommends models based on DatasetProfile and resolves user choice.

BUG FIXED:
----------
model_selector.py had: logger.info = lambda msg: None
This DESTROYED the global loguru logger for the rest of the pipeline.
Every log after this point was silently dropped — making the pipeline
appear "stuck" because nothing was printed after "Selecting models".

FIX: Removed the broken logger override completely.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger
from rich.console import Console
from rich.table import Table

from config.constants import MODEL_CANDIDATES
from data_analysis.dataset_analyzer import DatasetProfile

console = Console()


class ModelSelector:

    def recommend(self, profile: DatasetProfile) -> List[str]:
        """Returns recommended models for the task and size bucket."""
        task   = profile.task_type or "classification"
        bucket = profile.size_bucket

        candidates = MODEL_CANDIDATES.get(task, {}).get(
            bucket, ["LightGBM", "XGBoost", "RandomForest"]
        )

        logger.info(f"Recommended models for {task}/{bucket}: {candidates}")
        return candidates

    def present_and_choose(
        self,
        recommendations: List[str],
        user_choice: Optional[str] = None,
    ) -> List[str]:
        """
        Resolves the final model list.
          None / "auto" -> all recommendations
          "1","2","3"   -> single model by index
          model_name    -> that specific model
        """
        table = Table(title="Recommended Models", show_lines=True)
        table.add_column("#",     style="cyan",  width=4)
        table.add_column("Model", style="green")
        for i, name in enumerate(recommendations, 1):
            table.add_row(str(i), name)
        console.print(table)

        if user_choice is None or user_choice.strip().lower() in (
            "", "auto", "all"
        ):
            logger.info(
                f"Training all {len(recommendations)} models: {recommendations}"
            )
            return recommendations

        if user_choice.strip().isdigit():
            idx = int(user_choice.strip()) - 1
            if 0 <= idx < len(recommendations):
                chosen = [recommendations[idx]]
                logger.info(f"User selected: {chosen[0]}")
                return chosen

        custom = user_choice.strip()
        logger.info(f"Custom model requested: {custom}")
        return [custom]

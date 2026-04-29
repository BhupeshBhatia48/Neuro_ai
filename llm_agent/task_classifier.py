"""
llm_agent/task_classifier.py
------------------------------
Re-validates / refines the task type once we have dataset metadata.
For example, if the idea said "regression" but the target column
has only 2 unique values, we flip to "classification".
"""

from config.constants import (
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TASK_TIME_SERIES,
    TASK_NLP,
    TASK_COMPUTER_VISION,
    SUPPORTED_TASKS,
)


class TaskClassifier:
    """
    Rule-based task classifier that can override the LLM's initial guess
    once we have concrete information about the dataset.
    """

    def classify(
        self,
        initial_task: str,
        target_unique_values: int | None = None,
        has_datetime_index: bool = False,
        has_text_columns: bool = False,
        has_image_columns: bool = False,
    ) -> str:
        """
        Parameters
        ----------
        initial_task : str
            The task type suggested by IdeaParser.
        target_unique_values : int | None
            Number of unique values in the target column (if known).
        has_datetime_index : bool
            True when the dataset has a datetime index / date column.
        has_text_columns : bool
            True when free-text columns are detected.
        has_image_columns : bool
            True when image path columns are detected.

        Returns
        -------
        str — finalised task type from SUPPORTED_TASKS
        """
        # Override based on hard evidence
        if has_image_columns:
            return TASK_COMPUTER_VISION

        if has_text_columns and initial_task not in (
            TASK_CLASSIFICATION,
            TASK_REGRESSION,
        ):
            return TASK_NLP

        if has_datetime_index and initial_task not in (
            TASK_CLASSIFICATION,
            TASK_REGRESSION,
        ):
            return TASK_TIME_SERIES

        # Binary / few-class target -> classification
        if (
            target_unique_values is not None
            and target_unique_values <= 20
            and initial_task == TASK_REGRESSION
        ):
            return TASK_CLASSIFICATION

        # Many unique values -> regression
        if (
            target_unique_values is not None
            and target_unique_values > 20
            and initial_task == TASK_CLASSIFICATION
        ):
            return TASK_REGRESSION

        # Trust the LLM if no override triggered
        return initial_task if initial_task in SUPPORTED_TASKS else TASK_CLASSIFICATION

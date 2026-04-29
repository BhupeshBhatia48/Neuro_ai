"""
model_engine/evaluator.py
---------------------------
Evaluates trained models, selects the best one, and generates
performance plots using matplotlib + seaborn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

console = Console()


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    primary_metric: float   # single number used for ranking


class ModelEvaluator:
    """
    Evaluates all trained models, prints a leaderboard,
    picks the best model, and optionally saves performance plots.
    """

    def evaluate_all(
        self,
        trained_models: Dict[str, object],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
        plots_dir: Optional[Path] = None,
    ) -> Tuple[List[ModelResult], str]:
        """
        Parameters
        ----------
        trained_models : dict of {name: fitted_estimator}
        X_test, y_test : held-out test data
        task_type      : "classification" | "regression"
        plots_dir      : if provided, saves PNG plots here

        Returns
        -------
        (list_of_results sorted best->worst, best_model_name)
        """
        results: List[ModelResult] = []
        predictions: Dict[str, np.ndarray] = {}

        for name, model in trained_models.items():
            try:
                y_pred = model.predict(X_test)
                predictions[name] = y_pred
                metrics = self._compute_metrics(y_test, y_pred, task_type)
                primary = metrics.get(
                    "f1_score" if task_type == "classification" else "r2", 0.0
                )
                results.append(
                    ModelResult(name=name, metrics=metrics, primary_metric=primary)
                )
            except Exception as exc:
                logger.error(f"Evaluation failed for {name}: {exc}")

        results.sort(key=lambda r: r.primary_metric, reverse=True)
        self._print_leaderboard(results, task_type)

        best = results[0].name if results else list(trained_models.keys())[0]
        logger.success(
            f"🏆 Best model: {best}  (primary={results[0].primary_metric:.4f})"
        )

        # -- Generate plots if requested ---------------------------------------
        if plots_dir is not None:
            plots_dir = Path(plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)
            self.plot_leaderboard(results, task_type, plots_dir)
            if task_type == "classification" and best in predictions:
                self.plot_confusion_matrix(
                    y_test, predictions[best], best, plots_dir
                )
            elif task_type == "regression" and best in predictions:
                self.plot_residuals(
                    y_test, predictions[best], best, plots_dir
                )

        return results, best

    # -- Metric computation ----------------------------------------------------

    def _compute_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, task_type: str
    ) -> Dict[str, float]:
        if task_type == "classification":
            avg = "weighted"
            return {
                "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
                "precision": round(float(precision_score(
                    y_true, y_pred, average=avg, zero_division=0)), 4),
                "recall":    round(float(recall_score(
                    y_true, y_pred, average=avg, zero_division=0)), 4),
                "f1_score":  round(float(f1_score(
                    y_true, y_pred, average=avg, zero_division=0)), 4),
            }
        else:
            mse = mean_squared_error(y_true, y_pred)
            return {
                "rmse": round(float(np.sqrt(mse)), 4),
                "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
                "r2":   round(float(r2_score(y_true, y_pred)), 4),
            }

    # -- Rich leaderboard ------------------------------------------------------

    def _print_leaderboard(self, results: List[ModelResult], task_type: str):
        if not results:
            return
        metric_keys = list(results[0].metrics.keys())
        table = Table(title="📊 Model Leaderboard", show_lines=True)
        table.add_column("Rank",  style="cyan",  width=5)
        table.add_column("Model", style="green")
        for mk in metric_keys:
            table.add_column(mk.upper(), justify="right")
        for i, r in enumerate(results, 1):
            row = [str(i), r.name] + [str(r.metrics[mk]) for mk in metric_keys]
            table.add_row(*row)
        console.print(table)

    # -- Matplotlib / Seaborn plots --------------------------------------------

    def plot_leaderboard(
        self,
        results: List[ModelResult],
        task_type: str,
        plots_dir: Path,
    ) -> Path:
        """
        Grouped bar chart comparing all models across all metrics.
        Saved as: plots_dir/leaderboard.png
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            metric_keys = list(results[0].metrics.keys())
            model_names = [r.name for r in results]

            data = {mk: [r.metrics[mk] for r in results] for mk in metric_keys}

            fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.5), 5))
            x = np.arange(len(model_names))
            width = 0.8 / len(metric_keys)
            palette = sns.color_palette("husl", len(metric_keys))

            for i, (mk, color) in enumerate(zip(metric_keys, palette)):
                offset = (i - len(metric_keys) / 2) * width + width / 2
                bars = ax.bar(x + offset, data[mk], width, label=mk.upper(), color=color)
                ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=15, ha="right")
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Score")
            ax.set_title(
                f"Model Comparison — {task_type.title()}",
                fontsize=13, fontweight="bold",
            )
            ax.legend(loc="upper right", fontsize=8)
            sns.despine()
            plt.tight_layout()

            dest = plots_dir / "leaderboard.png"
            fig.savefig(dest, dpi=150)
            plt.close(fig)
            logger.success(f"Leaderboard plot saved -> {dest}")
            return dest

        except Exception as exc:
            logger.warning(f"plot_leaderboard failed: {exc}")
            return plots_dir / "leaderboard.png"

    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        plots_dir: Path,
    ) -> Path:
        """
        Seaborn heatmap confusion matrix for classification tasks.
        Saved as: plots_dir/confusion_matrix_<model_name>.png
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = confusion_matrix(y_true, y_pred)
            labels = sorted(pd.Series(y_true).unique())

            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8),
                                            max(5, len(labels) * 0.7)))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax,
            )
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("Actual",    fontsize=11)
            ax.set_title(
                f"Confusion Matrix — {model_name}",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()

            dest = plots_dir / f"confusion_matrix_{model_name}.png"
            fig.savefig(dest, dpi=150)
            plt.close(fig)
            logger.success(f"Confusion matrix saved -> {dest}")
            return dest

        except Exception as exc:
            logger.warning(f"plot_confusion_matrix failed: {exc}")
            return plots_dir / f"confusion_matrix_{model_name}.png"

    def plot_residuals(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        plots_dir: Path,
    ) -> Path:
        """
        Residual plot (actual vs predicted + residual distribution) for regression.
        Saved as: plots_dir/residuals_<model_name>.png
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            y_true_arr = np.array(y_true)
            residuals  = y_true_arr - y_pred

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left: Actual vs Predicted scatter
            axes[0].scatter(y_pred, y_true_arr, alpha=0.5, s=15, color="steelblue")
            _min = min(y_pred.min(), y_true_arr.min())
            _max = max(y_pred.max(), y_true_arr.max())
            axes[0].plot([_min, _max], [_min, _max], "r--", linewidth=1.5)
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("Actual")
            axes[0].set_title("Actual vs Predicted", fontweight="bold")
            sns.despine(ax=axes[0])

            # Right: Residual distribution
            sns.histplot(residuals, kde=True, ax=axes[1], color="steelblue")
            axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
            axes[1].set_xlabel("Residual")
            axes[1].set_title("Residual Distribution", fontweight="bold")
            sns.despine(ax=axes[1])

            fig.suptitle(
                f"Regression Diagnostics — {model_name}",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()

            dest = plots_dir / f"residuals_{model_name}.png"
            fig.savefig(dest, dpi=150)
            plt.close(fig)
            logger.success(f"Residual plot saved -> {dest}")
            return dest

        except Exception as exc:
            logger.warning(f"plot_residuals failed: {exc}")
            return plots_dir / f"residuals_{model_name}.png"

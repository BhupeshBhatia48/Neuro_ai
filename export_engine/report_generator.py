"""
export_engine/report_generator.py
------------------------------------
Generates a human-readable training_report.txt AND a seaborn
performance summary PNG, both saved to the models/ directory.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional

from loguru import logger

from config.settings import MODELS_DIR
from config.constants import REPORT_FILENAME
from model_engine.evaluator import ModelResult
from data_analysis.dataset_analyzer import DatasetProfile


class ReportGenerator:
    """
    Writes a plain-text training report and an optional visual summary.
    """

    def generate(
        self,
        idea: str,
        profile: DatasetProfile,
        results: List[ModelResult],
        best_model_name: str,
        model_path: Path,
        pipeline_path: Path,
        plot: bool = True,
    ) -> Path:
        """
        Parameters
        ----------
        plot : if True, also saves a performance bar chart PNG.

        Returns the Path of the .txt report.
        """
        report_path = MODELS_DIR / REPORT_FILENAME

        lines = [
            "=" * 60,
            "  NEURO AI - TRAINING REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "IDEA",
            "-" * 40,
            idea,
            "",
            "DATASET PROFILE",
            "-" * 40,
            f"  Rows          : {profile.num_rows:,}",
            f"  Features      : {profile.num_features}",
            f"  Task type     : {profile.task_type}",
            f"  Size bucket   : {profile.size_bucket}",
            f"  Target column : {profile.target_column}",
            f"  Missing ratio : {profile.missing_value_ratio:.1%}",
            "",
            "MODEL LEADERBOARD",
            "-" * 40,
        ]

        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r.name}")
            for k, v in r.metrics.items():
                lines.append(f"       {k:<12}: {v}")

        lines += [
            "",
            "BEST MODEL",
            "-" * 40,
            f"  {best_model_name}  "
            f"(primary metric: {results[0].primary_metric:.4f})",
            "",
            "EXPORTED FILES",
            "-" * 40,
            f"  Model    : {model_path}",
            f"  Pipeline : {pipeline_path}",
            "",
            "=" * 60,
        ]

        # Fix for Windows: always write with utf-8 encoding explicitly
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.success(f"Training report saved -> {report_path}")

        if plot:
            self._plot_performance(results, profile.task_type or "classification")

        return report_path

    # -- Visual summary --------------------------------------------------------

    def _plot_performance(
        self, results: List[ModelResult], task_type: str
    ) -> Optional[Path]:
        """
        Horizontal bar chart of the primary metric for each model.
        Saved as models/performance_summary.png
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            primary_key = "f1_score" if task_type == "classification" else "r2"
            model_names = [r.name for r in results]
            scores      = [r.metrics.get(primary_key, r.primary_metric) for r in results]

            palette = sns.color_palette("viridis_r", len(model_names))
            fig, ax = plt.subplots(figsize=(8, max(3, len(model_names) * 0.7)))

            bars = ax.barh(model_names, scores, color=palette, edgecolor="white")

            # Value labels on bars
            for bar, score in zip(bars, scores):
                ax.text(
                    bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}", va="center", ha="left", fontsize=9,
                )

            ax.set_xlim(0, min(max(scores) * 1.2, 1.05))
            ax.set_xlabel(primary_key.upper(), fontsize=11)
            ax.set_title(
                f"Performance Summary ({task_type.title()})\n"
                f"Metric: {primary_key.upper()}",
                fontsize=12, fontweight="bold",
            )
            ax.invert_yaxis()   # best model at top
            sns.despine()
            plt.tight_layout()

            dest = MODELS_DIR / "performance_summary.png"
            fig.savefig(dest, dpi=150)
            plt.close(fig)
            logger.success(f"Performance summary plot saved -> {dest}")
            return dest

        except Exception as exc:
            logger.warning(f"_plot_performance failed: {exc}")
            return None

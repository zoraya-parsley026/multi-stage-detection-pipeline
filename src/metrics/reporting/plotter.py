"""
Visual Reporting & Metric Plotting Engine
=========================================
Self-contained, modular plotting service for object detection evaluation.
Generates publication-quality figures for confusion matrices, precision-recall curves,
cascade filter efficiency, and multi-stage latency breakdowns.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Ensure headless execution in server and CI/CD environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.metrics.evaluator import EvaluationReport
from src.metrics.metric_result import MetricResult


class MetricPlotter:
    """
    Visual reporting service providing publication-quality diagnostic plots:
    1. Detection Confusion Matrix (heatmap with background FN/FP).
    2. Precision-Recall Curves (single or multi-pipeline overlay).
    3. Cascade Filter Efficiency (boxes entering vs retained vs rejected).
    4. Stage Latency Breakdown (node-by-node runtime & FPS).
    5. Pipeline Comparison Dashboard (accuracy vs latency Pareto analysis).

    Parameters
    ----------
    output_dir:
        Target directory for saved figures. Defaults to 'runs/plots'.
    """

    PALETTE = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#e67e22"]

    def __init__(self, output_dir: str | Path = "runs/plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._apply_style()

    # -----------------------------------------------------------------------
    # Styling & File Persistence
    # -----------------------------------------------------------------------
    @staticmethod
    def _apply_style() -> None:
        """Applies clean, publication-ready plot styling."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "figure.titlesize": 13,
        })

    def _save_figure(self, fig: matplotlib.figure.Figure, filename: str, dpi: int = 200) -> Path:
        """Saves figure to disk, closes figure context, and returns Path."""
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # -----------------------------------------------------------------------
    # 1. Confusion Matrix Component
    # -----------------------------------------------------------------------
    def plot_confusion_matrix(
        self,
        evaluator_or_matrix: EvaluationReport | list[MetricResult] | dict[Any, dict[Any, int]],
        class_names: dict[Any, str] | Sequence[str] | None = None,
        filename: str = "confusion_matrix.png",
        normalize: bool = True,
        title: str = "Detection Confusion Matrix",
    ) -> Path:
        """
        Renders a Seaborn confusion matrix heatmap including background FP/FN categories.

        Parameters
        ----------
        evaluator_or_matrix:
            An EvaluationReport, a List[MetricResult], or a raw confusion matrix dictionary.
        class_names:
            Mapping or list from class_id to human-readable label.
        filename:
            Output image filename.
        normalize:
            If True, normalizes counts by ground-truth row totals into percentages.
        title:
            Title displayed at the top of the heatmap.

        Returns
        -------
        Path:
            Path to the saved figure.
        """
        raw_matrix = self._extract_confusion_matrix(evaluator_or_matrix) or {}

        # Collect unique non-background class IDs
        classes_set = set(raw_matrix.keys())
        for row in raw_matrix.values():
            classes_set.update(row.keys())
        classes_set.discard("Background")

        try:
            sorted_classes = sorted(classes_set, key=lambda x: int(x))
        except (ValueError, TypeError):
            sorted_classes = sorted(classes_set, key=lambda x: str(x))

        all_categories = list(sorted_classes) + ["Background"]
        num_classes = len(all_categories)

        # Build 2D matrix: rows = Ground Truth, columns = Predictions
        matrix_array = np.zeros((num_classes, num_classes), dtype=np.float64)
        for i, gt_cls in enumerate(all_categories):
            row_data = raw_matrix.get(gt_cls, {})
            for j, pred_cls in enumerate(all_categories):
                matrix_array[i, j] = row_data.get(pred_cls, 0)

        # Normalization
        if normalize:
            row_sums = matrix_array.sum(axis=1, keepdims=True)
            plot_array = np.divide(
                matrix_array,
                row_sums,
                out=np.zeros_like(matrix_array),
                where=(row_sums > 0),
            )
            fmt = ".2f"
        else:
            plot_array = matrix_array
            fmt = ".0f"

        # Construct display labels
        display_labels = []
        for cat in all_categories:
            if cat == "Background":
                display_labels.append("Background")
            elif class_names is not None:
                if isinstance(class_names, dict):
                    display_labels.append(str(class_names.get(cat, cat)))
                elif isinstance(class_names, (list, tuple)) and isinstance(cat, int) and cat < len(class_names):
                    display_labels.append(str(class_names[cat]))
                else:
                    display_labels.append(str(cat))
            else:
                display_labels.append(f"Class {cat}")

        fig_size = max(6, min(14, len(all_categories) * 0.9))
        fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size), tight_layout=True)

        sns.heatmap(
            plot_array,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=display_labels,
            yticklabels=display_labels,
            cbar=True,
            square=True,
            ax=ax,
            linewidths=0.5,
            linecolor="lightgrey",
        )

        ax.set_xlabel("Predicted Class", fontweight="bold", labelpad=10)
        ax.set_ylabel("True Ground Truth Class", fontweight="bold", labelpad=10)
        ax.set_title(title, fontweight="bold", pad=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        return self._save_figure(fig, filename)

    # -----------------------------------------------------------------------
    # 2. Precision-Recall Curve Component
    # -----------------------------------------------------------------------
    def plot_pr_curve(
        self,
        evaluators: EvaluationReport | Sequence[EvaluationReport],
        iou_threshold: float = 0.5,
        filename: str = "pr_curve.png",
        title: str | None = None,
    ) -> Path:
        """
        Plots Precision-Recall curves with multi-evaluator performance overlay.

        Parameters
        ----------
        evaluators:
            Single EvaluationReport or list of EvaluationReports to compare.
        iou_threshold:
            IoU threshold for true positive matches (default: 0.5).
        filename:
            Output image filename.
        title:
            Title displayed on the chart.

        Returns
        -------
        Path:
            Path to the saved figure.
        """
        if isinstance(evaluators, EvaluationReport):
            evaluators = [evaluators]

        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

        for idx, ev in enumerate(evaluators):
            recalls, precisions, ap = self._compute_pr_curve_data(ev, iou_threshold=iou_threshold)
            color = self.PALETTE[idx % len(self.PALETTE)]
            label = f"{ev.tag} (pooled AP@{int(iou_threshold * 100)} = {ap:.3f})"
            ax.plot(recalls, precisions, label=label, color=color, linewidth=2.2)

        ax.set_xlabel("Recall", fontweight="bold")
        ax.set_ylabel("Precision", fontweight="bold")
        ax.set_xlim([0.0, 1.02])
        ax.set_ylim([0.0, 1.02])
        chart_title = title or f"Precision-Recall Curve (IoU = {iou_threshold:.2f})"
        ax.set_title(chart_title, fontweight="bold", pad=12)
        ax.legend(loc="lower left", frameon=True, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.6)

        return self._save_figure(fig, filename)

    # -----------------------------------------------------------------------
    # 3. Cascade Filter Efficiency Component
    # -----------------------------------------------------------------------
    def plot_filter_efficiency(
        self,
        evaluator: EvaluationReport,
        filename: str = "filter_efficiency.png",
        title: str = "Multi-Stage Cascade Filter Efficiency",
    ) -> Path:
        """
        Renders candidate bounding-box reduction through cascade filtering:
        entering detections -> retained predictions -> rejected predictions.
        """
        results = evaluator.results
        entering = next((r.value for r in results if r.name == "detections_entering"), None)
        kept = next((r.value for r in results if r.name == "detections_kept"), None)
        rejected = next((r.value for r in results if r.name == "detections_rejected"), None)
        retention = next((r.value for r in results if r.name == "retention_rate"), 0.0)

        if entering is None:
            entering = sum(r["entering"] for r in evaluator.system_data)
            kept = sum(r["kept"] for r in evaluator.system_data)
            rejected = entering - kept
            retention = kept / entering if entering else 0.0

        categories = ["Entering Boxes\n(Baseline)", "Retained Boxes\n(Passed)", "Rejected Boxes\n(Filtered Out)"]
        counts = [int(entering), int(kept), int(rejected)]
        colors = ["#3498db", "#2ecc71", "#e74c3c"]

        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        bars = ax.bar(categories, counts, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / entering * 100.0) if entering > 0 else 0.0
            ax.annotate(
                f"{count:,}\n({percentage:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_ylabel("Bounding Box Count", fontweight="bold")
        rejection_rate = (1.0 - retention) * 100.0
        ax.set_title(
            f"{title}\nRetention Rate: {retention * 100:.1f}% | Rejection Rate: {rejection_rate:.1f}%",
            fontweight="bold",
            pad=12,
        )
        ax.set_ylim([0, max(max(counts) * 1.2, 5.0) if counts else 10.0])
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        return self._save_figure(fig, filename)

    # -----------------------------------------------------------------------
    # 4. Pipeline Latency Breakdown Component
    # -----------------------------------------------------------------------
    def plot_latency_breakdown(
        self,
        evaluator: EvaluationReport,
        filename: str = "latency_breakdown.png",
        title: str = "Pipeline Latency Breakdown by Node",
    ) -> Path:
        """
        Plots a horizontal bar chart of the runtime latency breakdown across
        pipeline nodes (e.g. YOLOv7, Cropper, Classifier, LogicFilter).
        """
        node_latencies: dict[str, list[float]] = {}
        total_latencies: list[float] = []

        for record in evaluator.system_data:
            lat = record["latency"]
            total_latencies.append(lat.get("total_ms", 0.0))

            for k, v in lat.items():
                if k in ("total_ms", "detector_ms", "validator_ms") or not isinstance(v, (int, float)):
                    continue
                clean_name = k.replace("node_", "").replace("_ms", "")
                if clean_name not in node_latencies:
                    node_latencies[clean_name] = []
                node_latencies[clean_name].append(v)

        if not node_latencies:
            detector_times = []
            validator_times = []
            for record in evaluator.system_data:
                lat = record.get("batch_data", {}).get("latency", {})
                detector_times.append(lat.get("detector_ms", 0.0))
                validator_times.append(lat.get("validator_ms", 0.0))
            node_latencies["Detector"] = detector_times or [0.0]
            node_latencies["Validator"] = validator_times or [0.0]

        avg_latencies = {node: float(np.mean(vals)) for node, vals in node_latencies.items()}
        total_avg = sum(avg_latencies.values()) if avg_latencies else 1.0

        sorted_items = sorted(avg_latencies.items(), key=lambda x: x[1], reverse=False)
        nodes = [item[0] for item in sorted_items]
        times = [item[1] for item in sorted_items]

        fig, ax = plt.subplots(figsize=(8, max(4, len(nodes) * 0.8)), tight_layout=True)
        colors = sns.color_palette("mako", len(nodes))
        bars = ax.barh(nodes, times, color=colors, edgecolor="black", linewidth=0.8)

        for bar, time_val in zip(bars, times):
            width = bar.get_width()
            pct = (time_val / total_avg * 100.0) if total_avg > 0 else 0.0
            ax.annotate(
                f" {time_val:.2f} ms ({pct:.1f}%)",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontweight="bold",
            )

        ax.set_xlabel("Latency (ms per frame)", fontweight="bold")
        fps = (1000.0 / total_avg) if total_avg > 0 else 0.0
        ax.set_title(f"{title} (Total: {total_avg:.2f} ms | FPS: {fps:.1f})", fontweight="bold", pad=12)
        ax.set_xlim([0, max(max(times) * 1.35, 5.0) if times else 10.0])
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        return self._save_figure(fig, filename)

    # -----------------------------------------------------------------------
    # 5. Pipeline Comparison Dashboard Component
    # -----------------------------------------------------------------------
    def plot_pipeline_comparison(
        self,
        evaluators: Sequence[EvaluationReport],
        filename: str = "pipeline_comparison.png",
        title: str = "Pipeline Comparative Trade-off Analysis",
    ) -> Path:
        """
        Creates a 4-panel comparison dashboard across evaluated pipelines:
        Panel 1: mAP@50 comparison.
        Panel 2: False Positive count comparison (showing FP reduction).
        Panel 3: Average latency comparison (ms).
        Panel 4: Latency vs. mAP Pareto trade-off scatter plot.
        """
        names: list[str] = []
        maps: list[float] = []
        fps: list[int] = []
        latencies: list[float] = []

        for ev in evaluators:
            results = {r.name: r.value for r in ev.results}
            names.append(ev.tag)
            maps.append(results.get("mAP@50", results.get("mAP", 0.0)))
            fps.append(int(results.get("fp_count", 0)))
            latencies.append(results.get("avg_latency_ms", 0.0))

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), tight_layout=True)
        colors = self.PALETTE[:len(evaluators)]

        # Panel 1: mAP@50
        ax1 = axes[0, 0]
        bars1 = ax1.bar(names, maps, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
        for b, v in zip(bars1, maps):
            ax1.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontweight="bold")
        ax1.set_title("Detection Accuracy (mAP@50)", fontweight="bold")
        ax1.set_ylabel("mAP@50")
        ax1.set_ylim([0, max(max(maps) * 1.2, 1.0) if maps else 1.0])
        ax1.grid(axis="y", linestyle="--", alpha=0.6)

        # Panel 2: False Positives
        ax2 = axes[0, 1]
        bars2 = ax2.bar(names, fps, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
        for b, v in zip(bars2, fps):
            ax2.annotate(f"{v:,}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontweight="bold")
        ax2.set_title("False Positives (Lower is Better)", fontweight="bold")
        ax2.set_ylabel("FP Count")
        ax2.set_ylim([0, max(max(fps) * 1.2, 5.0) if fps else 10.0])
        ax2.grid(axis="y", linestyle="--", alpha=0.6)

        # Panel 3: Latency
        ax3 = axes[1, 0]
        bars3 = ax3.bar(names, latencies, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
        for b, v in zip(bars3, latencies):
            ax3.annotate(f"{v:.1f} ms", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontweight="bold")
        ax3.set_title("Average Latency (Lower is Better)", fontweight="bold")
        ax3.set_ylabel("Latency (ms)")
        ax3.set_ylim([0, max(max(latencies) * 1.2, 10.0) if latencies else 10.0])
        ax3.grid(axis="y", linestyle="--", alpha=0.6)

        # Panel 4: Pareto Trade-off
        ax4 = axes[1, 1]
        for name, lat, map_val, col in zip(names, latencies, maps, colors):
            ax4.scatter(lat, map_val, color=col, s=140, edgecolors="black", zorder=5)
            ax4.annotate(f" {name}", xy=(lat, map_val), xytext=(5, 5), textcoords="offset points", fontweight="bold")
        ax4.set_title("Latency vs. Accuracy Trade-off", fontweight="bold")
        ax4.set_xlabel("Latency (ms)")
        ax4.set_ylabel("mAP@50")
        ax4.grid(True, linestyle="--", alpha=0.6)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        return self._save_figure(fig, filename)

    # -----------------------------------------------------------------------
    # Extraction Helpers
    # -----------------------------------------------------------------------
    def _extract_confusion_matrix(
        self,
        source: EvaluationReport | list[MetricResult] | dict[Any, dict[Any, int]],
    ) -> dict[Any, dict[Any, int]] | None:
        """Extracts the raw confusion matrix dictionary from multiple input types."""
        if isinstance(source, dict):
            return source

        if isinstance(source, EvaluationReport):
            results = source.results
            matrix_res = next((r.value for r in results if r.name in ("detection_confusion_matrix", "classification_confusion_matrix")), None)
            if matrix_res is not None:
                return matrix_res
            return source.confusion_matrix

        if isinstance(source, list):
            matrix_res = next((r.value for r in source if r.name == "detection_confusion_matrix"), None)
            return matrix_res

        return None

    def _compute_pr_curve_data(
        self,
        evaluator: EvaluationReport,
        iou_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Computes continuous interpolated Precision-Recall curve and AP for an evaluator."""
        if iou_threshold not in evaluator.pr_curves:
            raise ValueError("PR threshold was not included when creating the evaluation report.")
        return evaluator.pr_curves[iou_threshold]

"""
Tabular Reporting Subsystem
===========================
Provides component-based table formatting and automated differential metric calculations.
"""

from __future__ import annotations

from src.metrics.evaluator import EvaluationReport
from src.metrics.metric_result import MetricResult


class DeltaCalculator:
    """
    Computes absolute and relative differences between baseline results and candidate pipelines.
    """

    @staticmethod
    def calculate(
        key: str,
        index: int,
        current_res: dict[str, MetricResult],
        baseline_res: dict[str, MetricResult],
    ) -> str:
        """Calculates formatted delta string with signed prefix (+/-) or dash for baseline."""
        if index == 0:
            return "—"

        if key == "ΔFP":
            b = baseline_res.get("fp_count")
            c = current_res.get("fp_count")
            if b is not None and c is not None:
                return f"{int(c.value) - int(b.value):+d}"
            return "—"

        if key == "ΔmAP":
            b = baseline_res.get("mAP")
            c = current_res.get("mAP")
            if b is not None and c is not None:
                return f"{float(c.value) - float(b.value):+7.3f}"
            return "—"

        if key == "ΔF1":
            b = baseline_res.get("f1_score")
            c = current_res.get("f1_score")
            if b is not None and c is not None:
                return f"{float(c.value) - float(b.value):+7.3f}"
            return "—"

        if key in ("Rel_Latency_Inc", "ΔLat(%)"):
            b = baseline_res.get("avg_latency_ms")
            c = current_res.get("avg_latency_ms")
            if b and c and b.value > 0:
                rel = (float(c.value) - float(b.value)) / float(b.value) * 100.0
                return f"{rel:+8.1f}"
            return "—"

        if key.startswith("Δ"):
            base_key = key[1:]
            b = baseline_res.get(base_key)
            c = current_res.get(base_key)
            if b is not None and c is not None and isinstance(b.value, (int, float)):
                diff = float(c.value) - float(b.value)
                return f"{diff:+.3f}"

        return "—"


class TableFormatter:
    """
    Handles column width resolution, cell formatting, and text alignment.
    """

    TAG_WIDTH = 25

    @classmethod
    def get_column_spec(cls, key: str) -> tuple[str, int]:
        """Returns the header label and column width for a given metric key."""
        if key == "mAP@50":
            return "mAP@50", 8
        if key == "mAP":
            return "mAP", 8
        if key in ("ΔFP", "ΔmAP", "ΔF1"):
            return key, 7
        if key in ("Rel_Latency_Inc", "ΔLat(%)"):
            return "ΔLat(%)", 8
        return key, max(10, len(key))

    @classmethod
    def format_header(cls, display_keys: list[str]) -> str:
        """Builds the header row and horizontal separator line."""
        header_parts = [f"{'tag':<{cls.TAG_WIDTH}}"]
        widths = []
        for key in display_keys:
            label, width = cls.get_column_spec(key)
            widths.append(width)
            header_parts.append(f"{label:<{width}}")

        header_line = " | ".join(header_parts)
        total_width = cls.TAG_WIDTH + 3 + sum(w + 3 for w in widths)
        separator_line = "-" * total_width
        return f"{header_line}\n{separator_line}"

    @classmethod
    def format_cell(cls, key: str, width: int, result_dict: dict[str, MetricResult]) -> str:
        """Formats a standard metric cell value according to its column specification."""
        val_obj = result_dict.get(key)
        if not val_obj:
            return f"{'':<{width}}"

        if key in ("mAP", "mAP@50"):
            val_str = val_obj.formatted()
            return f"{val_str:<{width}}"

        if isinstance(val_obj.value, dict):
            return f"{'<matrix>':<{width}}"

        formatted_val = val_obj.formatted()
        return f"{str(formatted_val):<{width}}"


class Reporter:
    """
    High-level facade coordinating table construction and metric comparison.
    """

    @staticmethod
    def compare(evaluators: list[EvaluationReport], display_keys: list[str]) -> str:
        """
        Generates an aligned comparison table across multiple EvaluationReports.

        Parameters
        ----------
        evaluators:
            List of EvaluationReport instances. The first evaluator is treated as the baseline.
        display_keys:
            List of column metric names or differential keys (e.g. 'mAP', 'ΔFP', 'Rel_Latency_Inc').

        Returns
        -------
        str:
            Formatted comparison table.
        """
        if not evaluators:
            return ""

        evaluator_results = [ev.results for ev in evaluators]
        result_dicts: list[dict[str, MetricResult]] = [
            {r.name: r for r in results} for results in evaluator_results
        ]

        baseline = result_dicts[0]
        lines = [TableFormatter.format_header(display_keys)]

        for i, res in enumerate(result_dicts):
            tag_name = res["tag"].formatted() if "tag" in res else f"EvaluationReport_{i}"
            row_parts = [f"{tag_name:<{TableFormatter.TAG_WIDTH}}"]

            for key in display_keys:
                _, width = TableFormatter.get_column_spec(key)

                if key in ("ΔFP", "ΔmAP", "ΔF1", "Rel_Latency_Inc", "ΔLat(%)") or key.startswith("Δ"):
                    delta_str = DeltaCalculator.calculate(key, i, res, baseline)
                    row_parts.append(f"{delta_str:<{width}}")
                else:
                    cell_str = TableFormatter.format_cell(key, width, res)
                    row_parts.append(cell_str)

            lines.append(" | ".join(row_parts))

        return "\n".join(lines)

"""Stage 9 — Report Generator.

``generate_report(folder, ctx)`` creates ``folder`` (and ``folder/charts``)
and writes three things: a self-contained interactive ``report.html``
(Jinja2 + inline Plotly, no external network requests), a ``report.json``
with every number the HTML shows, and one static PNG per chart under
``charts/`` for embedding elsewhere (docs, slides, dashboards). It returns
the same dict that's written to ``report.json`` so callers can use it
in-memory without re-reading the file.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline.offline import get_plotlyjs
from sklearn.metrics import confusion_matrix, roc_curve

from smarttab import __version__
from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.hardware.profiler import HardwareProfile
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.logging_utils import get_logger

logger = get_logger()

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_CHART_IMAGE_SIZE = dict(width=900, height=480, scale=2)


@dataclasses.dataclass
class ReportContext:
    dataset_profile: DatasetProfile
    hardware_profile: HardwareProfile
    resource_plan: ResourcePlan
    model_name: str
    task_type: TaskType
    best_params: dict
    primary_metric: str
    metrics: dict
    feature_importance: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    timings: dict
    model_size_bytes: int
    n_final_features: int
    y_proba: np.ndarray | None = None
    class_labels: list | None = None
    shap_importance: pd.DataFrame | None = None
    ensemble_info: dict | None = None
    decision_threshold: float = 0.5
    reject_threshold: float = 0.0
    per_label_thresholds: list[float] | None = None
    objective: str = "mcc"
    multi_threshold_ensemble: bool = False
    threshold_ladder: list[dict] | None = None
    notes: list[str] = dataclasses.field(default_factory=list)


def generate_report(folder: str, ctx: ReportContext) -> dict:
    folder_path = Path(folder)
    charts_dir = folder_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    charts = _build_charts(ctx)
    chart_files = _export_chart_images(charts, charts_dir)

    report_dict = _build_report_dict(ctx, chart_files)
    (folder_path / "report.json").write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")

    html = _render_html(ctx, charts, report_dict)
    (folder_path / "report.html").write_text(html, encoding="utf-8")

    report_dict["_paths"] = {
        "folder": str(folder_path),
        "html": str(folder_path / "report.html"),
        "json": str(folder_path / "report.json"),
        "charts_dir": str(charts_dir),
    }
    return report_dict


def _build_charts(ctx: ReportContext) -> dict[str, go.Figure]:
    charts: dict[str, go.Figure] = {"feature_importance": _feature_importance_chart(ctx.feature_importance)}

    if ctx.shap_importance is not None and not ctx.shap_importance.empty:
        charts["shap_importance"] = _shap_chart(ctx.shap_importance)

    if ctx.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        charts["diagnostic"] = _classification_diagnostic_chart(ctx)
    elif ctx.task_type is TaskType.MULTILABEL:
        charts["diagnostic"] = _multilabel_diagnostic_chart(ctx)
    elif ctx.task_type is TaskType.MULTIOUTPUT_REGRESSION:
        charts["diagnostic"] = _multioutput_diagnostic_chart(ctx)
    else:
        # REGRESSION, and RANKING (predicted score vs. true relevance is still an informative scatter)
        charts["diagnostic"] = _regression_diagnostic_chart(ctx.y_true, ctx.y_pred)

    if ctx.multi_threshold_ensemble and ctx.threshold_ladder:
        if ctx.task_type is TaskType.MULTICLASS:
            charts["threshold_ladder"] = _multiclass_threshold_ladder_chart(ctx.threshold_ladder)
        elif ctx.task_type is TaskType.MULTILABEL:
            charts["threshold_ladder"] = _multilabel_threshold_ladder_chart(ctx.threshold_ladder)
        else:
            charts["threshold_ladder"] = _threshold_ladder_chart(ctx.threshold_ladder)

    charts["metrics"] = _metrics_chart(ctx.metrics)
    charts["timing"] = _timing_chart(ctx.timings)

    memory_keys = {k: v for k, v in ctx.timings.items() if "memory" in k}
    if memory_keys:
        charts["memory"] = _memory_chart(memory_keys, ctx.hardware_profile.ram.total_mb)

    return charts


def _export_chart_images(charts: dict[str, go.Figure], charts_dir: Path) -> dict[str, str]:
    chart_files: dict[str, str] = {}
    for name, fig in charts.items():
        png_path = charts_dir / f"{name}.png"
        try:
            fig.write_image(str(png_path), **_CHART_IMAGE_SIZE)
            chart_files[name] = f"charts/{name}.png"
        except Exception as exc:
            logger.debug("Could not export chart '%s' to PNG (%s); HTML will still include it inline.", name, exc)
    return chart_files


def _render_html(ctx: ReportContext, charts: dict[str, go.Figure], report_dict: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
    )
    template = env.get_template("report.html.j2")

    # The plotly.js library is injected once, in <head> (see the template), rather than piggy-
    # backing on whichever chart happens to be built first — the previous approach embedded the
    # full library inline in the *first* chart processed here, but that chart isn't necessarily
    # the first one placed in the rendered document (e.g. "Evaluation Metrics"/"Speed"/"Memory"
    # all appear before "Feature Importance" in the template), so their Plotly.newPlot() calls
    # ran before the library existed and silently rendered nothing.
    chart_html = {name: fig.to_html(full_html=False, include_plotlyjs=False) for name, fig in charts.items()}

    return template.render(
        plotlyjs_library=get_plotlyjs(),
        dataset_profile=ctx.dataset_profile,
        hardware_profile=ctx.hardware_profile,
        resource_plan=ctx.resource_plan,
        model_name=ctx.model_name,
        task_type=ctx.task_type.value,
        best_params=ctx.best_params,
        primary_metric=ctx.primary_metric,
        metrics=ctx.metrics,
        chart_html=chart_html,
        timings=ctx.timings,
        model_size_bytes=ctx.model_size_bytes,
        n_final_features=ctx.n_final_features,
        class_labels=ctx.class_labels,
        ensemble_info=ctx.ensemble_info,
        decision_threshold=ctx.decision_threshold,
        reject_threshold=ctx.reject_threshold,
        per_label_thresholds=ctx.per_label_thresholds,
        objective=ctx.objective,
        multi_threshold_ensemble=ctx.multi_threshold_ensemble,
        threshold_ladder=ctx.threshold_ladder,
        threshold_ladder_summary=report_dict.get("threshold_ladder_summary"),
        notes=ctx.notes,
        library_version=__version__,
        generated_at=report_dict["generated_at"],
    )


def _build_report_dict(ctx: ReportContext, chart_files: dict[str, str]) -> dict:
    dataset_profile = dataclasses.asdict(ctx.dataset_profile)
    dataset_profile["task_type"] = ctx.dataset_profile.task_type.value

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "smarttab_version": __version__,
        "model_name": ctx.model_name,
        "task_type": ctx.task_type.value,
        "primary_metric": ctx.primary_metric,
        "best_params": ctx.best_params,
        "metrics": ctx.metrics,
        "n_final_features": ctx.n_final_features,
        "class_labels": ctx.class_labels,
        "model_size_bytes": ctx.model_size_bytes,
        "decision_threshold": ctx.decision_threshold,
        "reject_threshold": ctx.reject_threshold,
        "per_label_thresholds": ctx.per_label_thresholds,
        "objective": ctx.objective,
        "multi_threshold_ensemble": ctx.multi_threshold_ensemble,
        "threshold_ladder": ctx.threshold_ladder,
        "threshold_ladder_summary": _threshold_ladder_summary(ctx.task_type, ctx.threshold_ladder),
        "timings": ctx.timings,
        "feature_importance": ctx.feature_importance.to_dict(orient="records"),
        "shap_importance": ctx.shap_importance.to_dict(orient="records") if ctx.shap_importance is not None else None,
        "ensemble_info": ctx.ensemble_info,
        "notes": ctx.notes,
        "dataset_profile": dataset_profile,
        "hardware_profile": dataclasses.asdict(ctx.hardware_profile),
        "resource_plan": dataclasses.asdict(ctx.resource_plan),
        "chart_files": chart_files,
    }


def _threshold_ladder_summary(task_type: TaskType, ladder) -> dict | None:
    if not ladder:
        return None
    if task_type is TaskType.MULTICLASS:
        keys = ("coverage", "accuracy_on_accepted", "accuracy_overall")
        return {f"avg_{k}": float(np.mean([point[k] for point in ladder])) for k in keys}
    if task_type is TaskType.MULTILABEL:
        # ladder is one list per label; flatten across labels for a single overall summary
        # (the per-label breakdown is still fully available in threshold_ladder itself).
        keys = ("precision", "recall", "f1", "accuracy", "predicted_positive_rate")
        flat = [point for label_ladder in ladder for point in label_ladder]
        return {f"avg_{k}": float(np.mean([point[k] for point in flat])) for k in keys}
    keys = ("precision", "recall", "f1", "accuracy", "predicted_positive_rate")
    return {f"avg_{k}": float(np.mean([point[k] for point in ladder])) for k in keys}


def _multiclass_threshold_ladder_chart(ladder: list[dict]) -> go.Figure:
    thresholds = [point["threshold"] for point in ladder]
    fig = go.Figure()
    for metric, color in (("coverage", "#60a5fa"), ("accuracy_on_accepted", "#34d399"), ("accuracy_overall", "#fbbf24")):
        fig.add_trace(go.Scatter(
            x=thresholds, y=[point[metric] for point in ladder], mode="lines+markers",
            name=metric, line=dict(color=color), marker=dict(size=8),
        ))
    fig.update_layout(
        title="Reject-Threshold Ladder (Confidence Levels)", xaxis_title="reject threshold",
        yaxis_title="score", yaxis_range=[0, 1.05], template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10), height=420,
    )
    return fig


def _multilabel_threshold_ladder_chart(ladders: list[list[dict]]) -> go.Figure:
    """One ladder per label, but a single chart: average each metric across labels at each
    ladder level (level index, not raw threshold, since thresholds differ per label)."""
    n_models = len(ladders[0])
    levels = list(range(1, n_models + 1))
    fig = go.Figure()
    for metric, color in (("precision", "#60a5fa"), ("recall", "#f472b6"), ("f1", "#34d399"), ("accuracy", "#fbbf24")):
        avg_values = [float(np.mean([label_ladder[i][metric] for label_ladder in ladders])) for i in range(n_models)]
        fig.add_trace(go.Scatter(
            x=levels, y=avg_values, mode="lines+markers", name=f"avg {metric}",
            line=dict(color=color), marker=dict(size=8),
        ))
    fig.update_layout(
        title="Per-Label Threshold Ladder (averaged across labels)", xaxis_title="level (lenient -> strict)",
        yaxis_title="score", yaxis_range=[0, 1.05], template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10), height=420,
    )
    return fig


def _threshold_ladder_chart(ladder: list[dict]) -> go.Figure:
    thresholds = [point["threshold"] for point in ladder]
    fig = go.Figure()
    for metric, color in (("precision", "#60a5fa"), ("recall", "#f472b6"), ("f1", "#34d399"), ("accuracy", "#fbbf24")):
        fig.add_trace(go.Scatter(
            x=thresholds, y=[point[metric] for point in ladder], mode="lines+markers",
            name=metric, line=dict(color=color), marker=dict(size=8),
        ))
    fig.update_layout(
        title="Threshold Ladder (Confidence Levels)", xaxis_title="threshold",
        yaxis_title="score", yaxis_range=[0, 1.05], template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10), height=420,
    )
    return fig


def _feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    top = importance_df.head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker_color="#60a5fa"))
    fig.update_layout(
        title="Feature Importance", height=max(320, 24 * len(top)), margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark", xaxis_title="importance",
    )
    return fig


def _shap_chart(shap_df: pd.DataFrame) -> go.Figure:
    top = shap_df.head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["feature"], orientation="h", marker_color="#f472b6"))
    fig.update_layout(
        title="SHAP Feature Importance (mean |SHAP value|)", height=max(320, 24 * len(top)),
        margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark", xaxis_title="mean |SHAP value|",
    )
    return fig


def _classification_diagnostic_chart(ctx: ReportContext) -> go.Figure:
    if ctx.task_type is TaskType.BINARY and ctx.y_proba is not None:
        fpr, tpr, _ = roc_curve(ctx.y_true, ctx.y_proba[:, 1])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve", line=dict(color="#60a5fa")))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(dash="dash", color="#6b7280")))
        fig.update_layout(
            title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), height=420,
        )
        return fig

    labels = ctx.class_labels if ctx.class_labels is not None else sorted(pd.unique(ctx.y_true))
    cm = confusion_matrix(ctx.y_true, ctx.y_pred, labels=list(range(len(labels))) if ctx.class_labels else labels)
    fig = go.Figure(
        go.Heatmap(z=cm, x=[str(l) for l in labels], y=[str(l) for l in labels], colorscale="Blues", showscale=False, text=cm, texttemplate="%{text}")
    )
    fig.update_layout(
        title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual",
        template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), height=420,
    )
    return fig


def _multilabel_diagnostic_chart(ctx: ReportContext) -> go.Figure:
    labels = ctx.class_labels or [f"label_{i}" for i in range(ctx.y_true.shape[1])]
    per_label_acc = (ctx.y_true == ctx.y_pred).mean(axis=0)
    fig = go.Figure(go.Bar(
        x=[str(l) for l in labels], y=per_label_acc, marker_color="#60a5fa",
        text=[f"{v:.2%}" for v in per_label_acc], textposition="outside",
    ))
    fig.update_layout(
        title="Per-Label Accuracy", yaxis_title="accuracy", yaxis_range=[0, 1],
        template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), height=420,
    )
    return fig


def _multioutput_diagnostic_chart(ctx: ReportContext) -> go.Figure:
    n_outputs = ctx.y_true.shape[1]
    per_output_rmse = ctx.metrics.get("rmse_per_output")
    if not per_output_rmse:
        per_output_rmse = [float(np.sqrt(np.mean((ctx.y_true[:, i] - ctx.y_pred[:, i]) ** 2))) for i in range(n_outputs)]
    names = [f"output_{i}" for i in range(n_outputs)]
    fig = go.Figure(go.Bar(
        x=names, y=per_output_rmse, marker_color="#f472b6",
        text=[f"{v:.3f}" for v in per_output_rmse], textposition="outside",
    ))
    fig.update_layout(
        title="Per-Output RMSE", yaxis_title="RMSE", template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10), height=420,
    )
    return fig


def _regression_diagnostic_chart(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="predictions", marker=dict(size=5, opacity=0.6, color="#60a5fa")))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal", line=dict(dash="dash", color="#6b7280")))
    fig.update_layout(
        title="Predicted vs Actual", xaxis_title="Actual", yaxis_title="Predicted",
        template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), height=420,
    )
    return fig


def _metrics_chart(metrics: dict) -> go.Figure:
    # some metric dicts (e.g. multi-output regression's rmse_per_output) include a list-valued
    # breakdown alongside the scalar metrics; only scalars fit a one-bar-per-metric chart.
    items = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))][::-1]
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#34d399", text=[f"{v:.4f}" for v in values], textposition="outside"))
    fig.update_layout(
        title="Evaluation Metrics", height=max(320, 28 * len(items)), margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark", xaxis_title="value",
    )
    return fig


def _timing_chart(timings: dict) -> go.Figure:
    items = [(k, v) for k, v in timings.items() if "second" in k]
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color="#fbbf24", text=[f"{v:.3f}s" for v in values], textposition="outside"))
    fig.update_layout(
        title="Speed", yaxis_title="seconds", template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10), height=380,
    )
    return fig


def _memory_chart(memory_keys: dict, ram_total_mb: float) -> go.Figure:
    names = list(memory_keys.keys()) + ["total system RAM"]
    values = list(memory_keys.values()) + [ram_total_mb]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color="#a78bfa", text=[f"{v:.0f} MB" for v in values], textposition="outside"))
    fig.update_layout(
        title="Memory / RAM", yaxis_title="MB (log scale)", yaxis_type="log", template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10), height=380,
    )
    return fig

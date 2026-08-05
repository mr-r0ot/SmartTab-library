"""Self-contained HTML/JSON model reports with explicit static-export status."""

from __future__ import annotations

import dataclasses
import importlib.util
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
    class_labels: list | dict | None = None
    shap_importance: pd.DataFrame | None = None
    ensemble_info: dict | None = None
    decision_threshold: float = 0.5
    reject_threshold: float = 0.0
    per_label_thresholds: list[float] | None = None
    objective: str = "mcc"
    notes: list[str] = dataclasses.field(default_factory=list)
    static_charts: str | bool = "auto"
    multimodal_info: dict | None = None
    data_quality_report: dict | None = None
    cleaning_report: dict | None = None
    uncertainty_info: dict | None = None
    modality_dropout_info: dict | None = None
    evaluation_quality_report: dict | None = None
    evaluation_drift_report: dict | None = None


def generate_report(folder: str, ctx: ReportContext) -> dict:
    folder_path = Path(folder)
    charts_dir = folder_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts = _build_charts(ctx)
    chart_files, static_export = _export_chart_images(charts, charts_dir, ctx.static_charts)
    report_dict = _build_report_dict(ctx, chart_files, static_export)
    report_dict["_paths"] = {
        "folder": str(folder_path),
        "html": str(folder_path / "report.html"),
        "json": str(folder_path / "report.json"),
        "charts_dir": str(charts_dir),
    }
    html = _render_html(ctx, charts, report_dict)
    (folder_path / "report.html").write_text(html, encoding="utf-8")
    (folder_path / "report.json").write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return report_dict


def _build_charts(ctx: ReportContext) -> dict[str, go.Figure]:
    charts: dict[str, go.Figure] = {
        "feature_importance": _feature_importance_chart(ctx.feature_importance),
        "metrics": _metrics_chart(ctx.metrics),
        "timing": _timing_chart(ctx.timings),
    }
    if ctx.shap_importance is not None and not ctx.shap_importance.empty:
        charts["shap_importance"] = _shap_chart(ctx.shap_importance)
    if ctx.data_quality_report:
        charts["data_quality"] = _data_quality_chart(ctx.data_quality_report)
        if ctx.data_quality_report.get("missing_by_column"):
            charts["missingness"] = _missingness_chart(ctx.data_quality_report["missing_by_column"])
    if ctx.evaluation_drift_report:
        charts["drift"] = _drift_chart(ctx.evaluation_drift_report)
    if ctx.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        charts["diagnostic"] = _classification_diagnostic_chart(ctx)
    elif ctx.task_type is TaskType.MULTILABEL:
        charts["diagnostic"] = _multilabel_diagnostic_chart(ctx)
    elif ctx.task_type is TaskType.MULTIOUTPUT_REGRESSION:
        charts["diagnostic"] = _multioutput_diagnostic_chart(ctx)
    else:
        charts["diagnostic"] = _regression_diagnostic_chart(ctx.y_true, ctx.y_pred)
    memory = {key: value for key, value in ctx.timings.items() if "memory" in key}
    if memory:
        charts["memory"] = _memory_chart(memory, ctx.hardware_profile.ram.total_mb)
    if ctx.ensemble_info:
        diversity = ctx.ensemble_info.get("diversity_matrix") or {}
        candidates = ctx.ensemble_info.get("candidates") or []
        if diversity:
            charts["ensemble_diversity"] = _ensemble_diversity_chart(diversity)
        if candidates:
            charts["ensemble_candidates"] = _ensemble_candidates_chart(candidates)
    return charts


def _export_chart_images(
    charts: dict[str, go.Figure], charts_dir: Path, policy: str | bool
) -> tuple[dict[str, str], dict]:
    if policy is False or policy == "never":
        return {}, {"status": "disabled", "exported": [], "errors": {}}
    if policy == "auto" and importlib.util.find_spec("kaleido") is None:
        return {}, {
            "status": "unavailable",
            "exported": [],
            "errors": {"dependency": "Install smarttab[report-static] to export PNG charts."},
        }
    files: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name, figure in charts.items():
        output = charts_dir / f"{name}.png"
        try:
            figure.write_image(str(output), **_CHART_IMAGE_SIZE)
            files[name] = f"charts/{name}.png"
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            logger.warning("Static chart export failed for %s: %s", name, exc)
    if files and errors:
        status = "partial"
    elif files:
        status = "ok"
    else:
        status = "failed"
    return files, {"status": status, "exported": sorted(files), "errors": errors}


def _render_html(ctx: ReportContext, charts: dict[str, go.Figure], report_dict: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
    )
    template = env.get_template("report.html.j2")
    chart_html = {
        name: figure.to_html(full_html=False, include_plotlyjs=False)
        for name, figure in charts.items()
    }
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
        per_label_thresholds=ctx.per_label_thresholds,
        objective=ctx.objective,
        notes=ctx.notes,
        static_export=report_dict["static_chart_export"],
        library_version=__version__,
        generated_at=report_dict["generated_at"],
        multimodal_info=ctx.multimodal_info,
        data_quality_report=ctx.data_quality_report,
        cleaning_report=ctx.cleaning_report,
        uncertainty_info=ctx.uncertainty_info,
        modality_dropout_info=ctx.modality_dropout_info,
        evaluation_quality_report=ctx.evaluation_quality_report,
        evaluation_drift_report=ctx.evaluation_drift_report,
    )


def _build_report_dict(ctx: ReportContext, chart_files: dict[str, str], static_export: dict) -> dict:
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
        "per_label_thresholds": ctx.per_label_thresholds,
        "objective": ctx.objective,
        "timings": ctx.timings,
        "feature_importance": ctx.feature_importance.to_dict(orient="records"),
        "shap_importance": (
            ctx.shap_importance.to_dict(orient="records")
            if ctx.shap_importance is not None else None
        ),
        "ensemble_info": ctx.ensemble_info,
        "notes": ctx.notes,
        "dataset_profile": dataset_profile,
        "hardware_profile": dataclasses.asdict(ctx.hardware_profile),
        "resource_plan": dataclasses.asdict(ctx.resource_plan),
        "chart_files": chart_files,
        "static_chart_export": static_export,
        "multimodal_info": ctx.multimodal_info,
        "data_quality_report": ctx.data_quality_report,
        "cleaning_report": ctx.cleaning_report,
        "uncertainty_info": ctx.uncertainty_info,
        "modality_dropout_info": ctx.modality_dropout_info,
        "evaluation_quality_report": ctx.evaluation_quality_report,
        "evaluation_drift_report": ctx.evaluation_drift_report,
    }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _drift_chart(report: dict) -> go.Figure:
    rows = []
    for column, result in (report.get("raw_columns") or {}).items():
        rows.append((f"raw:{column}", float(result.get("score", 0.0))))
    for column, result in (report.get("transformed_features") or {}).items():
        rows.append((f"feature:{column}", float(result.get("score", 0.0))))
    rows = sorted(rows, key=lambda item: item[1], reverse=True)[:30][::-1]
    fig = go.Figure(go.Bar(x=[value for _, value in rows], y=[name for name, _ in rows], orientation="h"))
    fig.update_layout(
        title=f"Holdout Drift Diagnostics — {report.get('severity', 'unknown')}",
        template="plotly_dark", xaxis_title="drift score", xaxis_range=[0, 1],
        height=max(320, 24 * len(rows)), margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _data_quality_chart(report: dict) -> go.Figure:
    counts = report.get("severity_counts", {})
    labels = ["errors", "warnings", "information"]
    values = [int(counts.get("error", 0)), int(counts.get("warning", 0)), int(counts.get("info", 0))]
    fig = go.Figure(go.Bar(x=labels, y=values))
    fig.update_layout(
        title=f"Data Quality Findings — score {float(report.get('quality_score', 0.0)):.1f}/100",
        template="plotly_dark", yaxis_title="issue count", margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _missingness_chart(missing: dict[str, float]) -> go.Figure:
    ordered = sorted(missing.items(), key=lambda item: float(item[1]), reverse=True)[:30]
    labels = [str(name) for name, _ in ordered][::-1]
    values = [100.0 * float(value) for _, value in ordered][::-1]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h"))
    fig.update_layout(
        title="Raw Missingness by Column", template="plotly_dark", xaxis_title="missing (%)",
        height=max(320, 24 * len(labels)), margin=dict(l=10, r=10, t=50, b=10),
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



def _ensemble_diversity_chart(matrix: dict[str, dict[str, float]]) -> go.Figure:
    aliases = list(matrix)
    values = [[float(matrix[left].get(right, 0.0)) for right in aliases] for left in aliases]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=aliases,
            y=aliases,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="|corr|"),
            hovertemplate="%{y} vs %{x}<br>|corr|=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="OOF Prediction Correlation",
        template="plotly_dark",
        height=max(420, 42 * len(aliases)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _ensemble_candidates_chart(candidates: list[dict]) -> go.Figure:
    ordered = sorted(
        candidates,
        key=lambda item: float(item.get("oof_score") or float("-inf")),
        reverse=True,
    )
    aliases = [str(item.get("alias")) for item in ordered]
    scores = [float(item.get("oof_score") or 0.0) for item in ordered]
    retained = ["retained" if item.get("retained") else "discarded" for item in ordered]
    fig = go.Figure(
        go.Bar(
            x=aliases,
            y=scores,
            customdata=retained,
            hovertemplate="%{x}<br>OOF score=%{y:.5f}<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Ensemble Candidate OOF Scores",
        template="plotly_dark",
        xaxis_title="candidate",
        yaxis_title="OOF score",
        height=430,
        margin=dict(l=10, r=10, t=50, b=10),
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

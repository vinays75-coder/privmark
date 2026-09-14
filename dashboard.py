#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only PrivMark evidence dashboard for schema 1.0 artifacts.

Usage:
    uv run streamlit run dashboard.py

Example:
    Import ``make_summary_frame`` to flatten an artifact without loading models
    or importing the CLI. Missing measurements remain missing, never zero.

The CLI owns artifact validation and demo generation. This module never runs
models, recomputes statistical estimates, or derives an aggregate privacy score.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

EXIT_SUCCESS = 0
EXIT_ERROR = 2
ROOT = Path(__file__).resolve().parent
MINT = "#48E5C2"
VIOLET = "#B49AFF"
INK = "#0B1020"
TEXT = "#EDF2FF"
MUTED = "#AEBAD4"
CONDITIONS = ("baseline", "hardened")
COLORS = {"baseline": VIOLET, "hardened": MINT}
DEMO_LABEL = "ILLUSTRATIVE DEMO • NO MODEL WAS RUN"
MEASURED_LABEL = "MEASURED • LOCAL SYNTHETIC EXPERIMENT"
SCOPE_NOTE = (
    "Intervals reflect the sampled synthetic records, not a real-world privacy guarantee. "
    "Four fixed attack families are not comprehensive. Utility failure could mean model "
    "inability, not effective protection. No aggregate safety score is inferred."
)
RATE_FIELDS = ("rate", "low", "high", "n", "successes")
SUMMARY_COLUMNS = [
    "schema_version", "run_id", "mode", "report_label", "created_at", "model_id",
    "model_label", "status", "error", "condition",
    *[f"{metric}_{field}" for metric in ("leakage", "utility") for field in RATE_FIELDS],
    "latency_median_s", "truncated", "trial_count", "paired_difference",
    "paired_low", "paired_high", "paired_n", "paired_degenerate",
]
CSS = """
<style>
  .stApp { background: radial-gradient(ellipse at 85% 0%, #202447 0%, #0B1020 48%); }
  .block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }
  [data-testid="stSidebar"] { background: #10182B; border-right: 1px solid #29344E; }
  .pm-hero { padding: 2rem 2.2rem; border: 1px solid #354567; border-radius: 24px;
    background: linear-gradient(115deg, #152B3B, #1C2040 75%); margin-bottom: 1.3rem; }
  .pm-eyebrow { color: #65F1D1; font-size: .78rem; font-weight: 700;
    letter-spacing: .18em; text-transform: uppercase; }
  .pm-hero h1 { color: #EDF2FF; font-size: clamp(2.4rem, 5vw, 4rem);
    letter-spacing: -.045em; margin: .3rem 0; padding: 0; }
  .pm-hero p { color: #C6D2E8; max-width: 760px; font-size: 1.05rem; margin: .6rem 0 0; }
  [data-testid="stMetric"] { background: #151D33; border: 1px solid #33405B;
    border-radius: 18px; padding: 1.15rem; min-height: 125px; }
  [data-testid="stMetricLabel"] { color: #C6D2E8; }
  [data-testid="stMetricValue"] { color: #65F1D1; }
  button[data-baseweb="tab"] { font-weight: 600; }
  button:focus-visible, a:focus-visible { outline: 3px solid #65F1D1 !important;
    outline-offset: 3px !important; }
  [data-testid="stDownloadButton"] button { border-radius: 12px; }
</style>
"""


def _mapping(value: Any) -> dict[str, Any]:
    """Return a dictionary or an empty view for optional artifact sections."""
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    """Accept finite numeric observations without coercing missing data to zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _format(value: Any, *, percent: bool = False, digits: int = 2) -> str:
    """Format an observation with an explicit unavailable placeholder."""
    number = _number(value)
    if number is None:
        return "Not available"
    return f"{number:.1%}" if percent else f"{number:,.{digits}f}"


def _label(data: dict[str, Any]) -> str:
    """Keep the experiment mode explicit in every report and figure."""
    return DEMO_LABEL if data["mode"] == "demo" else MEASURED_LABEL


def _model_labels(data: dict[str, Any]) -> dict[str, str]:
    """Use fictional display names for every demo, including uploaded demos."""
    return {
        model["model_id"]: (
            f"Illustrative model {chr(65 + index) if index < 26 else index + 1}"
            if data["mode"] == "demo" else model["model_id"]
        )
        for index, model in enumerate(data.get("models", []))
    }


def make_summary_frame(
    data: dict[str, Any], model_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Flatten validated schema 1.0 summaries without estimating new metrics.

    Args:
        data: Validated artifact containing models and condition summaries.
        model_ids: IDs to retain; None selects all, an empty list selects none.

    Returns:
        Stable-column frame with one row per available condition. Models without
        conditions retain a status-only row. Null observations remain missing.
        Every row includes mode, report label, and original model ID for joins.
    """
    selected = None if model_ids is None else set(model_ids)
    labels = _model_labels(data)
    rows: list[dict[str, Any]] = []
    for model in data.get("models", []):
        if selected is not None and model["model_id"] not in selected:
            continue
        base = {key: data.get(key) for key in ("schema_version", "run_id", "mode", "created_at")}
        base.update(
            report_label=_label(data), model_id=model["model_id"],
            model_label=labels[model["model_id"]], status=model.get("status"),
            error=model.get("error"),
        )
        paired = _mapping(model.get("paired_difference"))
        for field in ("difference", "low", "high", "n", "degenerate"):
            base[f"paired_{field}"] = paired.get(field)
        conditions = _mapping(model.get("conditions"))
        available = [name for name in CONDITIONS if name in conditions]
        for name in available or [None]:
            row = {**base, "condition": name}
            summary = _mapping(_mapping(conditions.get(name)).get("summary"))
            for metric in ("leakage", "utility"):
                rate = _mapping(summary.get(metric))
                for field in RATE_FIELDS:
                    row[f"{metric}_{field}"] = _number(rate.get(field))
            for field in ("latency_median_s", "truncated", "trial_count"):
                row[field] = _number(summary.get(field))
            rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _figure(title: str, data: dict[str, Any]) -> go.Figure:
    """Create a consistently styled figure with its mode embedded in the title."""
    figure = go.Figure()
    figure.update_layout(
        title={"text": f"{html.escape(title)}<br><sup>{_label(data)}</sup>", "font": {"size": 19}},
        template="plotly_dark", paper_bgcolor=INK, plot_bgcolor=INK,
        font={"family": "Arial, sans-serif", "color": TEXT, "size": 12},
        colorway=[VIOLET, MINT, "#F4C87A"],
        margin={"l": 65, "r": 25, "t": 105, "b": 85},
        legend={"orientation": "h", "y": -0.23, "x": 0},
        height=460,
    )
    figure.update_xaxes(gridcolor="#26314B", zerolinecolor="#53637E", automargin=True)
    figure.update_yaxes(gridcolor="#26314B", zerolinecolor="#53637E", automargin=True)
    return figure


def _show(figure: go.Figure, key: str, figures: dict[str, go.Figure]) -> None:
    """Render and retain the same labeled figure for standalone HTML export."""
    figures[key] = figure
    # This keyword supports the entire declared Streamlit >=1.45,<2 range.
    st.plotly_chart(figure, use_container_width=True, key=key, config={"displaylogo": False})


def _error_bars(frame: pd.DataFrame, metric: str, scale: float = 1) -> dict[str, Any]:
    """Translate stored interval bounds to Plotly distances, preserving gaps."""
    upper, lower = [], []
    for _, row in frame.iterrows():
        rate, low, high = (_number(row.get(f"{metric}_{part}")) for part in ("rate", "low", "high"))
        valid = rate is not None and low is not None and high is not None and low <= rate <= high
        upper.append((high - rate) * scale if valid else None)
        lower.append((rate - low) * scale if valid else None)
    return {"type": "data", "symmetric": False, "array": upper, "arrayminus": lower}


def _rate_figure(frame: pd.DataFrame, metric: str, data: dict[str, Any]) -> go.Figure:
    """Plot only recorded rates with their stored Wilson intervals and counts."""
    title = "Any-attack record disclosure" if metric == "leakage" else "Public-field utility control"
    figure = _figure(f"{title} · 95% Wilson intervals", data)
    for condition in CONDITIONS:
        part = frame[frame["condition"] == condition]
        part = part[part[f"{metric}_rate"].notna()]
        if part.empty:
            continue
        figure.add_trace(go.Bar(
            name=condition.title(), x=[html.escape(str(v)) for v in part["model_label"]],
            y=part[f"{metric}_rate"], marker_color=COLORS[condition],
            marker_pattern_shape="/" if condition == "baseline" else "",
            error_y=_error_bars(part, metric),
            customdata=part[[f"{metric}_successes", f"{metric}_n"]].values,
            hovertemplate=("%{x}<br>Rate: %{y:.1%}<br>Successes: %{customdata[0]}"
                           " / %{customdata[1]}<extra>%{fullData.name}</extra>"),
        ))
    figure.update_layout(barmode="group")
    figure.update_yaxes(title="Fraction of synthetic records", tickformat=".0%", range=[0, 1.08])
    return figure


def _overview(
    data: dict[str, Any], frame: pd.DataFrame, figures: dict[str, go.Figure],
) -> None:
    """Render disclosure and utility without conflating either with safety."""
    st.subheader("Observed behavior, not a privacy rating")
    st.caption("Disclosure counts a record once if any fixed attack exposes its token. "
               "Utility separately checks retrieval of its public value.")
    if frame.empty:
        st.info("No completed condition summaries in this selection.")
        return
    left, right = st.columns(2)
    with left:
        _show(_rate_figure(frame, "leakage", data), "Record disclosure", figures)
    with right:
        _show(_rate_figure(frame, "utility", data), "Utility controls", figures)
    left, right = st.columns(2)
    for column, metric, title in (
        (left, "leakage", "Disclosure vs inference latency"),
        (right, "utility", "Utility vs inference latency"),
    ):
        with column:
            figure = _figure(title, data)
            for condition in CONDITIONS:
                part = frame[frame["condition"] == condition].dropna(
                    subset=["latency_median_s", f"{metric}_rate"]
                )
                if part.empty:
                    continue
                figure.add_trace(go.Scatter(
                    name=condition.title(), mode="markers", x=part["latency_median_s"],
                    y=part[f"{metric}_rate"], error_y=_error_bars(part, metric),
                    text=[html.escape(str(value)) for value in part["model_label"]],
                    marker={"color": COLORS[condition], "size": 14,
                            "symbol": "diamond" if condition == "baseline" else "circle"},
                    hovertemplate="%{text}<br>%{x:.3f} s<br>%{y:.1%}<extra>%{fullData.name}</extra>",
                ))
            figure.update_xaxes(title="Median observed latency (seconds)")
            figure.update_yaxes(title=f"{metric.title()} rate", tickformat=".0%", range=[0, 1.08])
            _show(figure, title, figures)
    st.caption("Latency includes prompt-length, hardware, and decoding effects; it does not isolate "
               "the cost of a privacy mechanism. Missing observations are omitted, not set to zero.")
    st.dataframe(frame[[
        "model_label", "condition", "leakage_successes", "leakage_n", "utility_successes",
        "utility_n", "trial_count", "truncated", "latency_median_s",
    ]], hide_index=True, use_container_width=True)


def _attacks(
    data: dict[str, Any], models: list[dict[str, Any]], conditions: list[str],
    figures: dict[str, go.Figure],
) -> None:
    """Show fixed-attack observations and the paired record-level comparison."""
    st.subheader("Four fixed attack families")
    st.caption("This is a fixed prompt suite, not adaptive red-teaming or comprehensive coverage. "
               "The public-field utility control is not an attack.")
    labels = _model_labels(data)
    entries: list[tuple[str, dict[str, Any]]] = []
    for model in models:
        if model.get("status") != "complete":
            continue
        for condition in conditions:
            value = _mapping(_mapping(model.get("conditions")).get(condition))
            if not value:
                continue
            name = f"{labels[model['model_id']]} · {condition}"
            summary = _mapping(value.get("summary"))
            entries.append((name, summary))
            truncated = _number(summary.get("truncated"))
            if truncated is not None and truncated > 0:
                st.warning(f"{name}: {int(truncated)} truncated generations. "
                           "Token limits can censor disclosure and utility observations.")
    attacks = sorted({
        attack for _, summary in entries
        for attack in _mapping(summary.get("by_attack")) if attack != "utility"
    })
    if entries and attacks:
        z, annotations, details = [], [], []
        for _, summary in entries:
            rates = [_mapping(_mapping(summary.get("by_attack")).get(name)) for name in attacks]
            values = [_number(rate.get("rate")) for rate in rates]
            z.append([value * 100 if value is not None else None for value in values])
            annotations.append([f"{value:.0%}" if value is not None else "N/A" for value in values])
            details.append([
                f"{_format(rate.get('successes'), digits=0)} / {_format(rate.get('n'), digits=0)}"
                f" records; Wilson CI {_format(rate.get('low'), percent=True)}"
                f"–{_format(rate.get('high'), percent=True)}" for rate in rates
            ])
        figure = _figure("Disclosure by fixed attack · annotated percentages", data)
        figure.add_trace(go.Heatmap(
            x=[html.escape(name) for name in attacks],
            y=[html.escape(name) for name, _ in entries], z=z, text=annotations,
            texttemplate="%{text}", textfont={"color": TEXT}, customdata=details,
            colorscale=[[0, "#172F41"], [0.5, "#514773"], [1, "#75445F"]],
            zmin=0, zmax=100, colorbar={"title": "Disclosure %"}, hoverongaps=False,
            hovertemplate="%{y}<br>%{x}: %{text}<br>%{customdata}<extra></extra>",
            xgap=4, ygap=4,
        ))
        figure.update_layout(height=max(430, 180 + len(entries) * 42))
        _show(figure, "Attack heatmap", figures)
    else:
        st.info("No attack-level measurements in this selection.")
    st.subheader("Paired hardening effect")
    st.caption("Hardened minus baseline record disclosure, in percentage points. "
               "Negative values favor hardening on this fixture only.")
    if len(conditions) != 2:
        st.info("Select Both conditions to inspect the paired comparison.")
        return
    figure = _figure("Paired disclosure delta · stored bootstrap interval", data)
    plotted = False
    for model in models:
        if model.get("status") != "complete":
            continue
        paired = _mapping(model.get("paired_difference"))
        delta = _number(paired.get("difference"))
        if delta is None:
            continue
        plotted = True
        name = labels[model["model_id"]]
        degenerate = paired.get("degenerate") is True
        row = pd.DataFrame([{
            "delta_rate": delta, "delta_low": paired.get("low"), "delta_high": paired.get("high"),
        }])
        figure.add_trace(go.Scatter(
            x=[delta * 100], y=[html.escape(name)], mode="markers", showlegend=False,
            marker={"size": 14, "color": VIOLET if degenerate else MINT,
                    "symbol": "diamond-open" if degenerate else "circle"},
            error_x=_error_bars(row, "delta", 100),
            text=[f"n={_format(paired.get('n'), digits=0)}; degenerate={degenerate}"],
            hovertemplate="%{y}<br>Delta: %{x:.1f} pp<br>%{text}<extra></extra>",
        ))
        if degenerate:
            st.warning(f"{name}: degenerate bootstrap interval. Identical observed paired "
                       "differences do not imply zero population uncertainty.")
    if plotted:
        figure.add_vline(x=0, line_dash="dash", line_color=MUTED)
        figure.update_xaxes(title="Hardened − baseline (percentage points)")
        _show(figure, "Paired difference", figures)
    else:
        st.info("No paired comparison was recorded.")


def _disclosures(models: list[dict[str, Any]], labels: dict[str, str]) -> None:
    """Present all disclosure dimensions as evidence, never a radar or score."""
    st.subheader("Scoped disclosure record")
    st.caption("Documented, partially measured, and unassessed are different evidence states. "
               "None is a certification of deployment safety or compliance.")
    if not models:
        st.info("Select a model to inspect its disclosure record.")
        return
    lookup = {model["model_id"]: model for model in models}
    selected = st.selectbox("Model for disclosure dimensions", list(lookup),
                            format_func=labels.get, key="disclosure_model")
    rows = lookup[selected].get("disclosures", [])
    if not rows:
        st.info("No disclosure dimensions were supplied for this model.")
        return
    fields = ["dimension", "status", "scope", "finding", "evidence"]
    st.dataframe(pd.DataFrame(rows).reindex(columns=fields), hide_index=True,
                 use_container_width=True)
    for index, row in enumerate(rows):
        with st.expander(f"Dimension {index + 1} · view scope and evidence"):
            # All artifact strings use text/table/JSON APIs, never HTML rendering.
            st.text(f"{row.get('dimension', 'Unspecified')} — {row.get('status', 'Not available')}")
            st.text(f"Scope: {row.get('scope', 'Not available')}")
            st.text(row.get("finding", "Not available"))
            st.text(f"Evidence: {row.get('evidence') or 'No evidence pointer supplied'}")


def _memorization(
    data: dict[str, Any], models: list[dict[str, Any]], figures: dict[str, go.Figure],
) -> None:
    """Visualize only supplied local fine-tuning membership measurements."""
    st.subheader("Controlled local fine-tuning")
    st.warning("Membership here means membership in this local fine-tuning split, NOT "
               "pretraining membership. These results cannot establish what a base model "
               "was trained on or its real-world privacy risk.")
    st.caption("Before/after are fine-tuning stages, independent of the sidebar's prompt conditions.")
    if not models:
        st.info("Select a model to inspect memorization evidence.")
        return
    labels = _model_labels(data)
    lookup = {model["model_id"]: model for model in models}
    selected = st.selectbox("Model for local memorization experiment", list(lookup),
                            format_func=labels.get, key="memorization_model")
    model = lookup[selected]
    experiment = _mapping(model.get("memorization"))
    metrics = _mapping(experiment.get("metrics"))
    if not metrics:
        st.info("No local fine-tuning metrics are available for this model.")
        st.text(f"Experiment status: {experiment.get('status', 'not evaluated')}")
        return
    if data["mode"] == "demo":
        st.warning(DEMO_LABEL + " · Any displayed memorization values are illustrative only.")
    warning = metrics.get("warning")
    if warning:
        st.text(warning)
    st.caption(f"Observed FPR resolution: {_format(metrics.get('fpr_resolution'), percent=True)}. "
               "TPR at 5% FPR is descriptive on this sample, not a held-out guarantee.")
    figure = _figure(f"{labels[selected]} · local membership ROC", data)
    stages = []
    for stage, color in (("before", VIOLET), ("after", MINT)):
        values = _mapping(metrics.get(stage))
        if not values:
            continue
        fpr, tpr = values.get("fpr", []), values.get("tpr", [])
        if isinstance(fpr, list) and isinstance(tpr, list) and fpr and len(fpr) == len(tpr):
            figure.add_trace(go.Scatter(
                x=fpr, y=tpr, name=stage.title(), mode="lines",
                line={"color": color, "width": 3, "dash": "dash" if stage == "before" else "solid"},
                hovertemplate="FPR %{x:.1%}<br>TPR %{y:.1%}<extra>%{fullData.name}</extra>",
            ))
        stages.append({
            "Stage": stage, "AUC": _format(values.get("auc"), digits=3),
            "95% AUC interval": f"{_format(values.get('low'), digits=3)} – "
                                f"{_format(values.get('high'), digits=3)}",
            "TPR at 5% FPR": _format(values.get("tpr_at_5pct_fpr"), percent=True),
        })
    left, right = st.columns(2)
    with left:
        if figure.data:
            figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance reference",
                                       line={"color": MUTED, "dash": "dot"}, hoverinfo="skip"))
            figure.update_xaxes(title="False positive rate", range=[0, 1], tickformat=".0%")
            figure.update_yaxes(title="True positive rate", range=[0, 1], tickformat=".0%")
            _show(figure, "Local membership ROC", figures)
        else:
            st.info("No ROC coordinates were supplied.")
    with right:
        st.markdown("#### Membership attack estimates")
        st.dataframe(pd.DataFrame(stages), hide_index=True, use_container_width=True)
        extraction_rows = []
        for stage in ("before", "after"):
            for group in ("member", "nonmember"):
                rate = _mapping(_mapping(metrics.get(stage)).get(f"{group}_extraction"))
                extraction_rows.append({
                    "Stage": stage, "Group": group,
                    "Extraction": _format(rate.get("rate"), percent=True),
                    "95% Wilson interval": f"{_format(rate.get('low'), percent=True)} – "
                                           f"{_format(rate.get('high'), percent=True)}",
                    "Successes / n": f"{_format(rate.get('successes'), digits=0)} / "
                                     f"{_format(rate.get('n'), digits=0)}",
                })
        st.markdown("#### Exact-canary extraction")
        st.dataframe(pd.DataFrame(extraction_rows), hide_index=True, use_container_width=True)
    records = experiment.get("records", [])
    points = [row for row in records if _number(row.get("nll_before")) is not None
              and _number(row.get("nll_after")) is not None
              and isinstance(row.get("member"), bool)]
    if points:
        figure = _figure(f"{labels[selected]} · per-record target-token NLL change", data)
        for member, color in ((True, MINT), (False, VIOLET)):
            group = [row for row in points if row["member"] is member]
            if not group:
                continue
            figure.add_trace(go.Bar(
                name="Local member" if member else "Local nonmember",
                x=[html.escape(str(row["record_id"])) for row in group],
                y=[row["nll_after"] - row["nll_before"] for row in group],
                marker_color=color, marker_pattern_shape="" if member else "/",
                customdata=[[row["nll_before"], row["nll_after"]] for row in group],
                hovertemplate=("Record %{x}<br>Δ NLL %{y:.3f}<br>Before %{customdata[0]:.3f}"
                               "<br>After %{customdata[1]:.3f}<extra>%{fullData.name}</extra>"),
            ))
        figure.update_xaxes(title="Synthetic record ID", type="category")
        figure.update_yaxes(title="Mean target-token NLL: after − before")
        _show(figure, "Per-record NLL changes", figures)
        st.caption("Negative change means lower target-token loss after local training. "
                   "It is not itself proof of extraction or pretraining membership.")
    else:
        st.info("No finite per-record before/after NLL observations were supplied.")
    with st.expander("Local fine-tuning configuration and record evidence"):
        st.json(experiment, expanded=False)


def _json_bytes(value: Any) -> bytes:
    """Serialize reports as strict UTF-8 JSON without executable markup."""
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    """Keep missing cells blank and neutralize spreadsheet formula injection."""
    safe = frame.copy()
    for column in safe.columns:
        if safe[column].dtype == object:
            safe[column] = safe[column].map(
                lambda value: "'" + value
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                else value
            )
    return safe.to_csv(index=False, na_rep="").encode("utf-8-sig")


def _evidence(
    data: dict[str, Any], models: list[dict[str, Any]], conditions: list[str],
    frame: pd.DataFrame, figures: dict[str, go.Figure],
) -> None:
    """Export original schema artifacts and explicitly scoped derived reports."""
    st.subheader("Evidence you can inspect and carry forward")
    st.caption("The artifact download preserves the full validated schema, including unselected "
               "models and conditions. Summary CSV and trial downloads follow the active filters. "
               "Exports contain synthetic secrets and raw prompts; review before sharing.")
    prefix = "privmark_DEMO" if data["mode"] == "demo" else "privmark_MEASURED"
    left, right = st.columns(2)
    with left:
        st.download_button(f"Download full artifact JSON · {data['mode'].upper()}",
                           _json_bytes(data), file_name=f"{prefix}_artifact.json",
                           mime="application/json")
    with right:
        st.download_button(f"Download filtered summary CSV · {data['mode'].upper()}",
                           _csv_bytes(frame), file_name=f"{prefix}_summary.csv", mime="text/csv")
    st.caption("CSV includes mode and report_label on every row. An empty selection produces "
               "headers only; its filename still identifies the mode. CSV formula-like text is escaped.")
    if figures:
        selected_figure = st.selectbox("Figure for standalone HTML report", list(figures))
        document = figures[selected_figure].to_html(full_html=True, include_plotlyjs=True)
        title = html.escape(f"PrivMark | {_label(data)} | {selected_figure}")
        document = document.replace("<head>", f"<head><title>{title}</title>", 1)
        document = document.replace(
            "<body>",
            '<body style="background:#0B1020;color:#EDF2FF;font-family:Arial,sans-serif">'
            f'<header style="padding:24px"><h1>{title}</h1>'
            f"<p>{html.escape(SCOPE_NOTE)}</p></header>", 1,
        )
        st.download_button(f"Download self-contained figure HTML · {data['mode'].upper()}",
                           document.encode("utf-8"), file_name=f"{prefix}_figure.html",
                           mime="text/html")
        st.caption("Plotly is embedded in the HTML; viewing the figure needs no CDN connection.")
    else:
        st.info("No figure is available for the current selection.")
    st.markdown("#### Raw trials")
    if models:
        labels = _model_labels(data)
        lookup = {model["model_id"]: model for model in models}
        selected = st.selectbox("Model for raw trial evidence", list(lookup),
                                format_func=labels.get, key="trial_model")
        candidates = []
        for condition in conditions:
            value = _mapping(_mapping(lookup[selected].get("conditions")).get(condition))
            candidates.extend(value.get("trials", []))
        available_attacks = sorted({row["attack"] for row in candidates})
        attacks = st.multiselect("Attack families or utility control in raw trials", available_attacks,
                                 default=available_attacks, key=f"trial_attacks_{selected}")
        trials = [row for row in candidates if row["attack"] in attacks]
        if trials:
            table = pd.DataFrame(trials)
            if "messages" in table:
                table["messages"] = table["messages"].map(lambda value: json.dumps(value, ensure_ascii=False))
            st.dataframe(table, hide_index=True, use_container_width=True)
            chosen = st.selectbox(
                "Trial to inspect as JSON", list(range(len(trials))),
                format_func=lambda index: (
                    f"{index + 1} · {trials[index]['record_id']} · "
                    f"{trials[index]['condition']} · {trials[index]['attack']}"
                ),
            )
            st.json(trials[chosen], expanded=False)
        else:
            st.info("No raw trials match these filters.")
        report = {
            "report_type": "filtered_trial_evidence", "source_schema_version": data["schema_version"],
            "run_id": data["run_id"], "mode": data["mode"], "report_label": _label(data),
            "model_id": selected, "conditions": conditions, "attacks": attacks, "trials": trials,
        }
        st.download_button(f"Download selected trials JSON · {data['mode'].upper()}",
                           _json_bytes(report), file_name=f"{prefix}_selected_trials.json",
                           mime="application/json")
        st.caption("The filtered trial report is not a reloadable schema artifact; use the full "
                   "artifact download for round trips.")
    else:
        st.info("Select at least one model to inspect raw trials.")
    with st.expander("Run provenance, configuration, and model metadata"):
        st.json({
            "schema_version": data["schema_version"], "run_id": data["run_id"],
            "mode": data["mode"], "created_at": data.get("created_at"),
            "provenance": data.get("provenance", {}), "config": data.get("config", {}),
            "selected_model_metadata": [
                {"model_id": model["model_id"], "status": model.get("status"),
                 "metadata": model.get("metadata", {}), "error": model.get("error")}
                for model in models
            ],
        }, expanded=False)


def main() -> int:
    """Load one explicit source, validate it, and render the read-only dashboard.

    Returns:
        Zero after rendering, or two if the CLI interface/source cannot be read.
        Streamlit stops the current render on malformed or incomplete input.
    """
    st.set_page_config(page_title="PrivMark | Privacy evidence", page_icon="◈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<section class="pm-hero"><div class="pm-eyebrow">PrivMark / Evidence studio</div>'
        '<h1>Privacy claims need evidence.</h1>'
        '<p>Explore synthetic disclosure, public-field utility, and scoped local experiments. '
        'See what was observed—and what remains unassessed.</p></section>',
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.title("◈ PrivMark")
        st.caption("Read-only experiment explorer")
        try:
            files = sorted(
                (path for path in (ROOT / "results").glob("*.json") if path.is_file()),
                key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True,
            )
        except OSError as exc:
            st.error("Cannot enumerate local results. No demo was substituted.")
            st.text(str(exc))
            st.stop()
            return EXIT_ERROR
        source = st.radio("Artifact source", ["Local results", "Upload JSON", "Illustrative demo"],
                          index=0 if files else 2)
        with st.expander("Launch outside the browser"):
            st.caption("Example commands only. This dashboard does not execute models.")
            st.code("uv run python privmark.py run --models MODEL_ID_1 MODEL_ID_2", language="bash")
            st.code("uv run streamlit run dashboard.py", language="bash")
        path, uploaded = None, None
        if source == "Local results":
            if not files:
                st.info("No results/*.json files found. Choose Upload JSON or Illustrative demo.")
                st.stop()
                return EXIT_SUCCESS
            path = st.selectbox("Local artifact (newest modification first)", files,
                                format_func=lambda value: value.name)
        elif source == "Upload JSON":
            uploaded = st.file_uploader("Upload a schema 1.0 PrivMark artifact", type=["json"])
            st.caption("Uploads are schema-validated. A local checksum sidecar is not uploaded "
                       "or verified by this option.")
            if uploaded is None:
                st.info("Choose a JSON artifact to continue. No demo is substituted.")
                st.stop()
                return EXIT_SUCCESS
    try:
        # Lazy CLI imports keep importing make_summary_frame independent of the
        # parent entry point and any optional model execution libraries.
        if source == "Illustrative demo":
            from privmark import make_demo

            data = make_demo(seed=42, count=12)
        elif source == "Local results":
            from privmark import load_artifact

            assert path is not None
            data = load_artifact(path)
        else:
            from privmark import validate_artifact

            assert uploaded is not None
            data = validate_artifact(json.loads(uploaded.getvalue().decode("utf-8-sig")))
    except Exception as exc:
        st.error("The selected artifact could not be loaded or validated. "
                 "No illustrative data was substituted.")
        st.text(f"{type(exc).__name__}: {exc}")
        st.stop()
        return EXIT_ERROR
    if data["mode"] == "demo":
        st.warning(DEMO_LABEL, icon="⚠️")
        st.caption("All model identities and outcomes below are fictional illustrations, "
                   "not measurements or model comparisons.")
    else:
        st.info(MEASURED_LABEL, icon="ℹ️")
    st.text(f"Run: {data['run_id']}  |  Created: {data.get('created_at', 'Not available')}  "
            f"|  Schema: {data['schema_version']}")
    if path is not None:
        st.caption("Local loading validates the schema and checks the checksum when a sidecar "
                   "exists. A checksum alone is not proof of origin or an audit.")
    labels = _model_labels(data)
    with st.sidebar:
        st.divider()
        selected_ids = st.multiselect("Models to include", list(labels), default=list(labels),
                                      format_func=labels.get)
        selected_condition = st.selectbox("Prompt conditions", ["Both", "baseline", "hardened"])
        st.caption("Baseline and hardened refer to prompt-level instructions, not access control.")
        st.divider()
        st.caption(_label(data))
    conditions = list(CONDITIONS) if selected_condition == "Both" else [selected_condition]
    models = [model for model in data.get("models", []) if model["model_id"] in selected_ids]
    completed = [model for model in models if model.get("status") == "complete"]
    observed_trials = sum(
        len(_mapping(_mapping(model.get("conditions")).get(condition)).get("trials", []))
        for model in models for condition in conditions
    )
    disclosures_known = bool(models) and all(model.get("disclosures") for model in models)
    unassessed = sum(
        str(row.get("status", "")).casefold() in {"not evaluated", "unassessed", "illustrative only"}
        for model in models for row in model.get("disclosures", [])
    )
    cards = st.columns(4)
    cards[0].metric("Evaluated models", 0 if data["mode"] == "demo" else len(completed))
    cards[1].metric("Synthetic records", len(data.get("records", [])))
    cards[2].metric("Illustrative trials" if data["mode"] == "demo" else "Observed trials", observed_trials)
    cards[3].metric("Unassessed dimensions", unassessed if disclosures_known else "—")
    st.caption("Model and trial counts follow the filters; records describe the whole fixture. "
               "Unassessed counts model–dimension entries, not a safety score; missing disclosure "
               "tables yield —. Demo models are not evaluated models.")
    st.caption(SCOPE_NOTE)
    if not models:
        st.info("No models selected. Choose models in the sidebar; full artifact export remains available.")
    elif not completed:
        st.warning("No selected model completed. Status and raw evidence remain available; "
                   "missing measurements are not interpreted as zero disclosure.")
    for model in models:
        if model.get("status") == "failed":
            st.error(f"{labels[model['model_id']]}: evaluation failed.")
            st.text(model.get("error") or "No error detail supplied.")
    summary = make_summary_frame(data, selected_ids)
    # Retain status-only failures in CSV even when one condition is selected.
    summary = summary[summary["condition"].isin(conditions) | summary["condition"].isna()]
    chart_frame = summary[summary["status"].eq("complete") & summary["condition"].isin(conditions)]
    figures: dict[str, go.Figure] = {}
    tabs = st.tabs(["Overview", "Attack analysis", "Disclosure cards", "Memorization", "Evidence & export"])
    with tabs[0]:
        _overview(data, chart_frame, figures)
    with tabs[1]:
        _attacks(data, models, conditions, figures)
    with tabs[2]:
        _disclosures(models, labels)
    with tabs[3]:
        _memorization(data, models, figures)
    with tabs[4]:
        _evidence(data, models, conditions, summary, figures)
    st.divider()
    st.caption("PrivMark · Synthetic evidence, explicit scope, no inferred privacy score.")
    return EXIT_SUCCESS


if __name__ == "__main__":
    # Streamlit owns the script thread; SystemExit interrupts its completion event.
    main()
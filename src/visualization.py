"""
visualization.py
----------------
Reusable Plotly chart builders for the dashboard.

All functions return a plotly.graph_objects.Figure object so they can
be rendered with st.plotly_chart(..., use_container_width=True).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Design tokens (match the CSS colour palette) ──────────────────────────
BG       = "#0B1020"
CARD_BG  = "#161F32"
BORDER   = "#1E293B"
TEXT     = "#F8FAFC"
MUTED    = "#94A3B8"
ACCENT   = "#6366F1"
CYAN     = "#06B6D4"
SUCCESS  = "#22C55E"
WARNING  = "#F59E0B"
DANGER   = "#EF4444"
COLORS   = [ACCENT, CYAN, SUCCESS, WARNING, DANGER, "#A78BFA", "#34D399"]


def _base_layout(**kwargs):
    """Return a common layout dict applied to every chart."""
    return dict(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="'Segoe UI', system-ui, sans-serif",
                  color=TEXT, size=12),
        legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, borderwidth=1),
        margin=dict(l=40, r=20, t=50, b=40),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Confusion Matrix Heatmap
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: list, model_name: str = "") -> go.Figure:
    """
    Parameters
    ----------
    cm : 2×2 list  [[TN, FP], [FN, TP]]
    """
    labels = ["Bad (0)", "Good (1)"]
    z = np.array(cm)
    text = [[str(v) for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[[0, CARD_BG], [0.5, "#312E81"], [1, ACCENT]],
        text=text, texttemplate="%{text}",
        textfont=dict(size=18, color=TEXT),
        showscale=False,
    ))
    fig.update_layout(
        **_base_layout(title=f"Confusion Matrix — {model_name}"),
        xaxis=dict(title="Predicted", color=MUTED),
        yaxis=dict(title="Actual", color=MUTED, autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Model Performance Bar Chart
# ---------------------------------------------------------------------------
def plot_model_performance(results: dict) -> go.Figure:
    """
    Parameters
    ----------
    results : {model_name: {"metrics": {accuracy, precision, recall, f1}}}
    """
    metrics   = ["accuracy", "precision", "recall", "f1"]
    labels    = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig       = go.Figure()

    for i, (name, res) in enumerate(results.items()):
        m = res["metrics"]
        fig.add_trace(go.Bar(
            name=name,
            x=labels,
            y=[m["accuracy"], m["precision"], m["recall"], m["f1"]],
            marker_color=COLORS[i % len(COLORS)],
            text=[f"{v:.3f}" for v in [m["accuracy"], m["precision"],
                                        m["recall"], m["f1"]]],
            textposition="outside",
        ))

    fig.update_layout(
        **_base_layout(title="Model Performance Comparison",
                       barmode="group"),
        yaxis=dict(title="Score", range=[0, 1.1], color=MUTED,
                   gridcolor=BORDER),
        xaxis=dict(color=MUTED),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Fairness Metric Bar Chart (before vs after mitigation)
# ---------------------------------------------------------------------------
def plot_fairness_comparison(fairness_results: dict,
                             metric_key: str = "demographic_parity_difference",
                             title: str = "Fairness Comparison") -> go.Figure:
    """
    Parameters
    ----------
    fairness_results : {model_name: run_all_metrics(...) output}
    metric_key       : which sub-key to plot from each metric dict
    """
    # Map top-level key → sub-dict key
    key_map = {
        "demographic_parity_difference":  ("demographic_parity",  "demographic_parity_difference"),
        "equalized_odds_difference":      ("equalized_odds",       "equalized_odds_difference"),
        "equal_opportunity_difference":   ("equal_opportunity",    "equal_opportunity_difference"),
        "predictive_parity_difference":   ("predictive_parity",    "predictive_parity_difference"),
    }

    top_key, sub_key = key_map.get(metric_key,
                                   ("demographic_parity",
                                    "demographic_parity_difference"))

    names  = []
    values = []
    colors_bar = []
    for name, res in fairness_results.items():
        v = res[top_key][sub_key]
        if v is not None:
            names.append(name)
            values.append(v)
            colors_bar.append(SUCCESS if v < 0.10 else
                              WARNING if v < 0.20 else DANGER)

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors_bar,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    # Threshold line at 0.10
    fig.add_hline(y=0.10, line_dash="dash", line_color=WARNING,
                  annotation_text="Fairness threshold (0.10)",
                  annotation_font_color=WARNING)

    fig.update_layout(
        **_base_layout(title=title),
        yaxis=dict(title="Difference (lower is fairer)", color=MUTED,
                   gridcolor=BORDER),
        xaxis=dict(color=MUTED),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Fairness–Accuracy Tradeoff Scatter Plot
# ---------------------------------------------------------------------------
def plot_fairness_accuracy_tradeoff(perf_results: dict,
                                    fairness_results: dict,
                                    fairness_key: str = "demographic_parity_difference"
                                    ) -> go.Figure:
    """
    X-axis: accuracy
    Y-axis: chosen fairness difference (lower is better)
    Each model is a labelled point.
    """
    key_map = {
        "demographic_parity_difference":  ("demographic_parity",  "demographic_parity_difference"),
        "equalized_odds_difference":      ("equalized_odds",       "equalized_odds_difference"),
        "equal_opportunity_difference":   ("equal_opportunity",    "equal_opportunity_difference"),
        "predictive_parity_difference":   ("predictive_parity",    "predictive_parity_difference"),
    }
    top_key, sub_key = key_map.get(fairness_key,
                                   ("demographic_parity",
                                    "demographic_parity_difference"))

    names, accs, diffs = [], [], []
    for name in perf_results:
        if name in fairness_results:
            acc  = perf_results[name]["metrics"]["accuracy"]
            diff = fairness_results[name][top_key][sub_key]
            if diff is not None:
                names.append(name)
                accs.append(acc)
                diffs.append(diff)

    fig = go.Figure()
    for i, (n, a, d) in enumerate(zip(names, accs, diffs)):
        fig.add_trace(go.Scatter(
            x=[a], y=[d],
            mode="markers+text",
            name=n,
            text=[n],
            textposition="top center",
            marker=dict(size=16, color=COLORS[i % len(COLORS)],
                        line=dict(width=2, color=TEXT)),
        ))

    fig.add_hline(y=0.10, line_dash="dash", line_color=WARNING,
                  annotation_text="Fairness threshold",
                  annotation_font_color=WARNING)

    fig.update_layout(
        **_base_layout(title="Fairness vs. Accuracy Tradeoff",
                       showlegend=True),
        xaxis=dict(title="Accuracy (higher is better)", color=MUTED,
                   gridcolor=BORDER),
        yaxis=dict(title="Fairness Difference (lower is fairer)",
                   color=MUTED, gridcolor=BORDER),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Group-wise Selection Rate Bar Chart
# ---------------------------------------------------------------------------
def plot_group_selection_rates(fairness_result: dict,
                               sensitive_labels=("Young (≤25)", "Older (>25)")
                               ) -> go.Figure:
    """Plot selection (approval) rates per demographic group."""
    dp = fairness_result["demographic_parity"]
    rates = [dp["selection_rate_group_0"], dp["selection_rate_group_1"]]

    fig = go.Figure(go.Bar(
        x=list(sensitive_labels),
        y=rates,
        marker_color=[CYAN, ACCENT],
        text=[f"{r:.1%}" for r in rates],
        textposition="outside",
    ))
    fig.update_layout(
        **_base_layout(title="Selection Rate by Demographic Group"),
        yaxis=dict(title="Selection Rate", range=[0, 1.1], color=MUTED,
                   gridcolor=BORDER),
        xaxis=dict(color=MUTED),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Radar Chart (all 4 performance metrics for N models)
# ---------------------------------------------------------------------------
def plot_radar(results: dict) -> go.Figure:
    """
    Multi-model radar chart comparing Accuracy, Precision, Recall, F1.
    """
    categories = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig = go.Figure()

    for i, (name, res) in enumerate(results.items()):
        m = res["metrics"]
        vals = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
        vals += [vals[0]]  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
            line=dict(color=COLORS[i % len(COLORS)]),
            fillcolor=COLORS[i % len(COLORS)].replace(")", ", 0.15)").replace("rgb", "rgba") if COLORS[i % len(COLORS)].startswith("rgb") else COLORS[i % len(COLORS)] + "26",
        ))

    fig.update_layout(
        **_base_layout(title="Performance Radar"),
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, range=[0, 1],
                            color=MUTED, gridcolor=BORDER),
            angularaxis=dict(color=MUTED, gridcolor=BORDER),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Class distribution pie chart
# ---------------------------------------------------------------------------
def plot_class_distribution(value_counts: dict, title: str = "Class Distribution"
                             ) -> go.Figure:
    labels = ["Good Credit (1)", "Bad Credit (0)"]
    vals   = [value_counts.get(1, 0), value_counts.get(0, 0)]

    fig = go.Figure(go.Pie(
        labels=labels, values=vals,
        marker=dict(colors=[SUCCESS, DANGER],
                    line=dict(color=BG, width=2)),
        hole=0.4,
        textinfo="label+percent",
        textfont=dict(color=TEXT),
    ))
    fig.update_layout(**_base_layout(title=title))
    return fig


# ---------------------------------------------------------------------------
# 8. Feature distribution histogram
# ---------------------------------------------------------------------------
def plot_feature_distribution(series: pd.Series, feature_name: str = "",
                               sensitive: pd.Series = None) -> go.Figure:
    fig = go.Figure()
    if sensitive is not None:
        for val, label, color in [(0, "Young (≤25)", CYAN),
                                   (1, "Older (>25)", ACCENT)]:
            subset = series[sensitive == val]
            fig.add_trace(go.Histogram(
                x=subset, name=label,
                marker_color=color, opacity=0.7,
                autobinx=True,
            ))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(
            x=series, marker_color=ACCENT, opacity=0.8))

    fig.update_layout(
        **_base_layout(title=f"Distribution: {feature_name}"),
        xaxis=dict(title=feature_name, color=MUTED, gridcolor=BORDER),
        yaxis=dict(title="Count", color=MUTED, gridcolor=BORDER),
    )
    return fig


# ---------------------------------------------------------------------------
# 9. TPR / FPR grouped bar (Equalized Odds visualisation)
# ---------------------------------------------------------------------------
def plot_equalized_odds_bars(fairness_result: dict,
                              sensitive_labels=("Young (≤25)", "Older (>25)")
                              ) -> go.Figure:
    eo = fairness_result["equalized_odds"]
    tprs = [eo["TPR_group_0"], eo["TPR_group_1"]]
    fprs = [eo["FPR_group_0"], eo["FPR_group_1"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="True Positive Rate",
                         x=list(sensitive_labels), y=tprs,
                         marker_color=SUCCESS,
                         text=[f"{v:.3f}" for v in tprs],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="False Positive Rate",
                         x=list(sensitive_labels), y=fprs,
                         marker_color=DANGER,
                         text=[f"{v:.3f}" for v in fprs],
                         textposition="outside"))

    fig.update_layout(
        **_base_layout(title="Equalized Odds — TPR & FPR by Group",
                       barmode="group"),
        yaxis=dict(title="Rate", range=[0, 1.1], color=MUTED,
                   gridcolor=BORDER),
        xaxis=dict(color=MUTED),
    )
    return fig

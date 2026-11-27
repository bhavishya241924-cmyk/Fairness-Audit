"""
5_Model_Comparison.py
----------------------
Page: Model Comparison

Compares all baseline models + all mitigated models side-by-side:
  - Performance table
  - Radar chart
  - Fairness metric comparison bar charts
  - Fairness-accuracy tradeoff scatter
  - User can select any model as "best" under a chosen criterion
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fairness_metrics import run_all_metrics
from src.visualization import (
    plot_model_performance, plot_radar,
    plot_fairness_comparison, plot_fairness_accuracy_tradeoff,
)

# ── Page config & CSS ──────────────────────────────────────────────────────
st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class='page-header'>
  <h1>📊 Model Comparison</h1>
  <p>Compare accuracy, fairness, and performance tradeoffs across all models</p>
</div>
""", unsafe_allow_html=True)

# ── Session defaults ───────────────────────────────────────────────────────
for k, v in [("models", {}), ("fairness", {}), ("mitigation", {}),
             ("mit_fairness", {}), ("preprocessed", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Collect all models + fairness ─────────────────────────────────────────
prep = st.session_state["preprocessed"]

all_perf    = {}   # {name: {"metrics": {...}}}
all_fairness = {}  # {name: run_all_metrics(...)}

# Baseline models
for name, res in st.session_state["models"].items():
    all_perf[name] = {"metrics": res["metrics"]}
    if name in st.session_state["fairness"]:
        all_fairness[name] = st.session_state["fairness"][name]
    elif prep is not None:
        y_pred = res["metrics"]["y_pred"]
        all_fairness[name] = run_all_metrics(
            prep["y_test"], y_pred, prep["A_test"].values, name)

# Mitigated models
mit_names = {
    "Pre-processing": "Pre (Reweighing)",
    "In-processing":  "In (ExpGrad)",
    "Post-processing":"Post (ThreshOpt)",
}
for tech, label in mit_names.items():
    if tech in st.session_state["mitigation"] and st.session_state["mitigation"][tech]:
        m_res = st.session_state["mitigation"][tech]
        all_perf[label]     = {"metrics": m_res["metrics"]}
        all_fairness[label] = st.session_state["mit_fairness"].get(tech, {})

if not all_perf:
    st.info("⚠️ No models found. Train models in **Model Training** first.")
    st.stop()

# ── 1. Performance summary table ───────────────────────────────────────────
st.markdown("### 1️⃣ Performance Comparison")
perf_rows = []
for name, res in all_perf.items():
    m = res["metrics"]
    perf_rows.append({
        "Model":     name,
        "Accuracy":  m["accuracy"],
        "Precision": m["precision"],
        "Recall":    m["recall"],
        "F1-Score":  m["f1"],
    })
perf_df = pd.DataFrame(perf_rows)
st.dataframe(
    perf_df.style.highlight_max(
        subset=["Accuracy","Precision","Recall","F1-Score"],
        color="#1a2e1a",
    ).format("{:.4f}", subset=["Accuracy","Precision","Recall","F1-Score"]),
    use_container_width=True,
)

# Performance bar + radar side by side
c1, c2 = st.columns(2)
with c1:
    fig_bar = plot_model_performance(all_perf)
    st.plotly_chart(fig_bar, use_container_width=True)
with c2:
    fig_radar = plot_radar(all_perf)
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── 2. Fairness comparison table ───────────────────────────────────────────
st.markdown("### 2️⃣ Fairness Comparison")

if all_fairness:
    fair_rows = []
    for name, fr in all_fairness.items():
        if not fr:
            continue
        fair_rows.append({
            "Model":           name,
            "Dem. Parity Diff":  fr.get("demographic_parity",  {}).get("demographic_parity_difference"),
            "Eq. Odds Diff":     fr.get("equalized_odds",       {}).get("equalized_odds_difference"),
            "Eq. Opp. Diff":     fr.get("equal_opportunity",    {}).get("equal_opportunity_difference"),
            "Pred. Parity Diff": fr.get("predictive_parity",    {}).get("predictive_parity_difference"),
        })
    fair_df = pd.DataFrame(fair_rows)
    fmt_cols = ["Dem. Parity Diff","Eq. Odds Diff","Eq. Opp. Diff","Pred. Parity Diff"]
    st.dataframe(
        fair_df.style.format("{:.4f}", subset=fmt_cols, na_rep="N/A")
                     .background_gradient(subset=fmt_cols, cmap="RdYlGn_r"),
        use_container_width=True,
    )

    metric_sel = st.selectbox(
        "Select fairness metric to chart",
        ["demographic_parity_difference", "equalized_odds_difference",
         "equal_opportunity_difference",  "predictive_parity_difference"],
        format_func=lambda x: x.replace("_"," ").title(),
    )
    fig_fair_cmp = plot_fairness_comparison(
        all_fairness, metric_sel,
        title=metric_sel.replace("_"," ").title() + " — All Models",
    )
    st.plotly_chart(fig_fair_cmp, use_container_width=True)

st.markdown("---")

# ── 3. Fairness-Accuracy tradeoff ──────────────────────────────────────────
st.markdown("### 3️⃣ Fairness-Accuracy Tradeoff")
st.caption("Bottom-right quadrant (high accuracy, low fairness difference) is ideal.")

if all_fairness:
    tradeoff_metric = st.selectbox(
        "Fairness metric for tradeoff plot",
        ["demographic_parity_difference", "equalized_odds_difference",
         "equal_opportunity_difference",  "predictive_parity_difference"],
        key="tradeoff_sel",
        format_func=lambda x: x.replace("_"," ").title(),
    )
    fig_tradeoff = plot_fairness_accuracy_tradeoff(
        all_perf, all_fairness, tradeoff_metric)
    st.plotly_chart(fig_tradeoff, use_container_width=True)

    st.markdown("""
    <div class='aft-card'>
      <div class='aft-card-title'>🎓 How to Read This Chart</div>
      <p style='color:#94A3B8;font-size:0.85rem;margin:0;'>
        • <b>X-axis (right)</b> = higher accuracy = better performance<br>
        • <b>Y-axis (down)</b> = lower fairness difference = fairer model<br>
        • The dashed line shows the 0.10 fairness threshold<br>
        • <b>No single model is best at everything</b> — this tradeoff
          is a fundamental tension in fair machine learning.
        A model that maximises accuracy often does so at the expense of fairness,
        and vice versa. Your choice should depend on the deployment context.
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── 4. Best model selector ─────────────────────────────────────────────────
st.markdown("### 4️⃣ Select Best Model by Criterion")

CRITERIA = {
    "Highest Accuracy":          ("Accuracy",        perf_df,    "Accuracy",   False),
    "Highest F1-Score":          ("F1-Score",         perf_df,    "F1-Score",   False),
    "Lowest Dem. Parity Diff":   ("Dem. Parity Diff", fair_df if all_fairness else perf_df, "Dem. Parity Diff",  True),
    "Lowest Eq. Odds Diff":      ("Eq. Odds Diff",    fair_df if all_fairness else perf_df, "Eq. Odds Diff",     True),
    "Lowest Equal Opp. Diff":    ("Eq. Opp. Diff",    fair_df if all_fairness else perf_df, "Eq. Opp. Diff",     True),
}

criterion = st.selectbox("Choose criterion", list(CRITERIA.keys()))
_, df_ref, col_ref, ascending_ref = CRITERIA[criterion]

if col_ref in df_ref.columns and not df_ref[col_ref].isnull().all():
    best_row = df_ref.loc[df_ref[col_ref].idxmin() if ascending_ref
                          else df_ref[col_ref].idxmax()]
    best_name = best_row["Model"]
    best_val  = best_row[col_ref]
    st.success(
        f"🏆 **Best model by '{criterion}':** `{best_name}` "
        f"({col_ref}: {best_val:.4f})"
    )
    st.markdown("""
    > ⚠️ This is based solely on the selected criterion.
    > No single model is universally best across all metrics.
    > Choose based on your deployment requirements and ethical context.
    """)
else:
    st.info(f"Run the Fairness Audit to unlock '{criterion}' ranking.")

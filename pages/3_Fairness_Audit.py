"""
3_Fairness_Audit.py
--------------------
Page: Fairness Audit

Runs all four fairness metrics on trained models and displays:
  - Per-metric KPI cards with pass/fail badges
  - Group-wise selection rates, TPR, FPR, precision
  - Interactive visualisations for each metric
  - Expandable formula + explanation cards
  - Sensitive attribute selector (age / sex)
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fairness_metrics import run_all_metrics, metrics_summary_df
from src.visualization import (
    plot_group_selection_rates, plot_equalized_odds_bars,
    plot_fairness_comparison,
)

# ── Page config & CSS ──────────────────────────────────────────────────────
st.set_page_config(page_title="Fairness Audit", page_icon="⚖️", layout="wide")
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class='page-header'>
  <h1>⚖️ Fairness Audit</h1>
  <p>Measure demographic bias using four established fairness metrics</p>
</div>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────
for k, v in [("models", {}), ("preprocessed", None), ("fairness", {}),
             ("audit_run", False), ("activity_log", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Guards ─────────────────────────────────────────────────────────────────
if not st.session_state["models"]:
    st.warning("⚠️ No trained models found. Go to **Model Training** first.")
    st.page_link("pages/2_Model_Training.py", label="Go to Model Training")
    st.stop()

if st.session_state["preprocessed"] is None:
    st.warning("⚠️ Preprocessing data not found. Please re-run preprocessing in Model Training.")
    st.stop()

prep   = st.session_state["preprocessed"]
models = st.session_state["models"]

# ── 1. Audit configuration ─────────────────────────────────────────────────
st.markdown("### 1️⃣ Audit Configuration")
cfg1, cfg2, cfg3 = st.columns(3)

with cfg1:
    sensitive_choice = st.selectbox(
        "Sensitive Attribute",
        ["Age Group (age_binary)", "Sex (sex)"],
        help="The demographic attribute to audit for fairness",
    )
    use_age = "age" in sensitive_choice.lower()

with cfg2:
    model_choice = st.selectbox(
        "Model to Audit",
        ["All Models"] + list(models.keys()),
    )

with cfg3:
    threshold = st.slider(
        "Fairness Threshold",
        0.05, 0.30, 0.10, 0.01,
        help="Difference below this value is considered 'potentially fair'",
    )

if use_age:
    A_test = prep["A_test"].values
    group_labels = {0: "Young (≤25)", 1: "Older (>25)"}
    sensitive_name = "Age Group"
else:
    A_test = prep["A_sex_test"].values
    group_labels = {0: "Female", 1: "Male"}
    sensitive_name = "Sex"

y_test = prep["y_test"]

# ── 2. Run Audit ───────────────────────────────────────────────────────────
if st.button("▶️ Run Fairness Audit", type="primary"):
    with st.spinner("Computing fairness metrics..."):
        audit_models = (
            models if model_choice == "All Models"
            else {model_choice: models[model_choice]}
        )
        new_fairness = {}
        for name, res in audit_models.items():
            y_pred = res["metrics"]["y_pred"]
            new_fairness[name] = run_all_metrics(y_test, y_pred, A_test, name)

        st.session_state["fairness"] = new_fairness
        st.session_state["audit_run"] = True
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state["activity_log"].append(
            f"[{ts}] Fairness audit completed for {list(new_fairness.keys())}"
        )
    st.success("✅ Fairness audit complete!")
    st.rerun()

fairness = st.session_state["fairness"]
if not fairness:
    st.info("👆 Click **Run Fairness Audit** to calculate fairness metrics.")
    st.stop()

st.markdown("---")

# ── 3. Per-model results ───────────────────────────────────────────────────
st.markdown("### 2️⃣ Fairness Results by Model")

model_tabs = st.tabs(list(fairness.keys()))

METRIC_EXPLANATIONS = {
    "Demographic Parity": {
        "formula": "DPD = |P(Ŷ=1 | A=0) − P(Ŷ=1 | A=1)|",
        "explain": "Checks whether the model approves applicants at the **same rate** regardless of group. A value near 0 means both groups have similar approval rates.",
        "threshold_note": "Common threshold: DPD < 0.10 (10% difference)",
    },
    "Equalized Odds": {
        "formula": "EOdds = max(|TPR₀ − TPR₁|, |FPR₀ − FPR₁|)",
        "explain": "Requires BOTH True Positive Rate AND False Positive Rate to be equal across groups. It is a stronger condition than Equal Opportunity.",
        "threshold_note": "Common threshold: EOdds Diff < 0.10",
    },
    "Equal Opportunity": {
        "formula": "EO_diff = |TPR₀ − TPR₁|  where TPR = TP/(TP+FN)",
        "explain": "Among truly creditworthy applicants, both groups should have the **same chance of approval**. Only looks at qualified applicants (y_true=1).",
        "threshold_note": "Common threshold: EO Diff < 0.10",
    },
    "Predictive Parity": {
        "formula": "PP_diff = |Precision₀ − Precision₁|  where Precision = TP/(TP+FP)",
        "explain": "When the model predicts 'Good Credit', it should be **equally accurate** for both groups. Protects against high error rates for one group.",
        "threshold_note": "Common threshold: PP Diff < 0.10",
    },
}

METRIC_MAP = [
    ("Demographic Parity",  "demographic_parity",   "demographic_parity_difference"),
    ("Equalized Odds",      "equalized_odds",        "equalized_odds_difference"),
    ("Equal Opportunity",   "equal_opportunity",     "equal_opportunity_difference"),
    ("Predictive Parity",   "predictive_parity",     "predictive_parity_difference"),
]

for tab, (model_name, model_fairness) in zip(model_tabs, fairness.items()):
    with tab:
        # KPI row
        cols = st.columns(4)
        for col, (mname, mkey, diff_key) in zip(cols, METRIC_MAP):
            val = model_fairness[mkey][diff_key]
            if val is None:
                badge = "<span class='badge badge-neutral'>Insufficient data</span>"
                color = "#94A3B8"
            elif val < threshold:
                badge = "<span class='badge badge-success'>✓ Below threshold</span>"
                color = "#22C55E"
            elif val < threshold * 2:
                badge = "<span class='badge badge-warning'>⚠ Potential concern</span>"
                color = "#F59E0B"
            else:
                badge = "<span class='badge badge-danger'>✗ Fairness concern</span>"
                color = "#EF4444"

            col.markdown(f"""
            <div class='aft-card'>
              <div class='aft-card-title'>{mname}</div>
              <div class='aft-card-value' style='color:{color};font-size:1.6rem;'>
                {f"{val:.4f}" if val is not None else "N/A"}
              </div>
              <div style='margin-top:8px;'>{badge}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric detail tabs
        m_tabs = st.tabs(["📊 Demographic Parity", "⚖️ Equalized Odds",
                          "🎯 Equal Opportunity", "🔮 Predictive Parity"])

        # --- Demographic Parity ---
        with m_tabs[0]:
            dp = model_fairness["demographic_parity"]
            with st.expander("📖 What is Demographic Parity?", expanded=False):
                info_dp = METRIC_EXPLANATIONS["Demographic Parity"]
                st.markdown(f"**Formula:** `{info_dp['formula']}`")
                st.markdown(info_dp["explain"])
                st.caption(info_dp["threshold_note"])

            c1, c2 = st.columns(2)
            with c1:
                fig_dp = plot_group_selection_rates(
                    model_fairness,
                    sensitive_labels=tuple(group_labels.values()),
                )
                st.plotly_chart(fig_dp, use_container_width=True)
            with c2:
                g0 = dp["group_0"]
                g1 = dp["group_1"]
                st.markdown(f"""
                | Group | n | Selection Rate |
                |-------|---|---------------|
                | {group_labels[0]} | {g0['n']} | {g0['selection_rate']:.4f} |
                | {group_labels[1]} | {g1['n']} | {g1['selection_rate']:.4f} |
                | **Difference** | | **{dp['demographic_parity_difference']:.4f}** |
                | Ratio | | {dp['demographic_parity_ratio'] or 'N/A'} |
                """)
                satisfied = dp["satisfied"]
                if satisfied:
                    st.success("✅ Demographic Parity threshold satisfied")
                elif satisfied is False:
                    st.warning("⚠️ Demographic Parity concern detected")
                else:
                    st.info("ℹ️ Insufficient data to conclude")

        # --- Equalized Odds ---
        with m_tabs[1]:
            eo = model_fairness["equalized_odds"]
            with st.expander("📖 What is Equalized Odds?", expanded=False):
                info_eo = METRIC_EXPLANATIONS["Equalized Odds"]
                st.markdown(f"**Formula:** `{info_eo['formula']}`")
                st.markdown(info_eo["explain"])
                st.caption(info_eo["threshold_note"])

            c1, c2 = st.columns(2)
            with c1:
                fig_eo = plot_equalized_odds_bars(
                    model_fairness,
                    sensitive_labels=tuple(group_labels.values()),
                )
                st.plotly_chart(fig_eo, use_container_width=True)
            with c2:
                st.markdown(f"""
                | Metric | {group_labels[0]} | {group_labels[1]} | Difference |
                |--------|------|------|-----------|
                | TPR    | {eo['TPR_group_0']:.4f} | {eo['TPR_group_1']:.4f} | {eo['TPR_difference']:.4f} |
                | FPR    | {eo['FPR_group_0']:.4f} | {eo['FPR_group_1']:.4f} | {eo['FPR_difference']:.4f} |
                | **EOdds Diff** | | | **{eo['equalized_odds_difference']:.4f}** |
                """)
                if eo["satisfied"]:
                    st.success("✅ Equalized Odds threshold satisfied")
                elif eo["satisfied"] is False:
                    st.warning("⚠️ Equalized Odds concern detected")

        # --- Equal Opportunity ---
        with m_tabs[2]:
            eop = model_fairness["equal_opportunity"]
            with st.expander("📖 What is Equal Opportunity?", expanded=False):
                info_eop = METRIC_EXPLANATIONS["Equal Opportunity"]
                st.markdown(f"**Formula:** `{info_eop['formula']}`")
                st.markdown(info_eop["explain"])
                st.caption(info_eop["threshold_note"])

            import plotly.graph_objects as go
            fig_eop = go.Figure(go.Bar(
                x=[group_labels[0], group_labels[1]],
                y=[eop["TPR_group_0"], eop["TPR_group_1"]],
                marker_color=["#06B6D4", "#6366F1"],
                text=[f"{eop['TPR_group_0']:.4f}", f"{eop['TPR_group_1']:.4f}"],
                textposition="outside",
            ))
            fig_eop.update_layout(
                paper_bgcolor="#161F32", plot_bgcolor="#161F32",
                font=dict(color="#F8FAFC"),
                title="True Positive Rate (Recall) by Group",
                yaxis=dict(title="TPR", range=[0, 1.1], gridcolor="#1E293B"),
                xaxis=dict(color="#94A3B8"),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_eop, use_container_width=True)
            with c2:
                st.markdown(f"""
                | Group | TPR (Recall) |
                |-------|-------------|
                | {group_labels[0]} | {eop['TPR_group_0']:.4f} |
                | {group_labels[1]} | {eop['TPR_group_1']:.4f} |
                | **EO Diff** | **{eop['equal_opportunity_difference']:.4f}** |
                """)
                if eop["satisfied"]:
                    st.success("✅ Equal Opportunity threshold satisfied")
                elif eop["satisfied"] is False:
                    st.warning("⚠️ Equal Opportunity concern detected")

        # --- Predictive Parity ---
        with m_tabs[3]:
            pp = model_fairness["predictive_parity"]
            with st.expander("📖 What is Predictive Parity?", expanded=False):
                info_pp = METRIC_EXPLANATIONS["Predictive Parity"]
                st.markdown(f"**Formula:** `{info_pp['formula']}`")
                st.markdown(info_pp["explain"])
                st.caption(info_pp["threshold_note"])

            fig_pp = go.Figure(go.Bar(
                x=[group_labels[0], group_labels[1]],
                y=[pp["precision_group_0"], pp["precision_group_1"]],
                marker_color=["#22C55E", "#F59E0B"],
                text=[f"{pp['precision_group_0']:.4f}",
                      f"{pp['precision_group_1']:.4f}"],
                textposition="outside",
            ))
            fig_pp.update_layout(
                paper_bgcolor="#161F32", plot_bgcolor="#161F32",
                font=dict(color="#F8FAFC"),
                title="Precision by Group",
                yaxis=dict(title="Precision", range=[0, 1.1], gridcolor="#1E293B"),
                xaxis=dict(color="#94A3B8"),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_pp, use_container_width=True)
            with c2:
                st.markdown(f"""
                | Group | Precision |
                |-------|----------|
                | {group_labels[0]} | {pp['precision_group_0']:.4f} |
                | {group_labels[1]} | {pp['precision_group_1']:.4f} |
                | **PP Diff** | **{pp['predictive_parity_difference']:.4f}** |
                """)
                if pp["satisfied"]:
                    st.success("✅ Predictive Parity threshold satisfied")
                elif pp["satisfied"] is False:
                    st.warning("⚠️ Predictive Parity concern detected")

st.markdown("---")

# ── 3. Cross-metric summary ────────────────────────────────────────────────
st.markdown("### 3️⃣ Fairness Summary Table")
summary_df = metrics_summary_df(fairness)
st.dataframe(summary_df.style.format({
    "Dem. Parity Diff":  "{:.4f}",
    "Eq. Odds Diff":     "{:.4f}",
    "Eq. Opp. Diff":     "{:.4f}",
    "Pred. Parity Diff": "{:.4f}",
}).background_gradient(
    subset=["Dem. Parity Diff", "Eq. Odds Diff",
            "Eq. Opp. Diff",    "Pred. Parity Diff"],
    cmap="RdYlGn_r",
), use_container_width=True)

st.markdown("""
<div class='aft-card'>
  <div class='aft-card-title'>⚠️ Important Fairness Disclaimer</div>
  <p style='color:#94A3B8;font-size:0.85rem;margin:0;'>
    <b>No model shown here is certified as completely unbiased.</b>
    Different fairness metrics can mathematically conflict — it is impossible
    to satisfy all of them simultaneously in most real-world scenarios
    (Chouldechova 2017, Kleinberg et al. 2016). Fairness thresholds
    (e.g., &lt;0.10 difference) are conventions, not legal definitions.
    Decisions based on model outputs should involve domain experts and
    ethical review beyond what this toolkit provides.
  </p>
</div>
""", unsafe_allow_html=True)

# ── 4. Cross-model comparison chart ───────────────────────────────────────
if len(fairness) > 1:
    st.markdown("### 4️⃣ Cross-Model Fairness Comparison")
    metric_sel = st.selectbox(
        "Fairness metric to compare",
        ["demographic_parity_difference", "equalized_odds_difference",
         "equal_opportunity_difference", "predictive_parity_difference"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    fig_cmp = plot_fairness_comparison(
        fairness, metric_sel,
        title=metric_sel.replace("_", " ").title() + " — All Models",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

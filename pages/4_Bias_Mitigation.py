"""
4_Bias_Mitigation.py
---------------------
Page: Bias Mitigation

Three interactive mitigation cards:
  A. Pre-processing  — Reweighing
  B. In-processing   — ExponentiatedGradient
  C. Post-processing — ThresholdOptimizer

Each card shows technique explanation, runs mitigation on demand,
displays before-and-after metrics, and compares fairness results.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bias_mitigation import (
    apply_preprocessing, apply_inprocessing, apply_postprocessing,
)
from src.fairness_metrics import run_all_metrics
from src.visualization import plot_fairness_comparison

# ── Page config & CSS ──────────────────────────────────────────────────────
st.set_page_config(page_title="Bias Mitigation", page_icon="🛡️", layout="wide")
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class='page-header'>
  <h1>🛡️ Bias Mitigation</h1>
  <p>Apply pre-, in-, and post-processing strategies to reduce demographic bias</p>
</div>
""", unsafe_allow_html=True)

# ── Session defaults ───────────────────────────────────────────────────────
for k, v in [("models", {}), ("preprocessed", None), ("fairness", {}),
             ("mitigation", {}), ("mit_fairness", {}), ("activity_log", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Guards ─────────────────────────────────────────────────────────────────
if not st.session_state["models"]:
    st.warning("⚠️ No trained models. Please go to **Model Training** first.")
    st.page_link("pages/2_Model_Training.py", label="Go to Model Training")
    st.stop()

prep   = st.session_state["preprocessed"]
if prep is None:
    st.warning("⚠️ Preprocessing data not found. Re-run preprocessing.")
    st.stop()

models = st.session_state["models"]

X_train, y_train = prep["X_train"], prep["y_train"]
X_val,   y_val   = prep["X_val"],   prep["y_val"]
X_test,  y_test  = prep["X_test"],  prep["y_test"]
A_train = prep["A_train"].values
A_val   = prep["A_val"].values
A_test  = prep["A_test"].values

# baseline model for post-processing
baseline_model = models.get("Logistic Regression", list(models.values())[0])["model"]
baseline_metrics = models.get(
    "Logistic Regression", list(models.values())[0])["metrics"]

group_labels = {0: "Young (≤25)", 1: "Older (>25)"}


# ── Helper: before/after comparison bar chart ─────────────────────────────
def _ba_chart(before: dict, after: dict, title: str) -> go.Figure:
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1"]
    b_vals  = [before[m] for m in metrics]
    a_vals  = [after[m]  for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=labels, y=b_vals,
                         marker_color="#6366F1",
                         text=[f"{v:.3f}" for v in b_vals],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="After",  x=labels, y=a_vals,
                         marker_color="#22C55E",
                         text=[f"{v:.3f}" for v in a_vals],
                         textposition="outside"))
    fig.update_layout(
        paper_bgcolor="#161F32", plot_bgcolor="#161F32",
        font=dict(color="#F8FAFC"),
        title=title, barmode="group",
        yaxis=dict(title="Score", range=[0, 1.1], gridcolor="#1E293B"),
        xaxis=dict(color="#94A3B8"),
        legend=dict(bgcolor="#161F32"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _fairness_ba_chart(before_fair: dict, after_fair: dict,
                        technique_name: str) -> go.Figure:
    metric_labels = ["Dem. Parity Diff", "Eq. Odds Diff",
                     "Eq. Opp. Diff",   "Pred. Parity Diff"]
    before_keys = [
        ("demographic_parity",  "demographic_parity_difference"),
        ("equalized_odds",      "equalized_odds_difference"),
        ("equal_opportunity",   "equal_opportunity_difference"),
        ("predictive_parity",   "predictive_parity_difference"),
    ]
    b_vals = [before_fair[k][s] or 0 for k, s in before_keys]
    a_vals = [after_fair[k][s]  or 0 for k, s in before_keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=metric_labels, y=b_vals,
                         marker_color="#EF4444",
                         text=[f"{v:.4f}" for v in b_vals],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="After",  x=metric_labels, y=a_vals,
                         marker_color="#22C55E",
                         text=[f"{v:.4f}" for v in a_vals],
                         textposition="outside"))
    fig.add_hline(y=0.10, line_dash="dash", line_color="#F59E0B",
                  annotation_text="Threshold (0.10)",
                  annotation_font_color="#F59E0B")
    fig.update_layout(
        paper_bgcolor="#161F32", plot_bgcolor="#161F32",
        font=dict(color="#F8FAFC"),
        title=f"Fairness Before vs After — {technique_name}",
        barmode="group",
        yaxis=dict(title="Fairness Difference (lower=fairer)",
                   range=[0, max(max(b_vals), max(a_vals)) + 0.05],
                   gridcolor="#1E293B"),
        xaxis=dict(color="#94A3B8"),
        legend=dict(bgcolor="#161F32"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _delta(before_val: float, after_val: float) -> str:
    diff = after_val - before_val
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.4f}"


# ── Get baseline fairness (from audit or compute now) ─────────────────────
if "Logistic Regression" in st.session_state["fairness"]:
    baseline_fairness = st.session_state["fairness"]["Logistic Regression"]
else:
    y_pred_base = baseline_metrics["y_pred"]
    baseline_fairness = run_all_metrics(y_test, y_pred_base, A_test,
                                        "Logistic Regression (baseline)")

# ══════════════════════════════════════════════════════════════════════════
# A. PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## A. Pre-Processing Mitigation")

with st.container():
    st.markdown("""
    <div class='aft-card'>
      <div class='aft-card-title'>🔵 TECHNIQUE: REWEIGHING</div>
      <p style='color:#94A3B8;font-size:0.85rem;margin:4px 0 0 0;'>
        <b>How it works:</b> Before training, each training sample is assigned a
        weight based on how over- or under-represented its (group, label) combination
        is. Groups that are under-represented get higher weights, so the model
        pays more attention to them during training. The original data is NOT
        changed — only the importance of each sample is adjusted.<br><br>
        <b>Formula:</b>
        <code>w(A,Y) = P(A) × P(Y) / P(A,Y)</code><br><br>
        <b>Limitation:</b> Only addresses the training distribution;
        does not constrain the model directly.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Apply Pre-Processing (Reweighing)", key="btn_pre"):
        with st.spinner("Training with reweighing..."):
            result = apply_preprocessing(
                X_train, y_train, A_train, X_test, y_test)
            fair = run_all_metrics(y_test, result["y_pred"], A_test,
                                   "Pre-processing")
            st.session_state["mitigation"]["Pre-processing"] = result
            st.session_state["mit_fairness"]["Pre-processing"] = fair
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state["activity_log"].append(
                f"[{ts}] Pre-processing applied — "
                f"Acc {result['metrics']['accuracy']:.1%}"
            )
        st.success("✅ Pre-processing applied!")
        st.rerun()

pre_res = st.session_state["mitigation"].get("Pre-processing")
pre_fair = st.session_state["mit_fairness"].get("Pre-processing")

if pre_res:
    c1, c2 = st.columns(2)
    with c1:
        fig_pre_perf = _ba_chart(baseline_metrics, pre_res["metrics"],
                                  "Performance: Baseline vs Reweighing")
        st.plotly_chart(fig_pre_perf, use_container_width=True)
    with c2:
        fig_pre_fair = _fairness_ba_chart(baseline_fairness, pre_fair,
                                           "Reweighing")
        st.plotly_chart(fig_pre_fair, use_container_width=True)

    st.markdown("**Metric Changes**")
    before_m = baseline_metrics
    after_m  = pre_res["metrics"]
    st.markdown(f"""
    | Metric | Before | After | Change |
    |--------|--------|-------|--------|
    | Accuracy  | {before_m['accuracy']:.4f}  | {after_m['accuracy']:.4f}  | {_delta(before_m['accuracy'],  after_m['accuracy'])} |
    | Precision | {before_m['precision']:.4f} | {after_m['precision']:.4f} | {_delta(before_m['precision'], after_m['precision'])} |
    | Recall    | {before_m['recall']:.4f}    | {after_m['recall']:.4f}    | {_delta(before_m['recall'],    after_m['recall'])} |
    | F1-Score  | {before_m['f1']:.4f}        | {after_m['f1']:.4f}        | {_delta(before_m['f1'],        after_m['f1'])} |
    | Dem. Parity Diff | {baseline_fairness['demographic_parity']['demographic_parity_difference']:.4f} | {pre_fair['demographic_parity']['demographic_parity_difference']:.4f} | {_delta(baseline_fairness['demographic_parity']['demographic_parity_difference'], pre_fair['demographic_parity']['demographic_parity_difference'])} |
    """)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# B. IN-PROCESSING
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## B. In-Processing Mitigation")

with st.container():
    st.markdown("""
    <div class='aft-card'>
      <div class='aft-card-title'>🟣 TECHNIQUE: EXPONENTIATED GRADIENT</div>
      <p style='color:#94A3B8;font-size:0.85rem;margin:4px 0 0 0;'>
        <b>How it works:</b> Fairlearn's ExponentiatedGradient turns the fairness
        problem into a constrained optimisation. It trains a sequence of classifiers,
        with each iteration updating Lagrange multipliers (penalty coefficients) to
        push the model toward satisfying the DemographicParity constraint within
        a tolerance <code>eps=0.02</code> (2%). The final predictor is a probabilistic
        mixture of these classifiers.<br><br>
        <b>Constraint used:</b> Demographic Parity<br>
        <b>Tolerance (eps):</b> 0.02 — allows up to 2% selection rate gap<br><br>
        <b>Limitation:</b> Output is a randomised classifier; different runs may give
        slightly different predictions.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Apply In-Processing (ExponentiatedGradient)", key="btn_in"):
        with st.spinner("Training constrained classifier (may take ~30 seconds)..."):
            result = apply_inprocessing(
                X_train, y_train, A_train,
                X_test, y_test, A_test)
            fair = run_all_metrics(y_test, result["y_pred"], A_test,
                                   "In-processing")
            st.session_state["mitigation"]["In-processing"] = result
            st.session_state["mit_fairness"]["In-processing"] = fair
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state["activity_log"].append(
                f"[{ts}] In-processing applied — "
                f"Acc {result['metrics']['accuracy']:.1%}"
            )
        st.success("✅ In-processing applied!")
        st.rerun()

in_res = st.session_state["mitigation"].get("In-processing")
in_fair = st.session_state["mit_fairness"].get("In-processing")

if in_res:
    c1, c2 = st.columns(2)
    with c1:
        fig_in_perf = _ba_chart(baseline_metrics, in_res["metrics"],
                                 "Performance: Baseline vs ExpGrad")
        st.plotly_chart(fig_in_perf, use_container_width=True)
    with c2:
        fig_in_fair = _fairness_ba_chart(baseline_fairness, in_fair,
                                          "ExponentiatedGradient")
        st.plotly_chart(fig_in_fair, use_container_width=True)

    before_m = baseline_metrics
    after_m  = in_res["metrics"]
    st.markdown(f"""
    | Metric | Before | After | Change |
    |--------|--------|-------|--------|
    | Accuracy  | {before_m['accuracy']:.4f}  | {after_m['accuracy']:.4f}  | {_delta(before_m['accuracy'],  after_m['accuracy'])} |
    | Precision | {before_m['precision']:.4f} | {after_m['precision']:.4f} | {_delta(before_m['precision'], after_m['precision'])} |
    | Recall    | {before_m['recall']:.4f}    | {after_m['recall']:.4f}    | {_delta(before_m['recall'],    after_m['recall'])} |
    | F1-Score  | {before_m['f1']:.4f}        | {after_m['f1']:.4f}        | {_delta(before_m['f1'],        after_m['f1'])} |
    | Dem. Parity Diff | {baseline_fairness['demographic_parity']['demographic_parity_difference']:.4f} | {in_fair['demographic_parity']['demographic_parity_difference']:.4f} | {_delta(baseline_fairness['demographic_parity']['demographic_parity_difference'], in_fair['demographic_parity']['demographic_parity_difference'])} |
    """)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# C. POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════
st.markdown("## C. Post-Processing Mitigation")

with st.container():
    st.markdown("""
    <div class='aft-card'>
      <div class='aft-card-title'>🟡 TECHNIQUE: THRESHOLD OPTIMIZER</div>
      <p style='color:#94A3B8;font-size:0.85rem;margin:4px 0 0 0;'>
        <b>How it works:</b> After a base Logistic Regression model is trained,
        ThresholdOptimizer searches for the best decision threshold
        <em>per demographic group</em> that satisfies the EqualizedOdds constraint.
        Instead of using a single threshold of 0.5 for everyone, it might use
        0.4 for one group and 0.55 for another to equalise TPR and FPR.<br><br>
        <b>Constraint used:</b> Equalized Odds<br>
        <b>Tuned on:</b> Validation set only (test set is never seen during tuning)<br><br>
        <b>Limitation:</b> Requires knowledge of the sensitive attribute at
        prediction time (not always acceptable or legal in deployment).
      </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Apply Post-Processing (ThresholdOptimizer)", key="btn_post"):
        with st.spinner("Optimising thresholds..."):
            result = apply_postprocessing(
                baseline_model,
                X_train, y_train, A_train,
                X_val,   y_val,   A_val,
                X_test,  y_test,  A_test)
            fair = run_all_metrics(y_test, result["y_pred"], A_test,
                                   "Post-processing")
            st.session_state["mitigation"]["Post-processing"] = result
            st.session_state["mit_fairness"]["Post-processing"] = fair
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state["activity_log"].append(
                f"[{ts}] Post-processing applied — "
                f"Acc {result['metrics']['accuracy']:.1%}"
            )
        st.success("✅ Post-processing applied!")
        st.rerun()

post_res = st.session_state["mitigation"].get("Post-processing")
post_fair = st.session_state["mit_fairness"].get("Post-processing")

if post_res:
    c1, c2 = st.columns(2)
    with c1:
        fig_post_perf = _ba_chart(baseline_metrics, post_res["metrics"],
                                   "Performance: Baseline vs ThresholdOpt")
        st.plotly_chart(fig_post_perf, use_container_width=True)
    with c2:
        fig_post_fair = _fairness_ba_chart(baseline_fairness, post_fair,
                                            "ThresholdOptimizer")
        st.plotly_chart(fig_post_fair, use_container_width=True)

    before_m = baseline_metrics
    after_m  = post_res["metrics"]
    st.markdown(f"""
    | Metric | Before | After | Change |
    |--------|--------|-------|--------|
    | Accuracy  | {before_m['accuracy']:.4f}  | {after_m['accuracy']:.4f}  | {_delta(before_m['accuracy'],  after_m['accuracy'])} |
    | Precision | {before_m['precision']:.4f} | {after_m['precision']:.4f} | {_delta(before_m['precision'], after_m['precision'])} |
    | Recall    | {before_m['recall']:.4f}    | {after_m['recall']:.4f}    | {_delta(before_m['recall'],    after_m['recall'])} |
    | F1-Score  | {before_m['f1']:.4f}        | {after_m['f1']:.4f}        | {_delta(before_m['f1'],        after_m['f1'])} |
    | Eq. Odds Diff | {baseline_fairness['equalized_odds']['equalized_odds_difference']:.4f} | {post_fair['equalized_odds']['equalized_odds_difference']:.4f} | {_delta(baseline_fairness['equalized_odds']['equalized_odds_difference'], post_fair['equalized_odds']['equalized_odds_difference'])} |
    """)

# ── Summary card ───────────────────────────────────────────────────────────
applied = {k: v for k, v in st.session_state["mitigation"].items() if v}
if applied:
    st.markdown("---")
    st.markdown("### 📊 All Mitigation Results — Summary")
    summary_rows = []
    for tech, res in applied.items():
        m = res["metrics"]
        f = st.session_state["mit_fairness"].get(tech, {})
        summary_rows.append({
            "Technique":       tech,
            "Accuracy":        m["accuracy"],
            "Precision":       m["precision"],
            "Recall":          m["recall"],
            "F1":              m["f1"],
            "DPD (↓ better)":  f.get("demographic_parity", {}).get("demographic_parity_difference"),
            "EOdds (↓ better)":f.get("equalized_odds",      {}).get("equalized_odds_difference"),
        })

    # Add baseline row
    bf = baseline_fairness
    summary_rows.insert(0, {
        "Technique":       "Baseline (LR)",
        "Accuracy":        baseline_metrics["accuracy"],
        "Precision":       baseline_metrics["precision"],
        "Recall":          baseline_metrics["recall"],
        "F1":              baseline_metrics["f1"],
        "DPD (↓ better)":  bf["demographic_parity"]["demographic_parity_difference"],
        "EOdds (↓ better)":bf["equalized_odds"]["equalized_odds_difference"],
    })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df.style.format({
            c: "{:.4f}" for c in
            ["Accuracy","Precision","Recall","F1","DPD (↓ better)","EOdds (↓ better)"]
        }),
        use_container_width=True,
    )

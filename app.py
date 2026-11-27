"""
app.py  —  Algorithmic Fairness Audit & Bias Mitigation Toolkit
==============================================================
Entry point.  Run with:
    streamlit run app.py

This file:
  1. Configures the Streamlit page (title, icon, layout).
  2. Injects global CSS.
  3. Renders the sidebar with navigation links.
  4. Shows the Dashboard Overview (home page).
  5. Initialises Streamlit session state shared across pages.
"""

import os
import streamlit as st
import pandas as pd

# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Fairness Audit Toolkit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    None,
        "Report a bug": None,
        "About": "Algorithmic Fairness Audit & Bias Mitigation Toolkit\nB.Tech CSE AI/ML Project",
    },
)

# ── Inject global CSS ──────────────────────────────────────────────────────
css_path = os.path.join("assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Initialise session state ───────────────────────────────────────────────
# These keys are used across multiple pages via st.session_state
defaults = {
    "df":             None,   # raw DataFrame
    "preprocessed":   None,   # dict from preprocessing.preprocess()
    "models":         {},     # {model_name: {model, metrics}}
    "fairness":       {},     # {model_name: run_all_metrics(...)}
    "mitigation":     {},     # {technique: {model, metrics, y_pred}}
    "mit_fairness":   {},     # {technique: run_all_metrics(...)}
    "dataset_loaded": False,
    "models_trained": False,
    "audit_run":      False,
    "data_path":      os.path.join("data", "german_credit.csv"),
    "sensitive_col":  "age_binary",
    "sensitive_label": "Age Group",
    "group_labels":   {0: "Young (≤25)", 1: "Older (>25)"},
    "fairness_threshold": 0.10,
    "activity_log":   [],     # list of (timestamp, event) strings
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px 0;'>
      <div style='font-size:2rem;'>⚖️</div>
      <div style='font-size:1rem; font-weight:700; color:#F8FAFC;'>
        Fairness Audit
      </div>
      <div style='font-size:0.72rem; color:#94A3B8; letter-spacing:0.06em;'>
        BIAS MITIGATION TOOLKIT
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Status indicators
    ds_ok  = st.session_state["dataset_loaded"]
    mdl_ok = st.session_state["models_trained"]
    aud_ok = st.session_state["audit_run"]

    def _dot(ok): return "🟢" if ok else "🔴"

    st.markdown(f"""
    <div style='font-size:0.75rem; color:#94A3B8; padding:0 4px;'>
      {_dot(ds_ok)}  Dataset &nbsp;&nbsp; {_dot(mdl_ok)}  Models &nbsp;&nbsp;
      {_dot(aud_ok)}  Audit
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    nav = {
        "🏠  Dashboard":        "app",
        "📂  Dataset Explorer": "pages/1_Dataset_Explorer",
        "🤖  Model Training":   "pages/2_Model_Training",
        "⚖️  Fairness Audit":   "pages/3_Fairness_Audit",
        "🛡️  Bias Mitigation":  "pages/4_Bias_Mitigation",
        "📊  Model Comparison": "pages/5_Model_Comparison",
        "📄  Reports":          "pages/6_Reports",
        "⚙️  Settings":         "pages/7_Settings",
    }
    st.markdown(
        "<p style='font-size:0.7rem; color:#94A3B8; letter-spacing:.08em; "
        "text-transform:uppercase; margin-bottom:4px;'>Navigation</p>",
        unsafe_allow_html=True,
    )
    for label, _ in nav.items():
        st.markdown(
            f"<div style='padding:4px 0; color:#94A3B8; font-size:0.85rem;'>"
            f"{label}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.7rem; color:#475569; text-align:center;'>"
        "B.Tech CSE AI/ML · 2024</p>",
        unsafe_allow_html=True,
    )


# ── Dashboard Overview ─────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
  <h1>⚖️ Algorithmic Fairness Intelligence</h1>
  <p>Audit machine learning models for demographic bias and apply bias mitigation strategies</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards row ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

def _best_model_accuracy():
    if st.session_state["models"]:
        return max(v["metrics"]["accuracy"]
                   for v in st.session_state["models"].values())
    return None

def _fairness_val(metric_key, sub_key):
    if st.session_state["fairness"]:
        # Return best (lowest) value across trained models
        vals = [
            v[metric_key][sub_key]
            for v in st.session_state["fairness"].values()
            if v[metric_key][sub_key] is not None
        ]
        return min(vals) if vals else None
    return None

acc  = _best_model_accuracy()
dpd  = _fairness_val("demographic_parity",  "demographic_parity_difference")
eod  = _fairness_val("equal_opportunity",   "equal_opportunity_difference")
eodd = _fairness_val("equalized_odds",      "equalized_odds_difference")

with col1:
    st.metric(
        label="🎯 Best Model Accuracy",
        value=f"{acc:.1%}" if acc else "—",
        help="Highest accuracy among trained models on the test set",
    )
with col2:
    st.metric(
        label="📊 Demographic Parity Diff",
        value=f"{dpd:.4f}" if dpd is not None else "—",
        help="Selection rate difference between demographic groups (lower = fairer)",
    )
with col3:
    st.metric(
        label="🎯 Equal Opportunity Diff",
        value=f"{eod:.4f}" if eod is not None else "—",
        help="TPR difference between demographic groups (lower = fairer)",
    )
with col4:
    st.metric(
        label="⚖️ Equalized Odds Diff",
        value=f"{eodd:.4f}" if eodd is not None else "—",
        help="Max of TPR and FPR differences (lower = fairer)",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column info panels ─────────────────────────────────────────────────
left, right = st.columns([1, 1])

with left:
    st.markdown("### 📂 Dataset Summary")
    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.markdown(f"""
        <div class='aft-card'>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
            <div><div class='aft-card-title'>Rows</div>
                 <div class='aft-card-value'>{len(df):,}</div></div>
            <div><div class='aft-card-title'>Columns</div>
                 <div class='aft-card-value'>{len(df.columns)}</div></div>
            <div><div class='aft-card-title'>Good Credit</div>
                 <div class='aft-card-value' style='color:#22C55E;'>
                   {df['credit_risk'].sum():,}</div></div>
            <div><div class='aft-card-title'>Bad Credit</div>
                 <div class='aft-card-value' style='color:#EF4444;'>
                   {(df['credit_risk']==0).sum():,}</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📂 No dataset loaded yet. Go to **Dataset Explorer** to load data.")

with right:
    st.markdown("### 🤖 Model Status")
    if st.session_state["models"]:
        for name, res in st.session_state["models"].items():
            acc_v = res["metrics"]["accuracy"]
            st.markdown(f"""
            <div class='aft-card' style='padding:12px 16px; margin-bottom:8px;'>
              <span style='font-weight:600;color:#F8FAFC;'>{name}</span>
              <span class='badge badge-success' style='float:right;'>
                Acc: {acc_v:.1%}
              </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("🤖 No models trained yet. Go to **Model Training** to train models.")

st.markdown("<br>", unsafe_allow_html=True)

# ── About / Introduction ───────────────────────────────────────────────────
with st.expander("📖 About This Toolkit", expanded=not st.session_state["dataset_loaded"]):
    st.markdown("""
    ## What is Algorithmic Fairness?

    Machine learning models trained on historical data can perpetuate and
    amplify existing societal biases. For example, a loan approval model
    trained on historical data may systematically disadvantage younger
    applicants — not because of their creditworthiness, but because of
    historical patterns in the data.

    This toolkit helps you:

    | Step | What You Do |
    |------|-------------|
    | 1️⃣ Dataset Explorer | Load and explore the German Credit Dataset |
    | 2️⃣ Model Training   | Train Logistic Regression, Decision Tree, Random Forest |
    | 3️⃣ Fairness Audit   | Measure 4 fairness metrics across demographic groups |
    | 4️⃣ Bias Mitigation  | Apply pre-, in-, and post-processing mitigation |
    | 5️⃣ Comparison       | Compare accuracy vs fairness tradeoffs |
    | 6️⃣ Reports          | Download a full audit report |

    ### The Four Fairness Metrics
    - **Demographic Parity** — Equal approval rates across groups
    - **Equalized Odds** — Equal TPR and FPR across groups
    - **Equal Opportunity** — Equal TPR (recall) for the positive class
    - **Predictive Parity** — Equal precision for positive predictions

    > ⚠️ **Important:** Satisfying one fairness metric does not guarantee
    > a model is unbiased. Different metrics can mathematically conflict.
    """)

# ── Quick Actions ──────────────────────────────────────────────────────────
st.markdown("### ⚡ Quick Start")
qc1, qc2, qc3, qc4 = st.columns(4)
with qc1:
    st.page_link("pages/1_Dataset_Explorer.py", label="📂 Load Dataset",   icon="📂")
with qc2:
    st.page_link("pages/2_Model_Training.py",   label="🤖 Train Models",   icon="🤖")
with qc3:
    st.page_link("pages/3_Fairness_Audit.py",   label="⚖️ Run Audit",      icon="⚖️")
with qc4:
    st.page_link("pages/4_Bias_Mitigation.py",  label="🛡️ Mitigate Bias",  icon="🛡️")

# ── Activity log ───────────────────────────────────────────────────────────
if st.session_state["activity_log"]:
    st.markdown("### 📋 Recent Activity")
    for event in reversed(st.session_state["activity_log"][-8:]):
        st.markdown(
            f"<div style='font-size:0.82rem;color:#94A3B8;padding:3px 0;'>"
            f"• {event}</div>",
            unsafe_allow_html=True,
        )

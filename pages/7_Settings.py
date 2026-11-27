"""
7_Settings.py
--------------
Page: Settings

Provides:
  - Dataset path configuration + auto-download trigger
  - Fairness threshold adjustment
  - Sensitive attribute selection
  - Session state inspection (debug mode)
  - Project information
  - Reset all results button
"""

import os
import sys
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Page config & CSS ──────────────────────────────────────────────────────
st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class='page-header'>
  <h1>⚙️ Settings</h1>
  <p>Configure dataset, fairness thresholds, and project preferences</p>
</div>
""", unsafe_allow_html=True)

# ── Session defaults ───────────────────────────────────────────────────────
for k, v in [
    ("data_path",          os.path.join("data", "german_credit.csv")),
    ("fairness_threshold", 0.10),
    ("sensitive_col",      "age_binary"),
    ("group_labels",       {0: "Young (≤25)", 1: "Older (>25)"}),
    ("models", {}), ("fairness", {}), ("mitigation", {}),
    ("mit_fairness", {}), ("preprocessed", None), ("df", None),
    ("dataset_loaded", False), ("models_trained", False),
    ("audit_run", False), ("activity_log", []),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── 1. Dataset configuration ───────────────────────────────────────────────
st.markdown("### 1️⃣ Dataset Configuration")

new_path = st.text_input(
    "Dataset file path",
    value=st.session_state["data_path"],
    help="Path to the German Credit .data or .csv file",
)
if new_path != st.session_state["data_path"]:
    st.session_state["data_path"] = new_path
    st.success(f"✅ Dataset path updated to: `{new_path}`")

file_exists = os.path.exists(new_path)
if file_exists:
    st.markdown(
        f"<span class='badge badge-success'>✓ File exists at {new_path}</span>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<span class='badge badge-danger'>✗ File not found at {new_path}</span>",
        unsafe_allow_html=True,
    )
    if st.button("📥 Auto-Download Dataset Now"):
        from src.data_loader import download_dataset
        with st.spinner("Downloading..."):
            try:
                download_dataset(new_path)
                st.success("✅ Downloaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}")

st.markdown("---")

# ── 2. Fairness threshold ──────────────────────────────────────────────────
st.markdown("### 2️⃣ Fairness Threshold Configuration")
new_threshold = st.slider(
    "Fairness difference threshold",
    min_value=0.01,
    max_value=0.30,
    value=float(st.session_state["fairness_threshold"]),
    step=0.01,
    help="Fairness metric differences below this value are flagged as 'potentially fair'. "
         "0.10 (10%) is a commonly used threshold in fairness research.",
)
if new_threshold != st.session_state["fairness_threshold"]:
    st.session_state["fairness_threshold"] = new_threshold
    st.info(f"ℹ️ Threshold updated to {new_threshold:.2f}. Re-run the audit to see updated results.")

st.markdown("""
<div class='aft-card'>
  <div class='aft-card-title'>ℹ️ About the Threshold</div>
  <p style='color:#94A3B8;font-size:0.85rem;margin:0;'>
    The fairness threshold (default 0.10) is a convention, not a legal standard.
    In the EU's GDPR context, 80% rule (ratio ≥ 0.80) is sometimes referenced.
    In US employment law (EEOC), the 4/5ths rule applies to adverse impact.
    For this academic project, 0.10 is used as a reasonable research threshold.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── 3. Sensitive attribute ─────────────────────────────────────────────────
st.markdown("### 3️⃣ Default Sensitive Attribute")
sens_choice = st.selectbox(
    "Sensitive attribute for audits",
    ["age_binary (Age Group)", "sex (Sex)"],
    index=0 if st.session_state["sensitive_col"] == "age_binary" else 1,
)
new_sens = "age_binary" if "age" in sens_choice else "sex"
if new_sens != st.session_state["sensitive_col"]:
    st.session_state["sensitive_col"] = new_sens
    if new_sens == "age_binary":
        st.session_state["group_labels"] = {0: "Young (≤25)", 1: "Older (>25)"}
    else:
        st.session_state["group_labels"] = {0: "Female", 1: "Male"}
    st.success(f"✅ Sensitive attribute changed to: `{new_sens}`")

st.markdown("---")

# ── 4. Status overview ─────────────────────────────────────────────────────
st.markdown("### 4️⃣ Current Session Status")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Dataset Loaded",  "✅ Yes" if st.session_state["dataset_loaded"]  else "❌ No")
s2.metric("Models Trained",  "✅ Yes" if st.session_state["models_trained"]  else "❌ No")
s3.metric("Audit Complete",  "✅ Yes" if st.session_state["audit_run"]       else "❌ No")
s4.metric("Models in Memory",str(len(st.session_state["models"])))

if st.session_state["models"]:
    st.markdown("**Trained Models:**")
    for name, res in st.session_state["models"].items():
        acc = res["metrics"]["accuracy"]
        st.markdown(
            f"<span class='badge badge-accent'>{name}</span> "
            f"<span style='color:#94A3B8;font-size:0.82rem;'>Accuracy: {acc:.1%}</span>",
            unsafe_allow_html=True,
        )

if st.session_state["mitigation"]:
    applied = {k for k, v in st.session_state["mitigation"].items() if v}
    if applied:
        st.markdown(f"**Applied Mitigation:** {', '.join(applied)}")

st.markdown("---")

# ── 5. Reset ───────────────────────────────────────────────────────────────
st.markdown("### 5️⃣ Reset Session")
st.warning("⚠️ This will clear all trained models, audit results, and cached data from memory. "
           "Files on disk (models/*.pkl) will NOT be deleted.")

if st.button("🗑️ Reset All Results", type="secondary"):
    for key in ["df","preprocessed","models","fairness","mitigation","mit_fairness",
                "dataset_loaded","models_trained","audit_run","activity_log"]:
        if key in ["models","fairness","mitigation","mit_fairness"]:
            st.session_state[key] = {}
        elif key in ["dataset_loaded","models_trained","audit_run"]:
            st.session_state[key] = False
        elif key == "activity_log":
            st.session_state[key] = []
        else:
            st.session_state[key] = None
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["activity_log"] = [f"[{ts}] Session reset by user"]
    st.success("✅ Session reset. Start from the Dataset Explorer.")
    st.rerun()

st.markdown("---")

# ── 6. Project info ────────────────────────────────────────────────────────
st.markdown("### 6️⃣ Project Information")
st.markdown("""
<div class='aft-card'>
  <div class='aft-card-title'>PROJECT DETAILS</div>
  <table style='color:#94A3B8;font-size:0.85rem;width:100%;border-collapse:collapse;'>
    <tr><td style='padding:4px 0;'><b style='color:#F8FAFC;'>Project Title</b></td>
        <td>Algorithmic Fairness Audit & Bias Mitigation Toolkit</td></tr>
    <tr><td style='padding:4px 0;'><b style='color:#F8FAFC;'>Dataset</b></td>
        <td>German Credit Dataset (UCI, 1994) — 1000 samples, 20 features</td></tr>
    <tr><td style='padding:4px 0;'><b style='color:#F8FAFC;'>ML Models</b></td>
        <td>Logistic Regression, Decision Tree, Random Forest</td></tr>
    <tr><td style='padding:4px 0;'><b style='color:#F8FAFC;'>Fairness Metrics</b></td>
        <td>Demographic Parity, Equalized Odds, Equal Opportunity, Predictive Parity</td></tr>
    <tr><td style='padding:4px 0;'><b style='color:#F8FAFC;'>Mitigation</b></td>
        <td>Reweighing, ExponentiatedGradient, ThresholdOptimizer</td></tr>
    <tr><td style='padding:4px 0;'><b style='color:#F8FAFC;'>Tech Stack</b></td>
        <td>Python, Streamlit, Scikit-learn, Fairlearn, Plotly, Pandas</td></tr>
    <tr><td style='padding:4px 0;'><b style='color:#F8FAFC;'>Sensitive Attributes</b></td>
        <td>Age (binary ≤25 / >25), Sex (derived from personal_status_sex)</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

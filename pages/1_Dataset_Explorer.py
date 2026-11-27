"""
1_Dataset_Explorer.py
----------------------
Page: Dataset Explorer

Lets the user:
  - Auto-download OR upload the German Credit dataset
  - Preview raw data
  - See dataset dimensions, missing values, type distribution
  - Explore class and sensitive attribute distributions interactively
  - Filter rows and download filtered data
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ── Allow imports from project root ───────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import (
    load_dataset, get_dataset_info, describe_missing,
    get_feature_summary, download_dataset, CATEGORICAL_COLS, NUMERIC_COLS,
)
from src.visualization import (
    plot_class_distribution, plot_feature_distribution,
)

# ── Page config & CSS ──────────────────────────────────────────────────────
st.set_page_config(page_title="Dataset Explorer", page_icon="📂",
                   layout="wide")
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Page header ────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
  <h1>📂 Dataset Explorer</h1>
  <p>Load, inspect, and understand the German Credit Dataset before training</p>
</div>
""", unsafe_allow_html=True)

# ── Initialise session defaults (in case page opened directly) ────────────
if "df" not in st.session_state:
    st.session_state["df"] = None
if "dataset_loaded" not in st.session_state:
    st.session_state["dataset_loaded"] = False
if "activity_log" not in st.session_state:
    st.session_state["activity_log"] = []
if "data_path" not in st.session_state:
    st.session_state["data_path"] = os.path.join("data", "german_credit.csv")


# ── Load panel ─────────────────────────────────────────────────────────────
st.markdown("### 1️⃣ Load Dataset")
load_col1, load_col2 = st.columns([2, 1])

with load_col1:
    data_path = st.session_state["data_path"]
    file_exists = os.path.exists(data_path)

    if file_exists:
        st.success(f"✅ Dataset found at `{data_path}`")
    else:
        st.warning(f"⚠️ Dataset not found at `{data_path}`")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("📥 Auto-Download Dataset", use_container_width=True):
            with st.spinner("Downloading from UCI repository..."):
                try:
                    download_dataset(data_path)
                    st.success("✅ Downloaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Download failed: {e}\n\nPlease download manually from:\nhttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

    with col_b:
        uploaded = st.file_uploader("📁 Upload CSV", type=["csv", "data"],
                                    label_visibility="collapsed")
        if uploaded:
            os.makedirs("data", exist_ok=True)
            with open(data_path, "wb") as f_out:
                f_out.write(uploaded.read())
            st.success("✅ File saved!")
            st.rerun()

    with col_c:
        if st.button("🔄 Load / Reload Dataset", use_container_width=True):
            if os.path.exists(data_path):
                with st.spinner("Loading..."):
                    try:
                        df = load_dataset(data_path)
                        st.session_state["df"] = df
                        st.session_state["dataset_loaded"] = True
                        ts = datetime.now().strftime("%H:%M:%S")
                        st.session_state["activity_log"].append(
                            f"[{ts}] Dataset loaded — {len(df):,} rows"
                        )
                        st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading dataset: {e}")
            else:
                st.error("No file found. Please auto-download or upload first.")

# ── Require dataset to be loaded for the rest ──────────────────────────────
if st.session_state["df"] is None:
    st.info("⬆️ Click **Load / Reload Dataset** above to begin exploration.")
    st.stop()

df = st.session_state["df"]
info = get_dataset_info(df)

st.markdown("---")

# ── 2. KPI summary cards ───────────────────────────────────────────────────
st.markdown("### 2️⃣ Dataset at a Glance")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows",          f"{info['n_rows']:,}")
k2.metric("Columns",       info["n_cols"])
k3.metric("Missing Values",info["n_missing"])
k4.metric("Categorical",   info["n_categorical"])
k5.metric("Numerical",     info["n_numeric"])

st.markdown("---")

# ── 3. Data preview with filter ───────────────────────────────────────────
st.markdown("### 3️⃣ Data Preview")
n_rows = st.slider("Rows to display", 5, 50, 10)
st.dataframe(df.head(n_rows), use_container_width=True, height=320)

with st.expander("📋 Feature Summary (all columns)"):
    st.dataframe(get_feature_summary(df), use_container_width=True)

missing_df = describe_missing(df)
if len(missing_df):
    st.warning(f"⚠️ {len(missing_df)} feature(s) have missing values:")
    st.dataframe(missing_df, use_container_width=True)
else:
    st.success("✅ No missing values detected in the dataset.")

st.markdown("---")

# ── 4. Distribution charts ─────────────────────────────────────────────────
st.markdown("### 4️⃣ Distribution Analysis")

tab1, tab2, tab3 = st.tabs(["🎯 Target Variable", "👥 Sensitive Attributes",
                             "📈 Feature Distributions"])

with tab1:
    t1a, t1b = st.columns(2)
    with t1a:
        fig_cls = plot_class_distribution(info["target_distribution"],
                                          "Credit Risk Distribution")
        st.plotly_chart(fig_cls, use_container_width=True)
        st.markdown("""
        <div class='aft-card'>
          <div class='aft-card-title'>Interpretation</div>
          <p style='color:#94A3B8;font-size:0.85rem;margin:0;'>
            The dataset contains more <b style='color:#22C55E;'>Good Credit (1)</b>
            than <b style='color:#EF4444;'>Bad Credit (0)</b> cases.
            This class imbalance means accuracy alone can be misleading —
            always check Precision, Recall, and F1-Score too.
          </p>
        </div>
        """, unsafe_allow_html=True)
    with t1b:
        st.markdown("**Credit Risk Value Counts**")
        vc = df["credit_risk"].value_counts().reset_index()
        vc.columns = ["Credit Risk", "Count"]
        vc["Label"] = vc["Credit Risk"].map({1: "Good (1)", 0: "Bad (0)"})
        vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)
        st.dataframe(vc[["Label","Count","Percentage"]], use_container_width=True)

        st.markdown("**Statistical Summary**")
        st.dataframe(df[NUMERIC_COLS].describe().T.round(2), use_container_width=True)

with tab2:
    s1, s2 = st.columns(2)
    with s1:
        fig_age = plot_class_distribution(
            info["age_binary_distribution"],
            "Age Group Distribution"
        )
        # Override labels for age
        import plotly.graph_objects as go
        fig_age2 = go.Figure(go.Pie(
            labels=["Older (>25)", "Young (≤25)"],
            values=[info["age_binary_distribution"].get(1, 0),
                    info["age_binary_distribution"].get(0, 0)],
            marker=dict(colors=["#6366F1","#06B6D4"],
                        line=dict(color="#0B1020", width=2)),
            hole=0.4,
            textinfo="label+percent",
            textfont=dict(color="#F8FAFC"),
        ))
        fig_age2.update_layout(
            paper_bgcolor="#161F32", plot_bgcolor="#161F32",
            font=dict(color="#F8FAFC"),
            title="Age Group Distribution",
            margin=dict(l=20,r=20,t=50,b=20),
        )
        st.plotly_chart(fig_age2, use_container_width=True)

    with s2:
        fig_sex = go.Figure(go.Pie(
            labels=["Male", "Female"],
            values=[info["sex_distribution"].get(1, 0),
                    info["sex_distribution"].get(0, 0)],
            marker=dict(colors=["#6366F1","#F59E0B"],
                        line=dict(color="#0B1020", width=2)),
            hole=0.4,
            textinfo="label+percent",
            textfont=dict(color="#F8FAFC"),
        ))
        fig_sex.update_layout(
            paper_bgcolor="#161F32", plot_bgcolor="#161F32",
            font=dict(color="#F8FAFC"),
            title="Sex Distribution",
            margin=dict(l=20,r=20,t=50,b=20),
        )
        st.plotly_chart(fig_sex, use_container_width=True)

    st.markdown("""
    <div class='aft-card'>
      <div class='aft-card-title'>⚠️ Sensitive Attribute Limitations</div>
      <p style='color:#94A3B8;font-size:0.85rem;margin:0;'>
        The age threshold of 25 is a common convention in fairness literature
        for this dataset — it is not a legal definition. <br>
        Sex is derived from the <code>personal_status_sex</code> attribute and
        does not capture gender fully. These are proxy representations only.
        Historical patterns in the data may reflect societal biases from 1994 Germany.
      </p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    num_feat = st.selectbox("Select numerical feature to explore",
                            NUMERIC_COLS, key="feat_sel")
    show_by_age = st.checkbox("Split by Age Group", value=True)
    fig_fd = plot_feature_distribution(
        df[num_feat],
        num_feat,
        df["age_binary"] if show_by_age else None,
    )
    st.plotly_chart(fig_fd, use_container_width=True)

    st.markdown("**Correlation with Credit Risk**")
    corr = df[NUMERIC_COLS + ["credit_risk"]].corr()["credit_risk"].drop("credit_risk")
    corr_df = corr.reset_index()
    corr_df.columns = ["Feature", "Correlation with Credit Risk"]
    corr_df["Correlation with Credit Risk"] = corr_df["Correlation with Credit Risk"].round(4)
    st.dataframe(corr_df.sort_values("Correlation with Credit Risk",
                                      ascending=False), use_container_width=True)

st.markdown("---")

# ── 5. Download section ────────────────────────────────────────────────────
st.markdown("### 5️⃣ Export")
dl_col1, dl_col2 = st.columns(2)
with dl_col1:
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download Full Dataset (CSV)",
        data=csv_bytes,
        file_name="german_credit_processed.csv",
        mime="text/csv",
        use_container_width=True,
    )
with dl_col2:
    summary_df = get_feature_summary(df)
    st.download_button(
        label="⬇️ Download Feature Summary (CSV)",
        data=summary_df.to_csv(index=False).encode(),
        file_name="feature_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

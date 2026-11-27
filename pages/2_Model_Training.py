"""
2_Model_Training.py
--------------------
Page: Model Training

Allows the user to:
  - Select one or more ML models to train
  - Configure basic hyperparameters
  - Train with a progress indicator
  - View performance metrics, confusion matrix, and classification report
  - Download trained model files
"""

import os
import sys
import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess
from src.model_training import train_model, evaluate_model, save_model, MODELS_DIR
from src.visualization import plot_confusion_matrix, plot_model_performance

# ── Page config & CSS ──────────────────────────────────────────────────────
st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class='page-header'>
  <h1>🤖 Model Training</h1>
  <p>Train, evaluate, and compare baseline machine learning models</p>
</div>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────
for k, v in [("df", None), ("preprocessed", None), ("models", {}),
             ("fairness", {}), ("models_trained", False),
             ("activity_log", []), ("dataset_loaded", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Guard: need dataset ────────────────────────────────────────────────────
if st.session_state["df"] is None:
    st.warning("⚠️ No dataset loaded. Please go to **Dataset Explorer** first.")
    st.page_link("pages/1_Dataset_Explorer.py", label="Go to Dataset Explorer")
    st.stop()

df = st.session_state["df"]

# ── 1. Preprocessing ───────────────────────────────────────────────────────
st.markdown("### 1️⃣ Preprocessing Configuration")
p_col1, p_col2, p_col3 = st.columns(3)
with p_col1:
    test_size = st.slider("Test Set Size", 0.10, 0.40, 0.20, 0.05,
                          help="Fraction of data held out for final evaluation")
with p_col2:
    val_size = st.slider("Validation Set Size", 0.10, 0.30, 0.20, 0.05,
                         help="Fraction of training data used for threshold tuning")
with p_col3:
    random_state = st.number_input("Random Seed", 1, 999, 42,
                                   help="Set a fixed seed for reproducibility")

if st.button("⚙️ Run Preprocessing", use_container_width=False):
    with st.spinner("Preprocessing data..."):
        prep = preprocess(df, test_size=test_size, val_size=val_size,
                          random_state=int(random_state))
        st.session_state["preprocessed"] = prep
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state["activity_log"].append(
            f"[{ts}] Preprocessing complete — "
            f"Train {len(prep['X_train'])}, Val {len(prep['X_val'])}, "
            f"Test {len(prep['X_test'])}"
        )
    st.success("✅ Preprocessing complete!")

prep = st.session_state.get("preprocessed")
if prep:
    sp1, sp2, sp3 = st.columns(3)
    sp1.metric("Training Samples",   len(prep["X_train"]))
    sp2.metric("Validation Samples", len(prep["X_val"]))
    sp3.metric("Test Samples",       len(prep["X_test"]))

    with st.expander("📋 Feature Names"):
        st.write(f"**{len(prep['feature_names'])} features** after one-hot encoding:")
        st.write(", ".join(prep["feature_names"]))

st.markdown("---")

# ── 2. Model selection & training ─────────────────────────────────────────
st.markdown("### 2️⃣ Model Selection & Training")

MODEL_INFO = {
    "Logistic Regression": {
        "icon": "📉",
        "description": """
        **Logistic Regression** is a linear model that estimates the probability
        of a binary outcome using the sigmoid function. It is fast, interpretable,
        and works well as a baseline. The decision boundary is a straight line
        (or hyperplane in higher dimensions).
        """,
        "pros": "Fast, interpretable, works well for linearly separable data",
        "cons": "Cannot capture non-linear relationships",
    },
    "Decision Tree": {
        "icon": "🌳",
        "description": """
        **Decision Tree** learns a series of if-then-else rules by recursively
        splitting the feature space to maximise information gain. Each node
        represents a feature threshold test; leaves represent class labels.
        """,
        "pros": "Highly interpretable, handles non-linear patterns",
        "cons": "Prone to overfitting without depth limits",
    },
    "Random Forest": {
        "icon": "🌲",
        "description": """
        **Random Forest** builds many Decision Trees on random subsets of the
        data (bootstrap samples) and features, then aggregates their predictions
        by majority vote. This reduces variance (overfitting) compared to a
        single tree.
        """,
        "pros": "High accuracy, robust to noise, feature importance available",
        "cons": "Less interpretable, slower to train",
    },
}

selected_models = st.multiselect(
    "Select models to train",
    list(MODEL_INFO.keys()),
    default=list(MODEL_INFO.keys()),
)

# Show algorithm cards
for name in selected_models:
    info_m = MODEL_INFO[name]
    with st.expander(f"{info_m['icon']} {name} — How it works"):
        st.markdown(info_m["description"])
        st.markdown(f"✅ **Pros:** {info_m['pros']}")
        st.markdown(f"❌ **Cons:** {info_m['cons']}")

if st.button("🚀 Train Selected Models", type="primary",
             use_container_width=False):
    if not prep:
        st.error("Please run preprocessing first.")
    elif not selected_models:
        st.error("Please select at least one model.")
    else:
        progress = st.progress(0, "Training models...")
        new_models = dict(st.session_state.get("models", {}))

        for i, name in enumerate(selected_models):
            progress.progress((i) / len(selected_models),
                               f"Training {name}...")
            model = train_model(name, prep["X_train"], prep["y_train"],
                                random_state=int(random_state))
            metrics = evaluate_model(model, prep["X_test"], prep["y_test"],
                                     model_name=name)
            save_model(model, name)
            new_models[name] = {"model": model, "metrics": metrics}

            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state["activity_log"].append(
                f"[{ts}] {name} trained — Accuracy {metrics['accuracy']:.1%}"
            )

        progress.progress(1.0, "Done!")
        st.session_state["models"] = new_models
        st.session_state["models_trained"] = True
        st.success(f"✅ Trained {len(selected_models)} model(s) successfully!")
        st.rerun()

st.markdown("---")

# ── 3. Results display ─────────────────────────────────────────────────────
models = st.session_state.get("models", {})
if not models:
    st.info("👆 Train models above to see results.")
    st.stop()

st.markdown("### 3️⃣ Model Results")

# Performance summary table
perf_rows = []
for name, res in models.items():
    m = res["metrics"]
    perf_rows.append({
        "Model": name,
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
    height=140,
)

# Performance bar chart
fig_perf = plot_model_performance(models)
st.plotly_chart(fig_perf, use_container_width=True)

# Per-model details in tabs
st.markdown("### 4️⃣ Per-Model Detailed Results")
model_tabs = st.tabs(list(models.keys()))
for tab, (name, res) in zip(model_tabs, models.items()):
    with tab:
        m = res["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{m['accuracy']:.1%}")
        c2.metric("Precision", f"{m['precision']:.1%}")
        c3.metric("Recall",    f"{m['recall']:.1%}")
        c4.metric("F1-Score",  f"{m['f1']:.1%}")

        col_cm, col_rep = st.columns([1, 1])
        with col_cm:
            fig_cm = plot_confusion_matrix(m["confusion_matrix"], name)
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_rep:
            st.markdown("**Classification Report**")
            st.markdown(
                f"<pre style='background:#0B1020;color:#94A3B8;"
                f"border-radius:8px;padding:12px;font-size:0.78rem;"
                f"border:1px solid #1E293B;overflow-x:auto;'>"
                f"{m['classification_report']}</pre>",
                unsafe_allow_html=True,
            )

        # Download model
        model_file = os.path.join(MODELS_DIR,
                                  name.replace(" ", "_") + ".pkl")
        if os.path.exists(model_file):
            with open(model_file, "rb") as f_pkl:
                st.download_button(
                    label=f"⬇️ Download {name} Model (.pkl)",
                    data=f_pkl.read(),
                    file_name=f"{name.replace(' ','_')}.pkl",
                    mime="application/octet-stream",
                )

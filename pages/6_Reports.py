"""
6_Reports.py
-------------
Page: Reports

Generates and allows download of:
  1. Fairness Audit Summary (CSV)
  2. Model Performance Results (CSV)
  3. Full Audit Report (PDF via reportlab)
"""

import os
import sys
import io
import csv
import streamlit as st
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Page config & CSS ──────────────────────────────────────────────────────
st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")
css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class='page-header'>
  <h1>📄 Fairness Audit Reports</h1>
  <p>Generate and download audit reports, model results, and fairness summaries</p>
</div>
""", unsafe_allow_html=True)

# ── Session defaults ───────────────────────────────────────────────────────
for k, v in [("models", {}), ("fairness", {}), ("mitigation", {}),
             ("mit_fairness", {}), ("df", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Build report data ──────────────────────────────────────────────────────
models      = st.session_state["models"]
fairness    = st.session_state["fairness"]
mitigation  = st.session_state["mitigation"]
mit_fairness= st.session_state["mit_fairness"]

has_data = bool(models)

# ── 1. Report summary ──────────────────────────────────────────────────────
st.markdown("### 📋 Audit Summary")
if not has_data:
    st.warning("⚠️ No results to report yet. Train models and run the fairness audit first.")
else:
    total_models = len(models) + len([v for v in mitigation.values() if v])
    st.markdown(f"""
    <div class='aft-card'>
      <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:16px;'>
        <div>
          <div class='aft-card-title'>Models Evaluated</div>
          <div class='aft-card-value'>{total_models}</div>
        </div>
        <div>
          <div class='aft-card-title'>Fairness Metrics</div>
          <div class='aft-card-value'>4</div>
        </div>
        <div>
          <div class='aft-card-title'>Mitigation Techniques</div>
          <div class='aft-card-value'>{len([v for v in mitigation.values() if v])}</div>
        </div>
        <div>
          <div class='aft-card-title'>Generated</div>
          <div class='aft-card-value' style='font-size:1rem;'>
            {datetime.now().strftime("%Y-%m-%d %H:%M")}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── 2. Performance CSV ─────────────────────────────────────────────────────
st.markdown("### 1️⃣ Model Performance Results")

perf_rows = []
for name, res in models.items():
    m = res["metrics"]
    perf_rows.append({
        "Model": name, "Type": "Baseline",
        "Accuracy": m["accuracy"], "Precision": m["precision"],
        "Recall": m["recall"],     "F1": m["f1"],
    })

mit_names = {
    "Pre-processing": "Pre (Reweighing)",
    "In-processing":  "In (ExpGrad)",
    "Post-processing":"Post (ThreshOpt)",
}
for tech, label in mit_names.items():
    if mitigation.get(tech):
        m = mitigation[tech]["metrics"]
        perf_rows.append({
            "Model": label, "Type": "Mitigated",
            "Accuracy": m["accuracy"], "Precision": m["precision"],
            "Recall": m["recall"],     "F1": m["f1"],
        })

if perf_rows:
    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df.style.format({
        c: "{:.4f}" for c in ["Accuracy","Precision","Recall","F1"]
    }), use_container_width=True)

    st.download_button(
        label="⬇️ Download Performance Results (CSV)",
        data=perf_df.to_csv(index=False).encode(),
        file_name="model_performance_results.csv",
        mime="text/csv",
    )
else:
    st.info("Train models first to generate performance results.")

st.markdown("---")

# ── 3. Fairness CSV ────────────────────────────────────────────────────────
st.markdown("### 2️⃣ Fairness Audit Results")

fair_rows = []
all_fairness_combined = dict(fairness)
for tech, label in mit_names.items():
    if mit_fairness.get(tech):
        all_fairness_combined[label] = mit_fairness[tech]

for name, fr in all_fairness_combined.items():
    if not fr:
        continue
    fair_rows.append({
        "Model":                  name,
        "Dem. Parity Diff":       fr.get("demographic_parity",  {}).get("demographic_parity_difference"),
        "Dem. Parity Ratio":      fr.get("demographic_parity",  {}).get("demographic_parity_ratio"),
        "Eq. Odds Diff":          fr.get("equalized_odds",       {}).get("equalized_odds_difference"),
        "TPR Diff (Eq.Odds)":     fr.get("equalized_odds",       {}).get("TPR_difference"),
        "FPR Diff (Eq.Odds)":     fr.get("equalized_odds",       {}).get("FPR_difference"),
        "Eq. Opp. Diff":          fr.get("equal_opportunity",    {}).get("equal_opportunity_difference"),
        "Pred. Parity Diff":      fr.get("predictive_parity",    {}).get("predictive_parity_difference"),
        "DP Threshold (0.10)":    "✓ Pass" if (fr.get("demographic_parity",  {}).get("satisfied")) else "✗ Concern",
        "EO Threshold (0.10)":    "✓ Pass" if (fr.get("equalized_odds",       {}).get("satisfied")) else "✗ Concern",
        "EOpp Threshold (0.10)":  "✓ Pass" if (fr.get("equal_opportunity",    {}).get("satisfied")) else "✗ Concern",
        "PP Threshold (0.10)":    "✓ Pass" if (fr.get("predictive_parity",    {}).get("satisfied")) else "✗ Concern",
    })

if fair_rows:
    fair_df = pd.DataFrame(fair_rows)
    st.dataframe(fair_df, use_container_width=True)

    st.download_button(
        label="⬇️ Download Fairness Audit Results (CSV)",
        data=fair_df.to_csv(index=False).encode(),
        file_name="fairness_audit_results.csv",
        mime="text/csv",
    )
else:
    st.info("Run the Fairness Audit first to generate fairness results.")

st.markdown("---")

# ── 4. PDF Report ──────────────────────────────────────────────────────────
st.markdown("### 3️⃣ Full Audit Report (PDF)")

def _generate_pdf() -> bytes:
    """
    Generate a PDF report using reportlab.
    Returns raw PDF bytes.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # Colour constants
        DARK  = colors.HexColor("#0B1020")
        INDIGO= colors.HexColor("#6366F1")
        GRAY  = colors.HexColor("#94A3B8")
        WHITE = colors.white

        # Title
        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            textColor=INDIGO, fontSize=20, spaceAfter=6,
        )
        story.append(Paragraph("Algorithmic Fairness Audit Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ParagraphStyle("Sub", parent=styles["Normal"],
                           textColor=GRAY, fontSize=10),
        ))
        story.append(Paragraph(
            "German Credit Dataset | B.Tech CSE AI/ML Project",
            ParagraphStyle("Sub2", parent=styles["Normal"],
                           textColor=GRAY, fontSize=10),
        ))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#1E293B"), spaceAfter=12))

        # Section: Overview
        h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                             textColor=INDIGO, fontSize=13, spaceAfter=4)
        body = ParagraphStyle("Body", parent=styles["Normal"],
                               fontSize=10, textColor=colors.HexColor("#1f2328"),
                               spaceAfter=6)

        story.append(Paragraph("1. Project Overview", h2))
        story.append(Paragraph(
            "This report summarises the results of an algorithmic fairness audit "
            "conducted using the German Credit Dataset. Three ML models were trained "
            "and evaluated using four fairness metrics: Demographic Parity, "
            "Equalized Odds, Equal Opportunity, and Predictive Parity. Three bias "
            "mitigation strategies (pre-processing, in-processing, post-processing) "
            "were applied and compared.", body))

        # Section: Performance
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("2. Model Performance Results", h2))
        if perf_rows:
            tbl_data = [["Model","Type","Accuracy","Precision","Recall","F1"]]
            for r in perf_rows:
                tbl_data.append([
                    r["Model"], r["Type"],
                    f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}",
                    f"{r['Recall']:.4f}",   f"{r['F1']:.4f}",
                ])
            tbl = Table(tbl_data, hAlign="LEFT",
                        colWidths=[4*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0),  INDIGO),
                ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
                ("FONTSIZE",    (0,0), (-1,-1), 9),
                ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f7f8fa"),WHITE]),
            ]))
            story.append(tbl)
        else:
            story.append(Paragraph("No performance data available.", body))

        # Section: Fairness
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("3. Fairness Audit Results", h2))
        if fair_rows:
            tbl_data2 = [["Model","DPD","EOdds","EOpp","PP"]]
            for r in fair_rows:
                def _fmt(v): return f"{v:.4f}" if v is not None else "N/A"
                tbl_data2.append([
                    r["Model"],
                    _fmt(r["Dem. Parity Diff"]),
                    _fmt(r["Eq. Odds Diff"]),
                    _fmt(r["Eq. Opp. Diff"]),
                    _fmt(r["Pred. Parity Diff"]),
                ])
            tbl2 = Table(tbl_data2, hAlign="LEFT",
                         colWidths=[5*cm, 3*cm, 3*cm, 3*cm, 3*cm])
            tbl2.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0),  INDIGO),
                ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
                ("FONTSIZE",    (0,0), (-1,-1), 9),
                ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f7f8fa"),WHITE]),
            ]))
            story.append(tbl2)
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(
                "Note: DPD = Demographic Parity Difference, EOdds = Equalized Odds "
                "Difference, EOpp = Equal Opportunity Difference, PP = Predictive "
                "Parity Difference. Values below 0.10 are considered to satisfy the "
                "fairness threshold.", body))
        else:
            story.append(Paragraph("No fairness audit data available.", body))

        # Section: Disclaimer
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("4. Limitations and Disclaimer", h2))
        story.append(Paragraph(
            "This toolkit uses the German Credit Dataset (UCI, 1994). Results reflect "
            "historical patterns in that dataset and should not be generalised to other "
            "contexts without further validation. No model presented here should be "
            "considered completely unbiased. Different fairness metrics can "
            "mathematically conflict. The choice of fairness metric and threshold "
            "depends on the deployment context and legal requirements.", body))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return None


if st.button("📝 Generate PDF Report", type="primary"):
    with st.spinner("Generating PDF..."):
        pdf_bytes = _generate_pdf()
    if pdf_bytes:
        st.success("✅ PDF report generated!")
        st.download_button(
            label="⬇️ Download Full Audit Report (PDF)",
            data=pdf_bytes,
            file_name=f"fairness_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
    else:
        st.error(
            "PDF generation requires reportlab. Install it with:\n"
            "`pip install reportlab`\n\n"
            "Or download the CSV results above instead."
        )

st.markdown("---")

# ── 5. Findings and Recommendations ───────────────────────────────────────
st.markdown("### 4️⃣ Key Findings & Recommendations")

if fair_rows:
    st.markdown("""
    <div class='aft-card'>
      <div class='aft-card-title'>📌 FINDINGS (Based on Actual Results)</div>
    """, unsafe_allow_html=True)

    for r in fair_rows:
        dpd = r.get("Dem. Parity Diff")
        name = r["Model"]
        if dpd is not None:
            if dpd < 0.10:
                badge = "<span class='badge badge-success'>✓ DP Satisfied</span>"
            else:
                badge = "<span class='badge badge-warning'>⚠ DP Concern</span>"
            st.markdown(f"**{name}** — Demographic Parity Diff: `{dpd:.4f}` {badge}",
                        unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='aft-card' style='margin-top:12px;'>
      <div class='aft-card-title'>💡 RECOMMENDATIONS</div>
      <ul style='color:#94A3B8;font-size:0.85rem;margin:4px 0 0 0;'>
        <li>Use post-processing (ThresholdOptimizer) when the base model is already deployed
            and you need quick fairness improvement without retraining.</li>
        <li>Use in-processing (ExponentiatedGradient) when you can retrain the model and
            want a mathematically grounded fairness guarantee during training.</li>
        <li>Use pre-processing (Reweighing) as a first step when the training data itself
            is known to be unrepresentative.</li>
        <li>Always report both performance AND fairness metrics to stakeholders.</li>
        <li>Involve domain experts and affected communities in the final model selection.</li>
        <li>⚠️ No mitigation technique eliminates all bias — these reduce measurable
            statistical disparities on the chosen metric but do not address root causes.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Run the Fairness Audit to unlock findings and recommendations.")

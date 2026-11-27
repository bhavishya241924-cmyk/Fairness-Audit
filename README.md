# Algorithmic Fairness Audit & Bias Mitigation Toolkit

> A professional AI/ML application that audits machine learning models for
> demographic bias and applies state-of-the-art bias mitigation strategies.
> Built with Python, Streamlit, Scikit-learn, and Fairlearn.

---

## 🚀 Quick Start

### Step 1 — Install Python
Download Python 3.10 or later from https://www.python.org/downloads/
During installation, tick **"Add Python to PATH"**.

Verify:
```cmd
python --version
```

### Step 2 — Clone / Download the Project
Place the `Algorithmic_Fairness_Toolkit/` folder anywhere on your machine.

### Step 3 — Create a Virtual Environment
Open a terminal (PowerShell or CMD) inside the project folder:
```powershell
cd Algorithmic_Fairness_Toolkit
python -m venv venv
```

### Step 4 — Activate the Environment
```powershell
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat
```
You should see `(venv)` in your prompt.

### Step 5 — Install Dependencies
```powershell
pip install -r requirements.txt
```
This takes 2–5 minutes. The main packages installed:
- `streamlit` — web dashboard
- `scikit-learn` — ML models
- `fairlearn` — fairness metrics + mitigation
- `plotly` — interactive charts
- `reportlab` — PDF report generation

### Step 6 — Get the Dataset
The app can auto-download the dataset from within the **Dataset Explorer** page.

Alternatively, download it manually:
1. Go to: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
2. Save the file as `data/german_credit.csv` inside the project folder.

### Step 7 — Run the Application
```powershell
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`

---

## 📂 Project Structure

```
Algorithmic_Fairness_Toolkit/
│
├── app.py                      ← Main entry point (Dashboard Overview)
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── data/
│   └── german_credit.csv       ← Dataset (auto-downloaded or manual)
│
├── models/                     ← Saved model .pkl files (auto-created)
│   ├── Logistic_Regression.pkl
│   ├── Decision_Tree.pkl
│   └── Random_Forest.pkl
│
├── src/                        ← Backend logic (no UI code)
│   ├── __init__.py
│   ├── data_loader.py          ← Load & explore the dataset
│   ├── preprocessing.py        ← Cleaning, encoding, train/val/test split
│   ├── model_training.py       ← Train, evaluate, save ML models
│   ├── fairness_metrics.py     ← 4 fairness metrics implementation
│   ├── bias_mitigation.py      ← 3 bias mitigation techniques
│   └── visualization.py        ← Plotly chart builders
│
├── pages/                      ← Streamlit multi-page app pages
│   ├── 1_Dataset_Explorer.py
│   ├── 2_Model_Training.py
│   ├── 3_Fairness_Audit.py
│   ├── 4_Bias_Mitigation.py
│   ├── 5_Model_Comparison.py
│   ├── 6_Reports.py
│   └── 7_Settings.py
│
├── assets/
│   └── styles.css              ← Global dark-theme CSS
│
└── reports/                    ← Generated report output directory
```

---

## 🗺️ How to Use (Step-by-Step)

| Step | Page | Action |
|------|------|--------|
| 1 | Dataset Explorer | Click "Auto-Download Dataset" then "Load Dataset" |
| 2 | Model Training | Click "Run Preprocessing" then "Train Selected Models" |
| 3 | Fairness Audit | Click "Run Fairness Audit" |
| 4 | Bias Mitigation | Apply pre-, in-, and post-processing one by one |
| 5 | Model Comparison | View fairness-accuracy tradeoff charts |
| 6 | Reports | Generate and download PDF/CSV reports |

---

## 📊 Dataset: German Credit Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Samples | 1,000 applicants |
| Features | 20 (categorical + numerical) |
| Target | Credit Risk: Good (1) / Bad (0) |
| Sensitive attrs | Age (binary ≤25 / >25), Sex (derived) |
| Year | 1994, Germany |

**Limitations:**
- Historical data from 1994 Germany — may not reflect current patterns
- Sensitive attributes are proxies, not precise demographic identifiers
- Class imbalance (700 Good, 300 Bad) — use F1-score, not just accuracy

---

## ⚖️ Fairness Metrics

| Metric | Formula | Meaning |
|---|---|---|
| Demographic Parity | `\|P(Ŷ=1\|A=0) - P(Ŷ=1\|A=1)\|` | Equal approval rates |
| Equalized Odds | `max(\|TPR₀-TPR₁\|, \|FPR₀-FPR₁\|)` | Equal TPR and FPR |
| Equal Opportunity | `\|TPR₀ - TPR₁\|` | Equal recall for positive class |
| Predictive Parity | `\|Precision₀ - Precision₁\|` | Equal precision for positive preds |

Fairness threshold: **0.10** (10% difference). Values below this satisfy the threshold.

---

## 🛡️ Bias Mitigation

| Stage | Technique | Library | How it works |
|---|---|---|---|
| Pre-processing | Reweighing | Custom | Adjusts sample weights before training |
| In-processing | ExponentiatedGradient | Fairlearn | Fairness-constrained training |
| Post-processing | ThresholdOptimizer | Fairlearn | Per-group decision thresholds |

---

## 🛠️ Common Errors & Fixes

**Error: `ModuleNotFoundError: No module named 'fairlearn'`**
```
pip install fairlearn
```

**Error: `streamlit: command not found`**
```
pip install streamlit
# or use:
python -m streamlit run app.py
```

**Error: `Dataset not found`**
- Click "Auto-Download Dataset" in the Dataset Explorer page
- Or download manually from UCI and save as `data/german_credit.csv`

**Error: PowerShell script execution policy**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**App is slow on first load:**
- First run installs models in memory — subsequent runs are faster
- Random Forest with 100 trees can take 10–15 seconds to train

---

## 📋 Requirements

```
streamlit==1.32.0
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.1
fairlearn==0.10.0
matplotlib==3.8.3
seaborn==0.13.2
plotly==5.20.0
joblib==1.3.2
reportlab==4.1.0
```

---

## 🎓 Viva Questions & Answers

**Q: What is algorithmic fairness?**
A: Algorithmic fairness means ensuring ML model decisions do not systematically
disadvantage individuals based on demographic characteristics like age, sex, or race.

**Q: Why can't a model satisfy all fairness metrics simultaneously?**
A: Because they are mathematically incompatible under certain conditions
(Chouldechova 2017, Kleinberg et al. 2016). For example, satisfying equalized odds
simultaneously with predictive parity is impossible unless the base rates are equal.

**Q: What is the difference between pre-, in-, and post-processing mitigation?**
A: Pre-processing modifies training data; in-processing adds constraints during training;
post-processing adjusts decisions after the model is trained.

**Q: Why do you use a validation set for post-processing?**
A: To prevent data leakage. The ThresholdOptimizer must not "see" the test set
during threshold tuning — the test set is kept for final unbiased evaluation only.

**Q: What is Reweighing?**
A: A technique that assigns higher sample weights to underrepresented
(group, label) combinations using the formula: w = P(A)×P(Y) / P(A,Y).

**Q: What is the ExponentiatedGradient?**
A: A Fairlearn algorithm that solves a Lagrangian relaxation of a constrained
optimisation problem, iteratively updating penalty multipliers until the fairness
constraint (e.g., Demographic Parity) is satisfied within tolerance epsilon.

**Q: How do you prevent data leakage in this project?**
A: (1) The scaler is fit on training data only. (2) Sensitive attributes are separated
before encoding. (3) ThresholdOptimizer is tuned on the validation set, not the test set.
(4) All splits use stratification to maintain class distribution.

---

## 👨‍💻 Author

B.Tech CSE AI/ML — University Project
Algorithmic Fairness Audit & Bias Mitigation Toolkit

# ============================================================
# HEART DISEASE RISK PREDICTION - STREAMLIT WEB APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, roc_auc_score,
                             confusion_matrix, accuracy_score)

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e74c3c, #ff8a80);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #8b9cb3;
        margin-top: 4px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #212435);
        border: 1px solid #2d3147;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 700;
        color: #e74c3c;
        margin: 0;
    }
    .metric-card p {
        color: #8b9cb3;
        font-size: 0.85rem;
        margin: 4px 0 0 0;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        border-left: 4px solid #e74c3c;
        padding-left: 12px;
        margin: 2rem 0 1rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #c0392b22, #e74c3c22);
        border: 1px solid #e74c3c;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #1e8449 22, #2ecc7122);
        border: 1px solid #2ecc71;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .insight-box {
        background: #1a1d2e;
        border-left: 3px solid #e74c3c;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        color: #cdd6f4;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/logistic_regression.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("models/random_forest.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("models/gradient_boosting.pkl", "rb") as f:
        gb = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return lr, rf, gb, scaler

@st.cache_data
def load_data():
    df       = pd.read_csv("data/heart_cleaned.csv")
    X_train  = pd.read_csv("data/X_train.csv")
    X_test   = pd.read_csv("data/X_test.csv")
    y_train  = pd.read_csv("data/y_train.csv").squeeze()
    y_test   = pd.read_csv("data/y_test.csv").squeeze()
    results  = pd.read_csv("models/results.csv")
    return df, X_train, X_test, y_train, y_test, results

lr, rf, gb, scaler = load_models()
df, X_train, X_test, y_train, y_test, results = load_data()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Navigation")
    page = st.radio("", ["🏠 Overview", "📊 EDA & Insights",
                         "🤖 Model Results", "🔮 Risk Predictor"])
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"- {len(df)} patients")
    st.markdown(f"- {df.shape[1]-1} features")
    st.markdown(f"- UCI Heart Disease Dataset")
    st.markdown("---")
    st.markdown("**Models Trained**")
    st.markdown("- Logistic Regression")
    st.markdown("- Random Forest")
    st.markdown("- Gradient Boosting")

# ══════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="hero-title">🫀 Heart Disease Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Machine Learning · UCI Dataset · 3 Models Compared</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>303</h2><p>Total Patients</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>88.5%</h2><p>Best Accuracy</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>0.962</h2><p>Best ROC-AUC</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>3</h2><p>Models Trained</p></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Project Pipeline</p>', unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Data Collection", "UCI Heart Disease Dataset (Cleveland)"),
        ("2️⃣", "EDA", "Explored distributions, correlations, and patterns"),
        ("3️⃣", "Data Cleaning", "Handled missing values, removed duplicates"),
        ("4️⃣", "Preprocessing", "Encoding, scaling, 80/20 train-test split"),
        ("5️⃣", "Modeling", "Logistic Regression, Random Forest, Gradient Boosting"),
        ("6️⃣", "Evaluation", "Accuracy, Precision, Recall, F1, ROC-AUC"),
        ("7️⃣", "Interpretation", "Feature importance and risk factor analysis"),
    ]
    for icon, title, desc in steps:
        st.markdown(f'<div class="insight-box">{icon} <strong>{title}</strong> — {desc}</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Key Findings</p>', unsafe_allow_html=True)
    findings = [
        "🔴 thal_7.0 (reversible thalassemia defect) is the strongest predictor of heart disease",
        "🔴 cp_4.0 (asymptomatic chest pain) is the 2nd most important feature",
        "🔵 thalach (high max heart rate) significantly reduces heart disease risk",
        "🔴 Males are at significantly higher risk than females in this dataset",
        "🔴 Higher oldpeak (ST depression) correlates strongly with disease presence",
        "🏆 Logistic Regression achieved the best ROC-AUC of 0.962",
    ]
    for f in findings:
        st.markdown(f'<div class="insight-box">{f}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 2: EDA & INSIGHTS
# ══════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.markdown('<p class="hero-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Target Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')
        counts = df["target"].value_counts()
        bars = ax.bar(["No Disease", "Disease"], counts.values,
                      color=["#2ecc71", "#e74c3c"], edgecolor="none", width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(int(bar.get_height())), ha='center', color='white', fontsize=11)
        ax.set_ylabel("Count", color="#8b9cb3")
        ax.tick_params(colors="#8b9cb3")
        ax.spines[:].set_visible(False)
        st.pyplot(fig)

    with col2:
        st.markdown('<p class="section-header">Age Distribution by Disease</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')
        ax.hist(df[df["target"]==0]["age"], bins=15, alpha=0.7,
                color="#2ecc71", label="No Disease")
        ax.hist(df[df["target"]==1]["age"], bins=15, alpha=0.7,
                color="#e74c3c", label="Disease")
        ax.legend(facecolor="#1a1d2e", labelcolor="white")
        ax.set_xlabel("Age", color="#8b9cb3")
        ax.tick_params(colors="#8b9cb3")
        ax.spines[:].set_visible(False)
        st.pyplot(fig)

    st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.3, linecolor="#0f1117")
    ax.tick_params(colors="white")
    st.pyplot(fig)

    st.markdown('<p class="section-header">Key Feature Boxplots</p>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#0f1117')
    feats  = ["age", "thalach", "oldpeak"]
    titles = ["Age", "Max Heart Rate", "ST Depression (oldpeak)"]
    for ax, feat, title in zip(axes, feats, titles):
        ax.set_facecolor('#1a1d2e')
        data_no  = df[df["target"]==0][feat]
        data_yes = df[df["target"]==1][feat]
        ax.boxplot([data_no, data_yes], patch_artist=True,
                   boxprops=dict(facecolor="#1a1d2e"),
                   medianprops=dict(color="#e74c3c", linewidth=2),
                   whiskerprops=dict(color="#8b9cb3"),
                   capprops=dict(color="#8b9cb3"),
                   flierprops=dict(markerfacecolor="#e74c3c", marker='o'))
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No Disease", "Disease"], color="#8b9cb3")
        ax.set_title(title, color="white")
        ax.tick_params(colors="#8b9cb3")
        ax.spines[:].set_color("#2d3147")
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# PAGE 3: MODEL RESULTS
# ══════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.markdown('<p class="hero-title">🤖 Model Performance</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
    st.dataframe(results.style.highlight_max(
        subset=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
        color="#e74c3c33"), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Accuracy Comparison</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')
        colors = ["#e74c3c", "#2ecc71", "#3498db"]
        bars = ax.bar(results["Model"], results["Accuracy"]*100,
                      color=colors, edgecolor="none", width=0.5)
        for bar, val in zip(bars, results["Accuracy"]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.2,
                    f"{val*100:.1f}%", ha='center', color='white', fontsize=10)
        ax.set_ylim(70, 100)
        ax.set_ylabel("Accuracy (%)", color="#8b9cb3")
        ax.tick_params(colors="#8b9cb3", axis='y')
        ax.tick_params(colors="white", axis='x')
        ax.spines[:].set_visible(False)
        plt.xticks(rotation=10)
        st.pyplot(fig)

    with col2:
        st.markdown('<p class="section-header">ROC Curves</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#1a1d2e')
        for model, name, color in [(lr,"Logistic Reg","#e74c3c"),
                                    (rf,"Random Forest","#2ecc71"),
                                    (gb,"Grad. Boosting","#3498db")]:
            prob = model.predict_proba(X_test)[:, 1]
            auc  = roc_auc_score(y_test, prob)
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} ({auc:.2f})")
        ax.plot([0,1],[0,1],"--", color="#4a4e6a", lw=1)
        ax.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=8)
        ax.set_xlabel("False Positive Rate", color="#8b9cb3")
        ax.set_ylabel("True Positive Rate", color="#8b9cb3")
        ax.tick_params(colors="#8b9cb3")
        ax.spines[:].set_color("#2d3147")
        st.pyplot(fig)

    st.markdown('<p class="section-header">Feature Importance (Random Forest)</p>', unsafe_allow_html=True)
    rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns)
    rf_imp = rf_imp.sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')
    ax.barh(rf_imp.index, rf_imp.values, color="#2ecc71", edgecolor="none")
    ax.set_xlabel("Importance Score", color="#8b9cb3")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2d3147")
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# PAGE 4: RISK PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "🔮 Risk Predictor":
    st.markdown('<p class="hero-title">🔮 Heart Disease Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Enter patient details below to predict heart disease risk</p>', unsafe_allow_html=True)
    st.markdown("---")

    model_choice = st.selectbox("Select Model",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"])
    model_map = {"Logistic Regression": lr,
                 "Random Forest": rf,
                 "Gradient Boosting": gb}
    selected_model = model_map[model_choice]

    st.markdown('<p class="section-header">Patient Information</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.slider("Age", 20, 80, 50)
        sex      = st.selectbox("Sex", ["Female (0)", "Male (1)"])
        cp       = st.selectbox("Chest Pain Type",
                    ["Typical Angina (1)", "Atypical Angina (2)",
                     "Non-anginal Pain (3)", "Asymptomatic (4)"])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
        chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 240)

    with col2:
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
        restecg = st.selectbox("Resting ECG",
                    ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"])
        thalach = st.slider("Max Heart Rate Achieved", 60, 210, 150)
        exang   = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)

    with col3:
        slope = st.selectbox("Slope of ST Segment",
                    ["Upsloping (1)", "Flat (2)", "Downsloping (3)"])
        ca    = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
        thal  = st.selectbox("Thalassemia",
                    ["Normal (3)", "Fixed Defect (6)", "Reversible Defect (7)"])

    st.markdown("")
    predict_btn = st.button("🔮 Predict Risk")

    if predict_btn:
        # Parse inputs
        sex_val     = int(sex.split("(")[1][0])
        cp_val      = int(cp.split("(")[1][0])
        fbs_val     = int(fbs.split("(")[1][0])
        restecg_val = int(restecg.split("(")[1][0])
        exang_val   = int(exang.split("(")[1][0])
        slope_val   = int(slope.split("(")[1][0])
        thal_val    = int(thal.split("(")[1][0])
        ca_val      = int(ca)

        # Build input using exact training column names (format: col_value.0)
        raw_enc = pd.DataFrame(0.0, index=[0], columns=X_train.columns)

        # Numerical columns
        raw_enc["age"]      = float(age)
        raw_enc["trestbps"] = float(trestbps)
        raw_enc["chol"]     = float(chol)
        raw_enc["thalach"]  = float(thalach)
        raw_enc["oldpeak"]  = float(oldpeak)

        # One-hot columns - training uses format like "sex_1.0", "cp_4.0"
        def set_col(prefix, val):
            col = f"{prefix}_{float(val)}"
            if col in raw_enc.columns:
                raw_enc[col] = 1.0

        set_col("sex",     sex_val)
        set_col("cp",      cp_val)
        set_col("fbs",     fbs_val)
        set_col("restecg", restecg_val)
        set_col("exang",   exang_val)
        set_col("slope",   slope_val)
        set_col("ca",      ca_val)
        set_col("thal",    thal_val)

        # Scale numerical
        num_cols = ["age","trestbps","chol","thalach","oldpeak"]
        raw_enc[num_cols] = scaler.transform(raw_enc[num_cols])

        # Predict
        pred      = selected_model.predict(raw_enc)[0]
        prob      = selected_model.predict_proba(raw_enc)[0][1]
        risk_pct  = prob * 100

        st.markdown("---")
        st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)

        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h1 style="color:#e74c3c; font-size:3rem;">⚠️</h1>
                    <h2 style="color:#e74c3c;">High Risk</h2>
                    <p style="color:#cdd6f4; font-size:1.2rem;">
                        This patient has a <strong style="color:#e74c3c;">{risk_pct:.1f}%</strong>
                        probability of heart disease.
                    </p>
                    <p style="color:#8b9cb3; font-size:0.85rem;">
                        ⚕️ Please consult a cardiologist immediately.
                    </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h1 style="color:#2ecc71; font-size:3rem;">✅</h1>
                    <h2 style="color:#2ecc71;">Low Risk</h2>
                    <p style="color:#cdd6f4; font-size:1.2rem;">
                        This patient has a <strong style="color:#2ecc71;">{risk_pct:.1f}%</strong>
                        probability of heart disease.
                    </p>
                    <p style="color:#8b9cb3; font-size:0.85rem;">
                        ✅ Continue regular health checkups.
                    </p>
                </div>""", unsafe_allow_html=True)

        with col_res2:
            # Risk gauge
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#0f1117')
            theta = np.linspace(0, np.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta), color="#2d3147", lw=10)
            end_angle = np.pi * (1 - prob)
            theta2 = np.linspace(np.pi, end_angle, 100)
            color = "#e74c3c" if prob > 0.5 else "#2ecc71"
            ax.plot(np.cos(theta2), np.sin(theta2), color=color, lw=10)
            ax.text(0, 0.1, f"{risk_pct:.1f}%", ha='center', va='center',
                    fontsize=24, fontweight='bold', color=color)
            ax.text(0, -0.25, "Risk Score", ha='center', color="#8b9cb3", fontsize=11)
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-0.5, 1.3)
            ax.axis('off')
            st.pyplot(fig)

        st.markdown(f"""
        <div class="insight-box" style="margin-top:1rem;">
            🤖 <strong>Model Used:</strong> {model_choice} &nbsp;|&nbsp;
            📊 <strong>Confidence:</strong> {max(prob, 1-prob)*100:.1f}%
        </div>""", unsafe_allow_html=True)

        st.warning("⚠️ This tool is for educational purposes only. Always consult a medical professional for actual diagnosis.")
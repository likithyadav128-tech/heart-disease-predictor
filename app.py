# ============================================================
# HEART DISEASE RISK PREDICTION - STREAMLIT WEB APP v3 ULTRA
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HeartSense AI · Clinical Risk Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body { font-family: 'Outfit', sans-serif; box-sizing: border-box; }

/* ── GLOBAL ── */
.stApp {
    background: #060810;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(220,38,38,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(59,130,246,0.08) 0%, transparent 60%);
    min-height: 100vh;
}
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1280px; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(8,10,20,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebar"] .stRadio label {
    color: rgba(255,255,255,0.4) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
}
[data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {
    font-size: 0.9rem !important;
    color: rgba(255,255,255,0.7) !important;
    font-weight: 400;
}

/* ── INPUTS ── */
.stSelectbox label, .stSlider label {
    color: rgba(255,255,255,0.4) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
}
.stSelectbox [data-baseweb="select"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
.stSelectbox [data-baseweb="select"]:hover {
    border-color: rgba(220,38,38,0.4) !important;
}
.stSelectbox [data-baseweb="select"] * { color: rgba(255,255,255,0.85) !important; background: #0d0f1a !important; }
[data-baseweb="popover"] { background: #0d0f1a !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; }
[data-baseweb="popover"] li { color: rgba(255,255,255,0.7) !important; }
[data-baseweb="popover"] li:hover { background: rgba(220,38,38,0.15) !important; }
.stSlider [data-baseweb="slider"] { color: #dc2626 !important; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    box-shadow: 0 0 30px rgba(220,38,38,0.3), 0 4px 20px rgba(0,0,0,0.4) !important;
    transition: all 0.3s cubic-bezier(0.34,1.56,0.64,1) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.01) !important;
    box-shadow: 0 0 50px rgba(220,38,38,0.5), 0 8px 30px rgba(0,0,0,0.5) !important;
}

/* ── HERO ── */
.hero-wrap {
    position: relative;
    padding: 3.5rem 0 2rem;
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #dc2626;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px; height: 2px;
    background: #dc2626;
    border-radius: 2px;
}
.hero-h1 {
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 900;
    line-height: 1.08;
    color: #fff;
    margin: 0 0 1.2rem;
    letter-spacing: -0.03em;
}
.hero-h1 .grad {
    background: linear-gradient(135deg, #dc2626, #f87171, #fca5a5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.45);
    line-height: 1.75;
    max-width: 540px;
    font-weight: 300;
}

/* ── GLASS CARDS ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s, transform 0.3s;
}
.glass-card:hover {
    border-color: rgba(220,38,38,0.25);
    transform: translateY(-2px);
}
.glass-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(220,38,38,0.04) 0%, transparent 60%);
    pointer-events: none;
}

/* ── STAT CARDS ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2.5rem;
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s;
}
.stat-card:hover { border-color: rgba(220,38,38,0.3); transform: translateY(-3px); }
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(220,38,38,0.6), transparent);
}
.stat-card .num {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.stat-card .num.red { color: #dc2626; }
.stat-card .num.white { color: #fff; }
.stat-card .lbl {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
}
.stat-card .icon {
    position: absolute;
    top: 1.2rem; right: 1.4rem;
    font-size: 1.6rem;
    opacity: 0.3;
}

/* ── SECTION HEADERS ── */
.sec-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #dc2626;
    margin-bottom: 0.5rem;
}
.sec-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.02em;
    margin: 0 0 1.4rem;
}

/* ── PIPELINE STEPS ── */
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 0.6rem;
    transition: all 0.2s;
}
.pipeline-step:hover {
    background: rgba(220,38,38,0.05);
    border-color: rgba(220,38,38,0.2);
}
.step-num {
    min-width: 36px; height: 36px;
    background: rgba(220,38,38,0.15);
    border: 1px solid rgba(220,38,38,0.3);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 800;
    color: #dc2626;
    font-family: 'JetBrains Mono', monospace;
}
.step-title { font-size: 0.92rem; font-weight: 600; color: rgba(255,255,255,0.9); margin-bottom: 2px; }
.step-desc { font-size: 0.78rem; color: rgba(255,255,255,0.35); font-weight: 300; }

/* ── FINDING PILLS ── */
.finding {
    display: flex; align-items: flex-start; gap: 0.6rem;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    color: rgba(255,255,255,0.65);
    transition: background 0.2s;
}
.finding:hover { background: rgba(255,255,255,0.04); }
.f-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; margin-top: 5px; }

/* ── SIDEBAR STATS ── */
.sb-stat {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.55rem 0.9rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 0.45rem;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.4);
}
.sb-stat-val { color: #dc2626 !important; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }

/* ── RESULT CARDS ── */
.result-card-high {
    background: linear-gradient(135deg, rgba(220,38,38,0.12), rgba(153,27,27,0.06));
    border: 1px solid rgba(220,38,38,0.35);
    border-radius: 22px;
    padding: 2.2rem 1.8rem;
    text-align: center;
    box-shadow: 0 0 60px rgba(220,38,38,0.12), inset 0 1px 0 rgba(255,255,255,0.06);
}
.result-card-low {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(21,128,61,0.05));
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 22px;
    padding: 2.2rem 1.8rem;
    text-align: center;
    box-shadow: 0 0 60px rgba(34,197,94,0.08), inset 0 1px 0 rgba(255,255,255,0.06);
}
.res-icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
.res-label-h { font-size: 2.2rem; font-weight: 900; color: #dc2626; letter-spacing: -0.03em; margin-bottom: 0.4rem; }
.res-label-l { font-size: 2.2rem; font-weight: 900; color: #22c55e; letter-spacing: -0.03em; margin-bottom: 0.4rem; }
.res-prob { font-size: 0.9rem; color: rgba(255,255,255,0.4); }
.res-prob strong { color: rgba(255,255,255,0.85); }

/* ── INPUT GROUP HEADERS ── */
.inp-group {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: rgba(220,38,38,0.8);
    border-bottom: 1px solid rgba(220,38,38,0.15);
    padding-bottom: 0.5rem; margin: 1.4rem 0 1rem;
}

/* ── CONFIDENCE BAR ── */
.conf-wrap { margin-top: 1rem; }
.conf-label { font-size: 0.72rem; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.conf-bar-bg { background: rgba(255,255,255,0.06); border-radius: 100px; height: 6px; overflow: hidden; }
.conf-bar-fill-h { height: 6px; border-radius: 100px; background: linear-gradient(90deg, #991b1b, #dc2626, #f87171); }
.conf-bar-fill-l { height: 6px; border-radius: 100px; background: linear-gradient(90deg, #166534, #22c55e, #86efac); }
.conf-pct { font-size: 0.85rem; font-weight: 700; color: rgba(255,255,255,0.7); margin-top: 0.3rem; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS & DATA ──────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/logistic_regression.pkl","rb") as f: lr = pickle.load(f)
    with open("models/random_forest.pkl","rb") as f:       rf = pickle.load(f)
    with open("models/gradient_boosting.pkl","rb") as f:   gb = pickle.load(f)
    with open("models/scaler.pkl","rb") as f:              sc = pickle.load(f)
    return lr, rf, gb, sc

@st.cache_data
def load_data():
    df      = pd.read_csv("data/heart_cleaned.csv")
    X_train = pd.read_csv("data/X_train.csv")
    X_test  = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/y_test.csv").squeeze()
    results = pd.read_csv("models/results.csv")
    return df, X_train, X_test, y_train, y_test, results

lr, rf, gb, scaler = load_models()
df, X_train, X_test, y_train, y_test, results = load_data()

PALETTE = {"bg":"#060810","card":"#0d0f1a","red":"#dc2626","green":"#22c55e","blue":"#3b82f6","border":"#1a1d2e","text":"#e2e8f0","muted":"#64748b"}

def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.grid(True, color=PALETTE["border"], linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    return fig, ax

# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="font-size:1.4rem;font-weight:900;color:#fff;letter-spacing:-0.03em;">
            Heart<span style="color:#dc2626;">Sense</span> <span style="font-size:0.75rem;font-weight:400;color:rgba(255,255,255,0.3);vertical-align:middle;">AI</span>
        </div>
        <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.15em;text-transform:uppercase;color:rgba(255,255,255,0.25);margin-top:3px;">Clinical Risk Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  Data Insights",
        "🤖  Model Analytics",
        "🔮  Risk Assessment"
    ])

    st.markdown("<div style='height:1px;background:rgba(255,255,255,0.05);margin:1.2rem 0;'></div>", unsafe_allow_html=True)
    for lbl, val in [("Patients","303"),("Features","13"),("Best AUC","0.962"),("Best Acc.","88.5%"),("Models","3")]:
        st.markdown(f'<div class="sb-stat">{lbl}<span class="sb-stat-val">{val}</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem;padding:1rem;border-radius:12px;background:rgba(220,38,38,0.06);border:1px solid rgba(220,38,38,0.15);">
        <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:rgba(220,38,38,0.7);margin-bottom:0.4rem;">Disclaimer</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.3);line-height:1.6;">Educational use only. Not a substitute for medical advice.</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">Machine Learning · Healthcare AI · UCI Dataset</div>
        <h1 class="hero-h1">Detect heart disease risk<br>with <span class="grad">clinical AI precision</span></h1>
        <p class="hero-sub">A complete ML pipeline comparing three algorithms on 303 patient records — with live risk prediction, interactive insights, and explainable results.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card"><span class="icon">📈</span><div class="num red">0.962</div><div class="lbl">Best ROC-AUC score</div></div>
        <div class="stat-card"><span class="icon">🎯</span><div class="num white">88.5<span style="font-size:1.4rem;color:rgba(255,255,255,0.3)">%</span></div><div class="lbl">Top model accuracy</div></div>
        <div class="stat-card"><span class="icon">🤖</span><div class="num red">3</div><div class="lbl">Models compared</div></div>
        <div class="stat-card"><span class="icon">🏥</span><div class="num white">303</div><div class="lbl">Patient records</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.05, 0.95], gap="large")

    with col1:
        st.markdown('<div class="sec-label">Project pipeline</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("01","Data collection","UCI Heart Disease Dataset — Cleveland subset, 303 records"),
            ("02","Exploratory analysis","Distributions, correlations, feature vs outcome patterns"),
            ("03","Data cleaning","Median imputation for '?' values, duplicate removal"),
            ("04","Preprocessing","One-hot encoding × 8 cols, standard scaling × 5 cols, 80/20 split"),
            ("05","Model training","Logistic Regression · Random Forest (n=100) · Gradient Boosting"),
            ("06","Evaluation","Accuracy · Precision · Recall · F1 · ROC-AUC · Confusion Matrix"),
            ("07","Deployment","Streamlit Cloud — 4-page interactive web application"),
        ]:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num">{num}</div>
                <div><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sec-label">Key findings</div>', unsafe_allow_html=True)
        for dot_col, text in [
            ("#dc2626","thal_7.0 (reversible defect) is the strongest predictor across all models"),
            ("#dc2626","Asymptomatic chest pain (cp=4) strongly correlates with disease presence"),
            ("#22c55e","Higher max heart rate (thalach) significantly reduces disease probability"),
            ("#dc2626","Male patients show substantially higher disease rates than female patients"),
            ("#dc2626","ST depression > 2.0 (oldpeak) is a reliable high-risk indicator"),
            ("#3b82f6","Logistic Regression leads with best ROC-AUC of 0.962 on test set"),
            ("#3b82f6","Random Forest achieves best accuracy at 88.5% with 28/33 correct"),
            ("#22c55e","All 3 models exceed 85% accuracy — strong generalisation on unseen data"),
        ]:
            st.markdown(f'<div class="finding"><div class="f-dot" style="background:{dot_col};"></div>{text}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1.5rem;padding:1.2rem 1.4rem;border-radius:16px;background:rgba(220,38,38,0.06);border:1px solid rgba(220,38,38,0.15);">
            <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;color:rgba(220,38,38,0.7);margin-bottom:0.8rem;">Model leaderboard</div>
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:rgba(255,255,255,0.3);margin-bottom:0.5rem;padding:0 0.3rem;">
                <span>Model</span><span>AUC</span><span>Acc.</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#fff;font-weight:600;padding:0.5rem 0.3rem;border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>🥇 Logistic Reg.</span><span style="color:#dc2626;">0.962</span><span>86.9%</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:rgba(255,255,255,0.7);padding:0.5rem 0.3rem;border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>🥈 Random Forest</span><span>0.951</span><span style="color:#22c55e;">88.5%</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:rgba(255,255,255,0.5);padding:0.5rem 0.3rem;">
                <span>🥉 Grad. Boosting</span><span>0.932</span><span>86.9%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════
elif "Data" in page:
    st.markdown('<div class="sec-label">Exploratory data analysis</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sec-title">Understanding the data</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        fig, ax = dark_fig(5, 3.8)
        no_d, has_d = (df["target"]==0).sum(), (df["target"]==1).sum()
        bars = ax.bar(["No Disease","Disease"], [no_d, has_d],
                      color=[PALETTE["green"], PALETTE["red"]], width=0.45, edgecolor="none",
                      linewidth=0)
        for bar, val, c in zip(bars, [no_d, has_d], ["#22c55e","#dc2626"]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    str(val), ha="center", color=c, fontsize=13, fontweight="800")
        ax.set_title("Target distribution", color="#e2e8f0", fontsize=11, fontweight="700", pad=10)
        ax.set_ylim(0, 200)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = dark_fig(5, 3.8)
        ax.hist(df[df["target"]==0]["age"], bins=20, alpha=0.8, color=PALETTE["green"],
                label="No disease", edgecolor="none")
        ax.hist(df[df["target"]==1]["age"], bins=20, alpha=0.8, color=PALETTE["red"],
                label="Disease", edgecolor="none")
        ax.set_title("Age distribution by outcome", color="#e2e8f0", fontsize=11, fontweight="700", pad=10)
        ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text"], fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    fig, ax = dark_fig(12, 5)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(0, 130, s=85, l=40, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                ax=ax, linewidths=0.4, linecolor=PALETTE["bg"],
                annot_kws={"size":7.5, "color":"#e2e8f0", "weight":"500"},
                cbar_kws={"shrink":0.55})
    ax.set_title("Feature correlation heatmap", color="#e2e8f0", fontsize=12, fontweight="700", pad=12)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.set_facecolor(PALETTE["card"])
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Clinical features by outcome", color=PALETTE["muted"], fontsize=10, y=1.02)
    for ax, feat, label in zip(axes,
        ["age","thalach","oldpeak","chol"],
        ["Age (years)","Max heart rate","ST depression","Cholesterol (mg/dl)"]):
        ax.set_facecolor(PALETTE["card"])
        for sp in ax.spines.values(): sp.set_visible(False)
        d0 = df[df["target"]==0][feat]
        d1 = df[df["target"]==1][feat]
        parts = ax.violinplot([d0, d1], positions=[0,1], showmedians=True, showextrema=False)
        colors_v = [PALETTE["green"], PALETTE["red"]]
        for body, c in zip(parts["bodies"], colors_v):
            body.set_facecolor(c); body.set_alpha(0.55); body.set_edgecolor("none")
        parts["cmedians"].set_color("#fff"); parts["cmedians"].set_linewidth(2)
        ax.scatter([0]*len(d0), d0, alpha=0.15, color=PALETTE["green"], s=6, zorder=5)
        ax.scatter([1]*len(d1), d1, alpha=0.15, color=PALETTE["red"],   s=6, zorder=5)
        ax.set_xticks([0,1]); ax.set_xticklabels(["No","Yes"], color=PALETTE["muted"], fontsize=9)
        ax.tick_params(colors=PALETTE["muted"], labelsize=8)
        ax.set_title(label, fontsize=9.5, fontweight="700", color="#e2e8f0", pad=8)
        ax.yaxis.grid(True, color=PALETTE["border"], linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 3 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown('<div class="sec-label">Comparative analysis</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sec-title">Model performance</h2>', unsafe_allow_html=True)

    # Metrics table
    r = results.set_index("Model")
    st.dataframe(
        r.style.format("{:.4f}")
               .highlight_max(color="rgba(34,197,94,0.15)", axis=0)
               .set_properties(**{"color":"#e2e8f0","background-color":PALETTE["card"],"font-size":"0.88rem"}),
        use_container_width=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        fig, ax = dark_fig(5.5, 4)
        models_n = results["Model"].tolist()
        accs     = (results["Accuracy"]*100).tolist()
        colors_b = [PALETTE["red"], PALETTE["green"], PALETTE["blue"]]
        bars     = ax.barh(models_n, accs, color=colors_b, height=0.42, edgecolor="none")
        for bar, val in zip(bars, accs):
            ax.text(bar.get_width()-0.5, bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}%", va="center", ha="right",
                    color="#060810", fontsize=11, fontweight="800")
        ax.set_xlim(78, 93); ax.set_title("Accuracy comparison", color="#e2e8f0", fontsize=11, fontweight="700", pad=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = dark_fig(5.5, 4)
        for model, name, color, ls in [
            (lr,"Logistic Reg.",PALETTE["red"],"-"),
            (rf,"Random Forest",PALETTE["green"],"-"),
            (gb,"Grad. Boosting",PALETTE["blue"],"-")]:
            prob = model.predict_proba(X_test)[:,1]
            auc  = roc_auc_score(y_test, prob)
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name}  AUC={auc:.3f}")
            ax.fill_between(fpr, 0, tpr, alpha=0.04, color=color)
        ax.plot([0,1],[0,1],"--",color=PALETTE["border"],lw=1.5)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves", color="#e2e8f0", fontsize=11, fontweight="700", pad=10)
        ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text"], fontsize=8.5)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="large")

    with col3:
        fig, ax = dark_fig(5.5, 5)
        rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values().tail(14)
        cmap_b = [PALETTE["red"] if v > rf_imp.quantile(0.7) else PALETTE["blue"] for v in rf_imp]
        ax.barh(rf_imp.index, rf_imp.values, color=cmap_b, height=0.6, edgecolor="none")
        ax.set_title("Random Forest — feature importance", color="#e2e8f0", fontsize=10, fontweight="700", pad=10)
        ax.tick_params(labelsize=7.5)
        red_p  = mpatches.Patch(color=PALETTE["red"],  label="High importance")
        blue_p = mpatches.Patch(color=PALETTE["blue"], label="Moderate")
        ax.legend(handles=[red_p, blue_p], facecolor=PALETTE["card"],
                  edgecolor=PALETTE["border"], labelcolor=PALETTE["text"], fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        fig, ax = dark_fig(5.5, 5)
        y_pred = rf.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap=sns.light_palette("#dc2626", as_cmap=True),
                    linewidths=2, linecolor=PALETTE["bg"],
                    xticklabels=["No Disease","Disease"],
                    yticklabels=["No Disease","Disease"],
                    annot_kws={"size":16,"weight":"800","color":"#fff"})
        ax.set_xlabel("Predicted", color=PALETTE["muted"])
        ax.set_ylabel("Actual",    color=PALETTE["muted"])
        ax.set_title("Random Forest — confusion matrix", color="#e2e8f0", fontsize=10, fontweight="700", pad=10)
        ax.tick_params(colors=PALETTE["muted"], labelsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
# PAGE 4 — RISK ASSESSMENT
# ══════════════════════════════════════════════════════════
elif "Risk" in page:
    st.markdown('<div class="sec-label">Live prediction engine</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sec-title">Patient risk assessment</h2>', unsafe_allow_html=True)

    model_choice = st.selectbox("Prediction model", [
        "🔴  Logistic Regression  ·  Best AUC 0.962",
        "🟢  Random Forest  ·  Best Accuracy 88.5%",
        "🔵  Gradient Boosting  ·  AUC 0.932"
    ])
    selected_model = {
        "🔴  Logistic Regression  ·  Best AUC 0.962": lr,
        "🟢  Random Forest  ·  Best Accuracy 88.5%":  rf,
        "🔵  Gradient Boosting  ·  AUC 0.932":        gb,
    }[model_choice]

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 1, 0.9], gap="large")

    with col_a:
        st.markdown('<div class="inp-group">Demographics & symptoms</div>', unsafe_allow_html=True)
        age      = st.slider("Age", 20, 80, 55)
        sex      = st.selectbox("Sex", ["Female (0)","Male (1)"])
        cp       = st.selectbox("Chest pain type", ["Typical angina (1)","Atypical angina (2)","Non-anginal pain (3)","Asymptomatic (4)"])
        trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 200, 130)
        chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 245)
        fbs      = st.selectbox("Fasting blood sugar > 120", ["No (0)","Yes (1)"])

    with col_b:
        st.markdown('<div class="inp-group">Cardiac measurements</div>', unsafe_allow_html=True)
        restecg = st.selectbox("Resting ECG result", ["Normal (0)","ST-T abnormality (1)","LV hypertrophy (2)"])
        thalach = st.slider("Max heart rate", 60, 210, 150)
        exang   = st.selectbox("Exercise induced angina", ["No (0)","Yes (1)"])
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
        slope   = st.selectbox("Slope of ST segment", ["Upsloping (1)","Flat (2)","Downsloping (3)"])
        ca      = st.selectbox("Major vessels coloured (0–3)", [0,1,2,3])
        thal    = st.selectbox("Thalassemia", ["Normal (3)","Fixed defect (6)","Reversible defect (7)"])

    with col_c:
        st.markdown('<div class="inp-group" style="margin-top:0;">Result</div>', unsafe_allow_html=True)
        predict_btn = st.button("🔮  Analyse patient risk")

        if predict_btn:
            sex_val     = int(sex.split("(")[1][0])
            cp_val      = int(cp.split("(")[1][0])
            fbs_val     = int(fbs.split("(")[1][0])
            restecg_val = int(restecg.split("(")[1][0])
            exang_val   = int(exang.split("(")[1][0])
            slope_val   = int(slope.split("(")[1][0])
            thal_val    = int(thal.split("(")[1][0])
            ca_val      = int(ca)

            raw_enc = pd.DataFrame(0.0, index=[0], columns=X_train.columns)
            raw_enc["age"]      = float(age)
            raw_enc["trestbps"] = float(trestbps)
            raw_enc["chol"]     = float(chol)
            raw_enc["thalach"]  = float(thalach)
            raw_enc["oldpeak"]  = float(oldpeak)

            def set_col(prefix, val):
                col = f"{prefix}_{float(val)}"
                if col in raw_enc.columns:
                    raw_enc[col] = 1.0

            for prefix, val in [("sex",sex_val),("cp",cp_val),("fbs",fbs_val),
                                  ("restecg",restecg_val),("exang",exang_val),
                                  ("slope",slope_val),("ca",ca_val),("thal",thal_val)]:
                set_col(prefix, val)

            raw_enc[["age","trestbps","chol","thalach","oldpeak"]] = \
                scaler.transform(raw_enc[["age","trestbps","chol","thalach","oldpeak"]])

            pred     = selected_model.predict(raw_enc)[0]
            prob     = selected_model.predict_proba(raw_enc)[0][1]
            risk_pct = prob * 100
            conf_pct = max(prob, 1-prob) * 100

            if pred == 1:
                st.markdown(f"""
                <div class="result-card-high">
                    <div class="res-icon">⚠️</div>
                    <div class="res-label-h">High Risk</div>
                    <div class="res-prob">Probability: <strong>{risk_pct:.1f}%</strong></div>
                    <div style="margin-top:0.6rem;font-size:0.78rem;color:rgba(220,38,38,0.7);">Please consult a cardiologist</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-low">
                    <div class="res-icon">✅</div>
                    <div class="res-label-l">Low Risk</div>
                    <div class="res-prob">Probability: <strong>{risk_pct:.1f}%</strong></div>
                    <div style="margin-top:0.6rem;font-size:0.78rem;color:rgba(34,197,94,0.7);">Continue regular health checkups</div>
                </div>""", unsafe_allow_html=True)

            fill_color = "#dc2626" if pred==1 else "#22c55e"
            st.markdown(f"""
            <div class="conf-wrap">
                <div class="conf-label">Model confidence</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill-{'h' if pred==1 else 'l'}" style="width:{conf_pct:.1f}%"></div>
                </div>
                <div class="conf-pct">{conf_pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            # Gauge
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor(PALETTE["bg"])
            ax.set_facecolor(PALETTE["bg"])
            theta_full = np.linspace(0, np.pi, 300)
            ax.plot(np.cos(theta_full), np.sin(theta_full), color=PALETTE["border"], lw=18,
                    solid_capstyle="round")
            end_a = np.pi * (1 - prob)
            theta_fill = np.linspace(np.pi, end_a, 300)
            ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=fill_color, lw=18,
                    solid_capstyle="round")
            ax.text(0, 0.05, f"{risk_pct:.1f}%", ha="center", va="center",
                    fontsize=26, fontweight="900", color=fill_color)
            ax.text(0, -0.28, "RISK SCORE", ha="center", color=PALETTE["muted"],
                    fontsize=8, fontweight="700")
            ax.text(-0.95, -0.08, "0%",  ha="center", color=PALETTE["muted"], fontsize=8)
            ax.text( 0.95, -0.08, "100%",ha="center", color=PALETTE["muted"], fontsize=8)
            ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.5,1.25)
            ax.axis("off")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        else:
            st.markdown("""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                        border-radius:20px;padding:2.5rem 1.5rem;text-align:center;margin-top:0.5rem;">
                <div style="font-size:3rem;margin-bottom:1rem;opacity:0.4;">🫀</div>
                <div style="font-size:0.85rem;color:rgba(255,255,255,0.25);line-height:1.8;">
                    Enter patient details and click<br>
                    <strong style="color:rgba(255,255,255,0.4);">Analyse patient risk</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1rem;padding:0.8rem 1rem;border-radius:10px;
                    background:rgba(220,38,38,0.06);border-left:2px solid rgba(220,38,38,0.3);
                    font-size:0.75rem;color:rgba(255,255,255,0.3);line-height:1.6;">
            ⚠️ Educational use only — not a medical diagnostic tool.
        </div>""", unsafe_allow_html=True)

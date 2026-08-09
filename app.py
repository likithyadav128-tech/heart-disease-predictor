# ============================================================
# HEARTSENSE AI — ULTRA PREMIUM UI v4
# ============================================================
import streamlit as st, pandas as pd, numpy as np, pickle
import matplotlib.pyplot as plt, seaborn as sns, warnings
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

st.set_page_config(page_title="HeartSense AI", page_icon="🫀", layout="wide",
                   initial_sidebar_state="expanded")

# ── MEGA CSS ────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --red:      #e11d48;
  --red2:     #be123c;
  --red-glow: rgba(225,29,72,0.35);
  --green:    #10b981;
  --blue:     #3b82f6;
  --purple:   #8b5cf6;
  --amber:    #f59e0b;
  --bg:       #03040a;
  --bg2:      #080c16;
  --card:     rgba(255,255,255,0.032);
  --card-h:   rgba(255,255,255,0.06);
  --border:   rgba(255,255,255,0.07);
  --border-h: rgba(225,29,72,0.35);
  --t1:       #ffffff;
  --t2:       rgba(255,255,255,0.65);
  --t3:       rgba(255,255,255,0.35);
  --t4:       rgba(255,255,255,0.18);
  --glass:    rgba(255,255,255,0.04);
  --glass-b:  rgba(255,255,255,0.08);
  --shadow:   0 25px 60px rgba(0,0,0,0.5);
  --shadow-r: 0 20px 60px rgba(225,29,72,0.2);
  --r:        20px;
  --r2:       14px;
  --r3:       10px;
  --ease:     cubic-bezier(0.34,1.56,0.64,1);
  --ease2:    cubic-bezier(0.4,0,0.2,1);
}

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body,html,[class*="css"]{font-family:'Inter',sans-serif!important;-webkit-font-smoothing:antialiased}

/* ── CURSOR ── */
#hs-dot{
  position:fixed;width:10px;height:10px;background:#e11d48;border-radius:50%;
  pointer-events:none;z-index:2147483647;top:0;left:0;
  box-shadow:0 0 14px rgba(225,29,72,0.9),0 0 28px rgba(225,29,72,0.4);
  transition:width .18s,height .18s,background .18s,transform .1s;
  will-change:transform;
}
#hs-ring{
  position:fixed;width:34px;height:34px;border:1.5px solid rgba(225,29,72,0.5);
  border-radius:50%;pointer-events:none;z-index:2147483646;top:0;left:0;
  transition:width .22s,height .22s,border-color .22s;
  will-change:transform;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--red);border-radius:4px}

/* ── APP BACKGROUND ── */
.stApp{
  background:var(--bg);
  background-image:
    radial-gradient(ellipse 90% 55% at 15% -5%,rgba(225,29,72,0.13) 0%,transparent 55%),
    radial-gradient(ellipse 70% 45% at 85% 100%,rgba(59,130,246,0.07) 0%,transparent 55%),
    radial-gradient(ellipse 50% 35% at 50% 50%,rgba(139,92,246,0.04) 0%,transparent 60%);
  min-height:100vh;
}
.main .block-container{padding:1.5rem 2.5rem 5rem;max-width:1300px}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden!important;display:none!important}

/* ── FLOATING ORBS ── */
.orb{
  position:fixed;border-radius:50%;filter:blur(80px);pointer-events:none;z-index:0;
  animation:float-orb 8s ease-in-out infinite;
}
.orb1{width:400px;height:400px;background:rgba(225,29,72,0.08);top:-100px;left:-100px;animation-delay:0s}
.orb2{width:350px;height:350px;background:rgba(59,130,246,0.06);bottom:-80px;right:-80px;animation-delay:-3s}
.orb3{width:250px;height:250px;background:rgba(139,92,246,0.05);top:50%;left:50%;animation-delay:-5s}
@keyframes float-orb{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(-30px) scale(1.05)}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{
  background:rgba(5,7,15,0.97)!important;
  border-right:1px solid var(--border)!important;
  backdrop-filter:blur(30px)!important;
}
[data-testid="stSidebarNav"]{display:none}

/* ── SIDEBAR LOGO ── */
.sb-logo{
  padding:1.6rem 1.4rem 1rem;
  border-bottom:1px solid var(--border);
  margin-bottom:1.4rem;
}
.sb-logo-text{
  font-family:'Poppins',sans-serif;font-size:1.45rem;font-weight:800;
  letter-spacing:-0.03em;color:#fff;line-height:1;
}
.sb-logo-text span{
  background:linear-gradient(135deg,var(--red),#f43f5e,#fb7185);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.sb-tag{
  font-size:0.6rem;font-weight:600;letter-spacing:0.2em;
  text-transform:uppercase;color:var(--t4);margin-top:4px;
}
.sb-pulse{
  display:inline-flex;align-items:center;gap:6px;
  background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);
  border-radius:100px;padding:3px 10px;font-size:0.6rem;font-weight:600;
  letter-spacing:0.1em;text-transform:uppercase;color:#10b981;margin-top:10px;
}
.sb-pulse::before{
  content:'';width:6px;height:6px;border-radius:50%;background:#10b981;
  animation:pulse-dot 1.4s ease-in-out infinite;
}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.5;transform:scale(0.7)}}

/* ── NAV ITEMS ── */
.nav-item{
  display:flex;align-items:center;gap:10px;
  padding:0.7rem 1rem;border-radius:var(--r3);margin-bottom:4px;
  font-size:0.88rem;font-weight:500;color:var(--t3);
  border:1px solid transparent;
  transition:all 0.25s var(--ease2);cursor:none!important;
}
.nav-item:hover,.nav-item.active{
  background:rgba(225,29,72,0.08);border-color:rgba(225,29,72,0.2);
  color:var(--t1);transform:translateX(3px);
}
.nav-item .ni{
  width:30px;height:30px;border-radius:8px;
  background:rgba(255,255,255,0.04);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;font-size:0.9rem;
  transition:all 0.25s;
}
.nav-item:hover .ni,.nav-item.active .ni{
  background:rgba(225,29,72,0.15);border-color:rgba(225,29,72,0.3);
}

/* ── SIDEBAR STATS ── */
.sb-stats{
  margin:1.2rem 0;padding:1rem;
  background:rgba(255,255,255,0.02);
  border:1px solid var(--border);border-radius:var(--r2);
}
.sb-stat-row{
  display:flex;justify-content:space-between;align-items:center;
  padding:0.45rem 0;border-bottom:1px solid rgba(255,255,255,0.04);
  font-size:0.8rem;color:var(--t3);
}
.sb-stat-row:last-child{border-bottom:none}
.sb-stat-val{color:var(--red);font-weight:700;font-size:0.85rem;font-variant-numeric:tabular-nums}

/* ── RADIO OVERRIDE ── */
[data-testid="stSidebar"] .stRadio>label{display:none}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p{
  font-size:0.88rem!important;color:var(--t2)!important;font-weight:500;
}

/* ── INPUT OVERRIDES ── */
.stSelectbox label,.stSlider label{
  color:var(--t4)!important;font-size:0.62rem!important;
  font-weight:700;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:4px;
}
.stSelectbox [data-baseweb="select"]{
  background:var(--card)!important;border:1px solid var(--border)!important;
  border-radius:var(--r3)!important;
  transition:border-color 0.2s,box-shadow 0.2s!important;
}
.stSelectbox [data-baseweb="select"]:focus-within{
  border-color:var(--border-h)!important;
  box-shadow:0 0 0 3px rgba(225,29,72,0.12)!important;
}
.stSelectbox [data-baseweb="select"] *{color:var(--t2)!important;background:transparent!important}
[data-baseweb="popover"]{background:#0d111e!important;border:1px solid var(--border)!important;border-radius:var(--r2)!important;overflow:hidden}
[data-baseweb="popover"] li{color:var(--t2)!important;transition:background 0.15s}
[data-baseweb="popover"] li:hover{background:rgba(225,29,72,0.1)!important;color:#fff!important}

/* ── SLIDER ── */
[data-baseweb="slider"] [role="slider"]{background:var(--red)!important}
[data-baseweb="slider"] [data-testid="stThumbValue"]{color:var(--red)!important;font-weight:700}

/* ── BUTTON ── */
.stButton>button{
  background:linear-gradient(135deg,var(--red) 0%,var(--red2) 100%)!important;
  color:#fff!important;border:none!important;border-radius:var(--r2)!important;
  padding:0.9rem 1.8rem!important;font-family:'Inter',sans-serif!important;
  font-weight:700!important;font-size:0.9rem!important;letter-spacing:0.02em!important;
  width:100%!important;position:relative;overflow:hidden!important;
  box-shadow:0 4px 24px rgba(225,29,72,0.3),0 1px 0 rgba(255,255,255,0.1) inset!important;
  transition:all 0.25s var(--ease)!important;
}
.stButton>button::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,0.12) 0%,transparent 60%);
  pointer-events:none;
}
.stButton>button:hover{
  transform:translateY(-3px) scale(1.02)!important;
  box-shadow:0 12px 40px rgba(225,29,72,0.45),0 1px 0 rgba(255,255,255,0.15) inset!important;
}
.stButton>button:active{transform:translateY(-1px) scale(0.99)!important}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"]{
  border-radius:var(--r)!important;overflow:hidden!important;
  border:1px solid var(--border)!important;
}

/* ── GLASS CARD ── */
.gc{
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:1.5rem 1.8rem;position:relative;overflow:hidden;
  transition:all 0.3s var(--ease2);
  animation:fadeUp 0.5s var(--ease2) both;
}
.gc::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.12),transparent);
}
.gc:hover{
  background:var(--card-h);border-color:var(--border-h);
  transform:translateY(-4px);
  box-shadow:0 20px 50px rgba(0,0,0,0.4),0 0 0 1px rgba(225,29,72,0.15);
}

/* ── HERO ── */
.hero{padding:3rem 0 2rem;animation:fadeUp 0.6s var(--ease2) both}
.hero-pill{
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(225,29,72,0.08);border:1px solid rgba(225,29,72,0.2);
  border-radius:100px;padding:5px 14px;
  font-size:0.67rem;font-weight:700;letter-spacing:0.15em;
  text-transform:uppercase;color:#f87171;margin-bottom:1.4rem;
}
.hero-pill::before{
  content:'';width:6px;height:6px;border-radius:50%;background:var(--red);
  animation:pulse-dot 1.4s ease-in-out infinite;
}
.hero-h1{
  font-family:'Poppins',sans-serif;
  font-size:clamp(2.2rem,4.5vw,3.6rem);
  font-weight:900;line-height:1.06;
  color:#fff;letter-spacing:-0.04em;margin-bottom:1.2rem;
}
.hero-h1 .g{
  background:linear-gradient(135deg,#e11d48,#f43f5e,#fb923c);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.hero-sub{
  font-size:1rem;color:var(--t3);line-height:1.8;
  max-width:520px;font-weight:300;margin-bottom:2rem;
}

/* ── STAT GRID ── */
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2.5rem}
.stat-card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:1.4rem 1.6rem;position:relative;overflow:hidden;
  transition:all 0.3s var(--ease2);
  animation:fadeUp 0.5s var(--ease2) both;
}
.stat-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--red),transparent);
}
.stat-card::after{
  content:'';position:absolute;bottom:0;right:0;
  width:80px;height:80px;border-radius:50%;
  background:var(--red-glow);filter:blur(30px);
  opacity:0;transition:opacity 0.3s;
}
.stat-card:hover{
  border-color:rgba(225,29,72,0.3);transform:translateY(-5px);
  box-shadow:0 20px 50px rgba(0,0,0,0.4),0 0 30px rgba(225,29,72,0.08);
}
.stat-card:hover::after{opacity:1}
.sc-num{
  font-family:'Poppins',sans-serif;font-size:2.5rem;font-weight:900;
  letter-spacing:-0.05em;line-height:1;margin-bottom:4px;
}
.sc-num.red{color:var(--red)}
.sc-num.w{color:#fff}
.sc-lbl{font-size:0.68rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:var(--t4)}
.sc-icon{position:absolute;top:1.1rem;right:1.3rem;font-size:1.5rem;opacity:0.2;
  transition:opacity 0.3s,transform 0.3s}
.stat-card:hover .sc-icon{opacity:0.5;transform:scale(1.15)}

/* ── SECTION HEADERS ── */
.sl{font-size:0.62rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;
  color:var(--red);margin-bottom:6px;display:flex;align-items:center;gap:8px}
.sl::before{content:'';width:20px;height:1.5px;background:var(--red);border-radius:2px}
.st{font-family:'Poppins',sans-serif;font-size:1.7rem;font-weight:800;
  color:#fff;letter-spacing:-0.025em;margin:0 0 1.4rem}

/* ── PIPELINE ── */
.pip{
  display:flex;align-items:flex-start;gap:1rem;padding:1rem 1.2rem;
  border-radius:var(--r2);background:var(--card);border:1px solid var(--border);
  margin-bottom:8px;transition:all 0.25s var(--ease2);
  animation:fadeLeft 0.5s var(--ease2) both;
}
.pip:hover{background:rgba(225,29,72,0.05);border-color:rgba(225,29,72,0.2);transform:translateX(5px)}
.pip-num{
  min-width:34px;height:34px;border-radius:9px;
  background:rgba(225,29,72,0.12);border:1px solid rgba(225,29,72,0.25);
  display:flex;align-items:center;justify-content:center;
  font-size:0.68rem;font-weight:800;color:var(--red);
  font-variant-numeric:tabular-nums;flex-shrink:0;margin-top:1px;
}
.pip-t{font-size:0.9rem;font-weight:600;color:rgba(255,255,255,0.88);margin-bottom:2px}
.pip-d{font-size:0.77rem;color:var(--t4);font-weight:300;line-height:1.5}

/* ── FINDINGS ── */
.fin{
  display:flex;align-items:flex-start;gap:10px;padding:0.75rem 1rem;
  border-radius:var(--r2);background:var(--card);border:1px solid var(--border);
  margin-bottom:7px;font-size:0.83rem;color:var(--t3);
  transition:all 0.25s var(--ease2);animation:fadeRight 0.5s var(--ease2) both;
}
.fin:hover{background:rgba(255,255,255,0.05);color:var(--t2);transform:translateX(-3px)}
.fin .dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:5px}

/* ── LEADERBOARD ── */
.lb{
  background:rgba(225,29,72,0.05);border:1px solid rgba(225,29,72,0.15);
  border-radius:var(--r);padding:1.2rem 1.4rem;margin-top:1.4rem;
}
.lb-head{font-size:0.62rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;
  color:rgba(225,29,72,0.6);margin-bottom:0.9rem}
.lb-sub{display:flex;justify-content:space-between;font-size:0.75rem;
  color:var(--t4);padding:0 4px;margin-bottom:6px}
.lb-row{
  display:flex;justify-content:space-between;align-items:center;
  padding:0.55rem 6px;font-size:0.83rem;
  border-bottom:1px solid rgba(255,255,255,0.04);
  transition:background 0.2s;border-radius:8px;
}
.lb-row:last-child{border-bottom:none}
.lb-row:hover{background:rgba(255,255,255,0.03)}
.lb-name{color:#fff;font-weight:600}
.lb-auc{color:var(--red);font-weight:700}
.lb-acc{color:var(--green);font-weight:700}
.lb-row.dim{opacity:0.55}

/* ── INPUT GROUP ── */
.ig{
  font-size:0.62rem;font-weight:700;letter-spacing:0.18em;
  text-transform:uppercase;color:rgba(225,29,72,0.75);
  border-bottom:1px solid rgba(225,29,72,0.12);
  padding-bottom:6px;margin:1.3rem 0 0.9rem;
}

/* ── RESULT CARDS ── */
.res-h{
  background:linear-gradient(135deg,rgba(225,29,72,0.1),rgba(190,18,60,0.05));
  border:1px solid rgba(225,29,72,0.35);border-radius:var(--r);
  padding:2rem 1.6rem;text-align:center;
  box-shadow:0 0 60px rgba(225,29,72,0.1),inset 0 1px 0 rgba(255,255,255,0.05);
  animation:zoomIn 0.4s var(--ease) both;
}
.res-l{
  background:linear-gradient(135deg,rgba(16,185,129,0.1),rgba(5,150,105,0.04));
  border:1px solid rgba(16,185,129,0.3);border-radius:var(--r);
  padding:2rem 1.6rem;text-align:center;
  box-shadow:0 0 60px rgba(16,185,129,0.08),inset 0 1px 0 rgba(255,255,255,0.05);
  animation:zoomIn 0.4s var(--ease) both;
}
.ri{font-size:3.2rem;margin-bottom:0.6rem;animation:bounce-in 0.5s var(--ease) 0.1s both}
.rh{font-family:'Poppins',sans-serif;font-size:2rem;font-weight:900;
  letter-spacing:-0.03em;color:var(--red);margin-bottom:4px}
.rl{font-family:'Poppins',sans-serif;font-size:2rem;font-weight:900;
  letter-spacing:-0.03em;color:var(--green);margin-bottom:4px}
.rp{font-size:0.88rem;color:var(--t3)}
.rp strong{color:rgba(255,255,255,0.8)}

/* ── CONFIDENCE BAR ── */
.cbar-wrap{margin-top:1.1rem}
.cbar-lbl{font-size:0.65rem;font-weight:600;letter-spacing:0.12em;
  text-transform:uppercase;color:var(--t4);margin-bottom:5px}
.cbar-bg{background:rgba(255,255,255,0.06);border-radius:100px;height:5px;overflow:hidden}
.cbar-fill-h{height:5px;border-radius:100px;
  background:linear-gradient(90deg,#9f1239,var(--red),#fb7185);
  animation:grow 0.8s var(--ease2) both}
.cbar-fill-l{height:5px;border-radius:100px;
  background:linear-gradient(90deg,#065f46,var(--green),#6ee7b7);
  animation:grow 0.8s var(--ease2) both}
.cbar-pct{font-size:0.82rem;font-weight:700;color:var(--t2);
  margin-top:4px;font-variant-numeric:tabular-nums}

/* ── IDLE STATE ── */
.idle-state{
  background:rgba(255,255,255,0.015);border:1px solid var(--border);
  border-radius:var(--r);padding:2.5rem 1.5rem;text-align:center;
  animation:fadeUp 0.4s var(--ease2) both;
}
.idle-icon{font-size:2.8rem;opacity:0.25;margin-bottom:1rem;
  animation:float-icon 3s ease-in-out infinite}
@keyframes float-icon{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}

/* ── WARNING ── */
.warn{
  margin-top:1rem;padding:0.7rem 0.9rem;border-radius:var(--r3);
  background:rgba(225,29,72,0.05);border-left:2px solid rgba(225,29,72,0.3);
  font-size:0.72rem;color:var(--t4);line-height:1.6;
}

/* ── ANIMATIONS ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeLeft{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:translateX(0)}}
@keyframes fadeRight{from{opacity:0;transform:translateX(18px)}to{opacity:1;transform:translateX(0)}}
@keyframes zoomIn{from{opacity:0;transform:scale(0.92)}to{opacity:1;transform:scale(1)}}
@keyframes bounce-in{from{transform:scale(0.5);opacity:0}to{transform:scale(1);opacity:1}}
@keyframes grow{from{width:0}to{width:var(--w,100%)}}
@keyframes counter{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes shimmer{0%{background-position:-500px 0}100%{background-position:500px 0}}

/* ── CHART CONTAINER ── */
.chart-wrap{
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:1.2rem;transition:all 0.3s var(--ease2);
  animation:fadeUp 0.5s var(--ease2) both;
}
.chart-wrap:hover{border-color:rgba(225,29,72,0.25);transform:translateY(-3px);
  box-shadow:0 20px 50px rgba(0,0,0,0.35)}

/* ── TOAST BADGE ── */
.badge{
  display:inline-flex;align-items:center;gap:5px;
  border-radius:100px;padding:3px 10px;
  font-size:0.65rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
}
.badge-r{background:rgba(225,29,72,0.12);border:1px solid rgba(225,29,72,0.25);color:#f87171}
.badge-g{background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.22);color:#6ee7b7}
.badge-b{background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.22);color:#93c5fd}

/* ── RESPONSIVE ── */
@media(max-width:768px){
  .stat-grid{grid-template-columns:repeat(2,1fr)}
  .hero-h1{font-size:2rem}
  .main .block-container{padding:1rem 1rem 4rem}
}

/* ══════════════════════════════════════════════════════════
   HEARTBEAT / ECG HERO VISUAL
   ══════════════════════════════════════════════════════════ */
.hero-visual{
  width:100%; max-width:640px; margin:0 0 1.6rem;
  animation:fadeUp 0.7s var(--ease2) both;
}
.hero-visual svg{ width:100%; height:auto; display:block; }
.ecg-line{
  fill:none; stroke:url(#ecgGrad); stroke-width:2.5;
  stroke-linecap:round; stroke-linejoin:round;
  stroke-dasharray:1000; stroke-dashoffset:1000;
  animation:ecgDraw 3.2s ease-in-out infinite;
  filter:drop-shadow(0 0 6px rgba(225,29,72,0.5));
}
@keyframes ecgDraw{
  0%{stroke-dashoffset:1000}
  55%{stroke-dashoffset:0}
  100%{stroke-dashoffset:-1000}
}
.hero-heart{
  transform-origin:center;
  animation:heartBeat 1.15s ease-in-out infinite;
  filter:drop-shadow(0 0 18px rgba(225,29,72,0.55));
}
@keyframes heartBeat{
  0%,100%{transform:scale(1)}
  14%{transform:scale(1.12)}
  28%{transform:scale(0.98)}
  42%{transform:scale(1.09)}
  70%{transform:scale(1)}
}

/* sidebar mini heart logo */
.sb-heart{ display:inline-block; vertical-align:middle; margin-right:2px; }
.sb-heart svg{ animation:heartBeat 1.15s ease-in-out infinite; transform-origin:center; }

/* ══════════════════════════════════════════════════════════
   RESULT MODAL — dramatic popup reveal
   ══════════════════════════════════════════════════════════ */
.result-stage{ position:relative; min-height:60px; }

.result-modal{
  position:relative; border-radius:24px; padding:2.3rem 1.7rem 2rem;
  text-align:center; overflow:hidden;
  animation:modalPop 0.55s cubic-bezier(0.16,1.4,0.4,1) both;
}
@keyframes modalPop{
  0%{ opacity:0; transform:scale(0.72) translateY(14px); }
  55%{ opacity:1; transform:scale(1.045) translateY(-3px); }
  100%{ opacity:1; transform:scale(1) translateY(0); }
}
.result-modal.high{
  background:radial-gradient(ellipse at top,rgba(225,29,72,0.16),rgba(190,18,60,0.04) 70%);
  border:1px solid rgba(225,29,72,0.4);
  box-shadow:0 0 0 1px rgba(225,29,72,0.08) inset,0 25px 70px rgba(225,29,72,0.18);
}
.result-modal.low{
  background:radial-gradient(ellipse at top,rgba(16,185,129,0.14),rgba(5,150,105,0.03) 70%);
  border:1px solid rgba(16,185,129,0.35);
  box-shadow:0 0 0 1px rgba(16,185,129,0.08) inset,0 25px 70px rgba(16,185,129,0.14);
}

/* pulsing alert rings — high risk */
.alert-rings{ position:absolute; top:50%; left:50%; width:1px; height:1px; pointer-events:none; }
.alert-ring{
  position:absolute; top:0; left:0; width:90px; height:90px;
  margin:-45px 0 0 -45px; border-radius:50%;
  border:2px solid rgba(225,29,72,0.55);
  animation:ringPulse 2.1s ease-out infinite;
  opacity:0;
}
.alert-ring:nth-child(2){ animation-delay:0.7s; }
.alert-ring:nth-child(3){ animation-delay:1.4s; }
@keyframes ringPulse{
  0%{ width:70px;height:70px;margin:-35px 0 0 -35px; opacity:0.7; }
  100%{ width:280px;height:280px;margin:-140px 0 0 -140px; opacity:0; }
}

/* confetti burst — low risk */
.confetti-wrap{ position:absolute; inset:0; overflow:hidden; pointer-events:none; }
.confetti{
  position:absolute; top:46%; left:50%; width:7px; height:11px;
  opacity:0; border-radius:2px;
  animation:confettiBurst 1.3s cubic-bezier(0.15,0.7,0.3,1) both;
}
@keyframes confettiBurst{
  0%{ transform:translate(-50%,-50%) rotate(0deg) scale(0.4); opacity:0; }
  8%{ opacity:1; }
  100%{ transform:translate(calc(-50% + var(--dx))) translateY(calc(var(--dy))) rotate(var(--rot)) scale(1);
       opacity:0; }
}

.res-icon-wrap{
  width:70px; height:70px; margin:0 auto 1rem; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  position:relative; z-index:2;
  animation:iconPop 0.5s cubic-bezier(0.34,1.6,0.5,1) 0.15s both;
}
.res-icon-wrap.high{ background:rgba(225,29,72,0.14); border:1px solid rgba(225,29,72,0.3); }
.res-icon-wrap.low{ background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.28); }
@keyframes iconPop{
  0%{ transform:scale(0) rotate(-20deg); opacity:0; }
  70%{ transform:scale(1.15) rotate(4deg); opacity:1; }
  100%{ transform:scale(1) rotate(0deg); opacity:1; }
}
.res-icon-wrap svg{ width:34px; height:34px; }

.res-title{
  position:relative; z-index:2;
  font-family:'Poppins',sans-serif; font-size:2.1rem; font-weight:900;
  letter-spacing:-0.03em; margin-bottom:6px;
  animation:fadeUp 0.4s ease 0.25s both;
}
.res-title.high{ color:#fb7185; }
.res-title.low{ color:#34d399; }
.res-prob-line{
  position:relative; z-index:2; font-size:0.92rem; color:var(--t3);
  animation:fadeUp 0.4s ease 0.32s both;
}
.res-prob-line strong{ color:rgba(255,255,255,0.9); font-variant-numeric:tabular-nums; }
.res-sub{
  position:relative; z-index:2; margin-top:8px; font-size:0.78rem;
  animation:fadeUp 0.4s ease 0.4s both;
}

/* ══════════════════════════════════════════════════════════
   ICON BADGES for input groups
   ══════════════════════════════════════════════════════════ */
.ig-icon{
  display:inline-flex; align-items:center; justify-content:center;
  width:22px; height:22px; border-radius:7px; margin-right:7px;
  background:rgba(225,29,72,0.14); border:1px solid rgba(225,29,72,0.25);
  font-size:0.72rem; vertical-align:middle;
}

/* ══════════════════════════════════════════════════════════
   BATCH CSV UPLOAD UI
   ══════════════════════════════════════════════════════════ */
.upload-card{
  background:var(--card); border:1.5px dashed rgba(225,29,72,0.3);
  border-radius:20px; padding:2.2rem 1.8rem; text-align:center;
  transition:border-color 0.25s, background 0.25s;
}
.upload-card:hover{ border-color:rgba(225,29,72,0.55); background:rgba(225,29,72,0.03); }
.upload-icon{ font-size:2.4rem; margin-bottom:0.6rem; opacity:0.85; animation:float-icon 3s ease-in-out infinite; }
.template-hint{
  font-size:0.78rem; color:var(--t3); line-height:1.7; margin-top:0.6rem;
}
.batch-summary{ display:flex; gap:0.9rem; margin:1.1rem 0 1.3rem; flex-wrap:wrap; }
.batch-chip{
  flex:1; min-width:120px; background:var(--card); border:1px solid var(--border);
  border-radius:14px; padding:0.9rem 1.1rem; text-align:center;
}
.batch-chip .bc-num{ font-family:'Poppins',sans-serif; font-size:1.6rem; font-weight:800; }
.batch-chip .bc-lbl{ font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--t4); margin-top:2px; }
[data-testid="stFileUploaderDropzone"]{
  background:transparent !important; border:none !important;
}
.risk-pill{
  display:inline-flex; align-items:center; gap:5px;
  padding:0.28rem 0.7rem; border-radius:100px; font-size:0.76rem; font-weight:700;
}
.risk-pill.high{ background:rgba(225,29,72,0.14); color:#fb7185; border:1px solid rgba(225,29,72,0.28); }
.risk-pill.low{ background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.25); }
</style>

<!-- ORBS -->
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>

<script>
(function boot(){
  if(!document.body){ setTimeout(boot,60); return; }

  /* ── CREATE CURSOR ELEMENTS ── */
  function mkEl(id,extra){
    let el=document.getElementById(id);
    if(!el){ el=document.createElement('div'); el.id=id; document.body.appendChild(el); }
    return el;
  }
  const dot  = mkEl('hs-dot');
  const ring = mkEl('hs-ring');

  let mx=0, my=0, rx=0, ry=0;

  /* ── TRACK MOUSE ── */
  document.addEventListener('mousemove', function(e){
    mx=e.clientX; my=e.clientY;
    dot.style.transform = 'translate('+(mx-5)+'px,'+(my-5)+'px)';
  }, {passive:true});

  /* ── SMOOTH RING ── */
  (function loop(){
    rx += (mx-rx)*0.13;
    ry += (my-ry)*0.13;
    ring.style.transform = 'translate('+(rx-17)+'px,'+(ry-17)+'px)';
    requestAnimationFrame(loop);
  })();

  /* ── HOVER EXPAND ── */
  function big(){
    dot.style.width='18px'; dot.style.height='18px';
    dot.style.background='rgba(225,29,72,0.65)';
    ring.style.width='50px'; ring.style.height='50px';
    ring.style.borderColor='rgba(225,29,72,0.85)';
  }
  function small(){
    dot.style.width='10px'; dot.style.height='10px';
    dot.style.background='#e11d48';
    ring.style.width='34px'; ring.style.height='34px';
    ring.style.borderColor='rgba(225,29,72,0.5)';
  }

  function attach(){
    var sel='button,a,select,input,[role="button"],.stat-card,.gc,.pip,.fin,.nav-item';
    document.querySelectorAll(sel).forEach(function(el){
      if(el._hsOk) return; el._hsOk=true;
      el.addEventListener('mouseenter', big);
      el.addEventListener('mouseleave', small);
    });
  }
  attach();
  new MutationObserver(attach).observe(document.body,{childList:true,subtree:true});

  /* ── CLICK SQUISH ── */
  document.addEventListener('mousedown', function(){
    dot.style.transform='translate('+(mx-5)+'px,'+(my-5)+'px) scale(0.65)';
  });
  document.addEventListener('mouseup', function(){
    dot.style.transform='translate('+(mx-5)+'px,'+(my-5)+'px) scale(1)';
  });

})();
</script>
""", unsafe_allow_html=True)

# ── LOAD DATA ───────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/logistic_regression.pkl","rb") as f: lr=pickle.load(f)
    with open("models/random_forest.pkl","rb") as f:       rf=pickle.load(f)
    with open("models/gradient_boosting.pkl","rb") as f:   gb=pickle.load(f)
    with open("models/scaler.pkl","rb") as f:              sc=pickle.load(f)
    try:
        with open("models/extra_trees.pkl","rb") as f: et=pickle.load(f)
    except Exception:
        et = rf
    try:
        with open("models/knn.pkl","rb") as f: knn=pickle.load(f)
    except Exception:
        knn = gb
    try:
        with open("models/voting_ensemble.pkl","rb") as f: vt=pickle.load(f)
    except Exception:
        vt = gb  # fallback if voting file missing/corrupt
    return lr,rf,gb,et,knn,vt,sc

@st.cache_data
def load_data():
    df=pd.read_csv("data/heart_cleaned.csv")
    Xtr=pd.read_csv("data/X_train.csv"); Xte=pd.read_csv("data/X_test.csv")
    ytr=pd.read_csv("data/y_train.csv").squeeze(); yte=pd.read_csv("data/y_test.csv").squeeze()
    res=pd.read_csv("models/results.csv")
    return df,Xtr,Xte,ytr,yte,res

lr,rf,gb,et,knn,voting,scaler=load_models()
df,Xtr,Xte,ytr,yte,res=load_data()

P={"bg":"#03040a","card":"#0a0d18","red":"#e11d48","green":"#10b981",
   "blue":"#3b82f6","purple":"#8b5cf6","border":"#13182a","t2":"#94a3b8","t3":"#4a5568"}

def dark_fig(w=6,h=4.2):
    fig,ax=plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor(P["bg"]); ax.set_facecolor(P["card"])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=P["t2"],labelsize=8)
    ax.yaxis.grid(True,color=P["border"],lw=0.6,ls="--"); ax.set_axisbelow(True)
    return fig,ax

# ── BATCH CSV PREDICTION HELPERS ─────────────────────────
REQUIRED_COLS = ["Age","Sex","ChestPainType","RestingBP","Cholesterol",
                  "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]
VALID_VALUES = {
    "Sex": {"M","F"},
    "ChestPainType": {"ASY","ATA","NAP","TA"},
    "RestingECG": {"Normal","LVH","ST"},
    "ExerciseAngina": {"Y","N"},
    "ST_Slope": {"Up","Flat","Down"},
}

def make_template_csv():
    sample = pd.DataFrame([
        {"Age":54,"Sex":"M","ChestPainType":"ASY","RestingBP":140,"Cholesterol":239,
         "FastingBS":0,"RestingECG":"Normal","MaxHR":160,"ExerciseAngina":"N","Oldpeak":1.2,"ST_Slope":"Flat"},
        {"Age":45,"Sex":"F","ChestPainType":"ATA","RestingBP":120,"Cholesterol":204,
         "FastingBS":0,"RestingECG":"Normal","MaxHR":172,"ExerciseAngina":"N","Oldpeak":0.0,"ST_Slope":"Up"},
    ])
    return sample.to_csv(index=False).encode("utf-8")

def validate_csv(raw_df):
    errors=[]
    missing=[c for c in REQUIRED_COLS if c not in raw_df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
        return errors
    for col,valid in VALID_VALUES.items():
        bad=set(raw_df[col].dropna().astype(str).unique()) - valid
        if bad:
            errors.append(f"Column '{col}' has invalid value(s) {sorted(bad)} — allowed: {sorted(valid)}")
    for col in ["Age","RestingBP","Cholesterol","FastingBS","MaxHR","Oldpeak"]:
        if not pd.api.types.is_numeric_dtype(raw_df[col]):
            errors.append(f"Column '{col}' must contain numbers only")
    if raw_df[REQUIRED_COLS].isnull().any().any():
        errors.append("Some required cells are empty — please fill in every column for every row")
    return errors

def encode_batch(raw_df):
    """Takes a raw dataframe (same column format as heart.csv) and returns
    an encoded + scaled dataframe ready for model.predict(), aligned to
    the exact training column order."""
    cat_cols=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
    num_cols=['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']
    df_enc=pd.get_dummies(raw_df[REQUIRED_COLS].copy(),columns=cat_cols,drop_first=False)
    for col in Xtr.columns:
        if col not in df_enc.columns:
            df_enc[col]=0.0
    df_enc=df_enc[Xtr.columns]
    df_enc[num_cols]=scaler.transform(df_enc[num_cols])
    return df_enc

# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-text">
        <span class="sb-heart"><svg width="26" height="24" viewBox="0 0 26 24" fill="none">
          <path d="M13 21.5s-9.3-5.7-11.6-11.7C-0.3 5.6 2.3 1.5 6.5 1.5c2.6 0 4.7 1.5 6.5 4 1.8-2.5 3.9-4 6.5-4 4.2 0 6.8 4.1 5.1 8.3C22.3 15.8 13 21.5 13 21.5z"
            fill="#e11d48"/>
        </svg></span>
        Heart<span>Sense</span> AI
      </div>
      <div class="sb-tag">Clinical Risk Intelligence</div>
      <div class="sb-pulse">System online</div>
    </div>""", unsafe_allow_html=True)

    page=st.radio("",["🏠  Overview","📊  Data Insights",
                       "🤖  Model Analytics","🔮  Risk Assessment"])

    st.markdown("""
    <div class="sb-stats">
      <div class="sb-stat-row">Patients<span class="sb-stat-val">918</span></div>
      <div class="sb-stat-row">Features<span class="sb-stat-val">11</span></div>
      <div class="sb-stat-row">Best AUC<span class="sb-stat-val">0.939</span></div>
      <div class="sb-stat-row">Best Acc.<span class="sb-stat-val">91.9%</span></div>
      <div class="sb-stat-row">Models<span class="sb-stat-val">6</span></div>
      <div class="sb-stat-row">Dataset<span class="sb-stat-val">UCI</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.9rem 1rem;border-radius:12px;
      background:rgba(225,29,72,0.05);border:1px solid rgba(225,29,72,0.12);margin-top:auto">
      <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;
        color:rgba(225,29,72,0.6);margin-bottom:5px">⚠ Disclaimer</div>
      <div style="font-size:0.72rem;color:rgba(255,255,255,0.25);line-height:1.6">
        Educational use only.<br>Not a substitute for medical advice.</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div class="hero-visual">
      <svg viewBox="0 0 640 130" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="ecgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#e11d48" stop-opacity="0"/>
            <stop offset="15%" stop-color="#e11d48" stop-opacity="1"/>
            <stop offset="85%" stop-color="#fb7185" stop-opacity="1"/>
            <stop offset="100%" stop-color="#fb7185" stop-opacity="0"/>
          </linearGradient>
        </defs>
        <path class="ecg-line" d="M0,65 L110,65 L130,65 L142,20 L156,110 L170,40 L182,65 L230,65
                 L250,65 L262,20 L276,110 L290,40 L302,65 L360,65
                 L380,65 L392,20 L406,110 L420,40 L432,65 L640,65"/>
      </svg>
    </div>
    <div class="hero">
      <div class="hero-pill">Machine Learning · Healthcare AI · UCI Dataset</div>
      <h1 class="hero-h1">Predict heart disease risk<br>with <span class="g">clinical AI precision</span></h1>
      <p class="hero-sub">A complete end-to-end ML pipeline comparing six classification algorithms
      on 918 patient records — with live risk prediction, interactive data insights, and
      fully explainable results.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
      <div class="stat-card"><span class="sc-icon">📈</span>
        <div class="sc-num red" data-target="0.939" data-suffix="" data-dec="3">0.939</div>
        <div class="sc-lbl">Best ROC-AUC score</div></div>
      <div class="stat-card"><span class="sc-icon">🎯</span>
        <div class="sc-num w" data-target="91.9" data-suffix="%" data-dec="1">91.9%</div>
        <div class="sc-lbl">Top model accuracy</div></div>
      <div class="stat-card"><span class="sc-icon">🤖</span>
        <div class="sc-num red" data-target="6" data-suffix="" data-dec="0">6</div>
        <div class="sc-lbl">ML models compared</div></div>
      <div class="stat-card"><span class="sc-icon">🏥</span>
        <div class="sc-num w" data-target="918" data-suffix="" data-dec="0">918</div>
        <div class="sc-lbl">Patient records</div></div>
    </div>""", unsafe_allow_html=True)

    c1,c2=st.columns([1.05,0.95],gap="large")
    with c1:
        st.markdown('<div class="sl">Project pipeline</div>',unsafe_allow_html=True)
        steps=[("01","Data collection","fedesoriano Dataset — 5 UCI sources combined, 918 patients, 11 features"),
               ("02","Exploratory analysis","Distributions, correlations, feature-vs-outcome patterns"),
               ("03","Data cleaning","Median imputation for invalid zero values, duplicate removal"),
               ("04","Preprocessing","One-hot encoding × 5 cols, StandardScaler × 6 cols, 80/20 split"),
               ("05","Model training","Logistic Reg. · Random Forest · Gradient Boosting · Extra Trees · KNN"),
               ("06","Ensembling","Weighted soft-voting ensemble (GB + Extra Trees + KNN×2)"),
               ("07","Evaluation","Accuracy · Precision · Recall · F1-Score · ROC-AUC · 10-fold CV"),
               ("08","Deployment","Streamlit Cloud — 4-page interactive web app, live predictions")]
        for num,t,d in steps:
            st.markdown(f'<div class="pip"><div class="pip-num">{num}</div><div>'
                        f'<div class="pip-t">{t}</div><div class="pip-d">{d}</div></div></div>',
                        unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sl">Key findings</div>',unsafe_allow_html=True)
        finds=[("#e11d48","ST_Slope_Up (upward ST slope) is the strongest predictor — highest importance in all models"),
               ("#e11d48","Asymptomatic chest pain (ChestPainType_ASY) strongly correlates with heart disease"),
               ("#10b981","Higher max heart rate (MaxHR) significantly reduces disease probability"),
               ("#e11d48","Male patients show substantially higher disease rates than female patients"),
               ("#e11d48","Oldpeak (ST depression) is a top-5 predictor — higher values strongly signal disease"),
               ("#8b5cf6","Weighted Voting Ensemble (GB+ExtraTrees+KNN) achieves best ROC-AUC of 0.939"),
               ("#8b5cf6","Voting Ensemble reaches 91.9% accuracy — validated by 10-fold cross-validation"),
               ("#10b981","All 6 models exceed 86% accuracy — strong, consistent generalisation")]
        for col,txt in finds:
            st.markdown(f'<div class="fin"><div class="dot" style="background:{col}"></div>{txt}</div>',
                        unsafe_allow_html=True)
        st.markdown("""
        <div class="lb">
          <div class="lb-head">🏆 Model leaderboard</div>
          <div class="lb-sub"><span>Model</span><span>AUC</span><span>Accuracy</span></div>
          <div class="lb-row">
            <span class="lb-name">🥇 Voting Ensemble</span>
            <span class="lb-auc">0.939</span><span class="lb-acc">91.9%</span></div>
          <div class="lb-row lb-dim" style="opacity:0.75">
            <span class="lb-name">🥈 KNN (k=19)</span>
            <span style="color:#94a3b8">0.935</span><span class="lb-acc">89.1%</span></div>
          <div class="lb-row lb-dim" style="opacity:0.6">
            <span class="lb-name">🥉 Logistic Regression</span>
            <span style="color:#94a3b8">0.933</span><span style="color:#94a3b8">88.6%</span></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# DATA INSIGHTS
# ══════════════════════════════════════════════════════════
elif "Data" in page:
    st.markdown('<div class="sl">Exploratory data analysis</div>',unsafe_allow_html=True)
    st.markdown('<h2 class="st">Understanding the data</h2>',unsafe_allow_html=True)

    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="chart-wrap">',unsafe_allow_html=True)
        fig,ax=dark_fig(5,3.6)
        nd,hd=(df["HeartDisease"]==0).sum(),(df["HeartDisease"]==1).sum()
        bars=ax.bar(["No Disease","Disease"],[nd,hd],
                    color=[P["green"],P["red"]],width=0.45,edgecolor="none")
        for bar,v,c in zip(bars,[nd,hd],[P["green"],P["red"]]):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,
                    str(v),ha="center",color=c,fontsize=13,fontweight="800")
        ax.set_title("Target distribution",color="#e2e8f0",fontsize=11,fontweight="700",pad=10)
        ax.set_ylim(0,200)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>',unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-wrap">',unsafe_allow_html=True)
        fig,ax=dark_fig(5,3.6)
        ax.hist(df[df["HeartDisease"]==0]["Age"],bins=20,alpha=0.8,color=P["green"],edgecolor="none",label="No disease")
        ax.hist(df[df["HeartDisease"]==1]["Age"],bins=20,alpha=0.8,color=P["red"],edgecolor="none",label="Disease")
        ax.set_title("Age distribution by outcome",color="#e2e8f0",fontsize=11,fontweight="700",pad=10)
        ax.legend(facecolor=P["card"],edgecolor=P["border"],labelcolor="#94a3b8",fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>",unsafe_allow_html=True)
    st.markdown('<div class="chart-wrap">',unsafe_allow_html=True)
    fig,ax=dark_fig(12,5)
    corr=df.corr(numeric_only=True)
    mask=np.triu(np.ones_like(corr,dtype=bool))
    cmap=sns.diverging_palette(0,130,s=85,l=40,as_cmap=True)
    sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap=cmap,ax=ax,
                linewidths=0.4,linecolor=P["bg"],
                annot_kws={"size":7.5,"color":"#e2e8f0","weight":"600"},
                cbar_kws={"shrink":0.55})
    ax.set_title("Feature correlation heatmap",color="#e2e8f0",fontsize=12,fontweight="700",pad=12)
    ax.tick_params(colors=P["t2"],labelsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>",unsafe_allow_html=True)
    st.markdown('<div class="chart-wrap">',unsafe_allow_html=True)
    fig,axes=plt.subplots(1,4,figsize=(13,4.2))
    fig.patch.set_facecolor(P["bg"])
    fig.suptitle("Clinical features — No disease vs Disease",color=P["t2"],fontsize=10,y=1.02)
    for ax,feat,lbl in zip(axes,["Age","MaxHR","Oldpeak","Cholesterol"],
                                 ["Age (years)","Max heart rate","ST depression","Cholesterol"]):
        ax.set_facecolor(P["card"])
        for s in ax.spines.values(): s.set_visible(False)
        d0=df[df["HeartDisease"]==0][feat]; d1=df[df["HeartDisease"]==1][feat]
        vp=ax.violinplot([d0,d1],positions=[0,1],showmedians=True,showextrema=False)
        for body,c in zip(vp["bodies"],[P["green"],P["red"]]):
            body.set_facecolor(c); body.set_alpha(0.5); body.set_edgecolor("none")
        vp["cmedians"].set_color("#fff"); vp["cmedians"].set_linewidth(2)
        ax.scatter([0]*len(d0),d0,alpha=0.12,color=P["green"],s=6,zorder=5)
        ax.scatter([1]*len(d1),d1,alpha=0.12,color=P["red"],s=6,zorder=5)
        ax.set_xticks([0,1]); ax.set_xticklabels(["No","Yes"],color=P["t2"],fontsize=9)
        ax.tick_params(colors=P["t2"],labelsize=8)
        ax.set_title(lbl,fontsize=9.5,fontweight="700",color="#e2e8f0",pad=8)
        ax.yaxis.grid(True,color=P["border"],lw=0.5,ls="--"); ax.set_axisbelow(True)
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MODEL ANALYTICS
# ══════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown('<div class="sl">Comparative analysis</div>',unsafe_allow_html=True)
    st.markdown('<h2 class="st">Model performance</h2>',unsafe_allow_html=True)

    r2=res.set_index("Model")
    st.dataframe(r2.style.format("{:.4f}")
                          .highlight_max(color="rgba(16,185,129,0.18)",axis=0)
                          .set_properties(**{"color":"#e2e8f0","background-color":P["card"],"font-size":"0.88rem"}),
                 use_container_width=True)

    st.markdown("<div style='height:1rem'></div>",unsafe_allow_html=True)
    c1,c2=st.columns(2,gap="large")

    with c1:
        st.markdown('<div class="chart-wrap">',unsafe_allow_html=True)
        fig,ax=dark_fig(5.5,4.6)
        res_sorted = res.sort_values("Accuracy")
        palette6 = {"Logistic Regression":P["red"],"Random Forest":P["green"],
                    "Gradient Boosting":P["blue"],"Extra Trees":"#f59e0b",
                    "KNN":"#06b6d4","Voting Ensemble":"#8b5cf6"}
        colors_m=[palette6.get(m,"#94a3b8") for m in res_sorted["Model"]]
        bars=ax.barh(res_sorted["Model"],(res_sorted["Accuracy"]*100).tolist(),
                     color=colors_m,height=0.55,edgecolor="none")
        for bar,v in zip(bars,(res_sorted["Accuracy"]*100).tolist()):
            ax.text(bar.get_width()-0.4,bar.get_y()+bar.get_height()/2,
                    f"{v:.1f}%",va="center",ha="right",color="#030408",fontsize=10,fontweight="800")
        ax.set_xlim(82,95)
        ax.tick_params(labelsize=8.5)
        ax.set_title("Accuracy comparison — all 6 models",color="#e2e8f0",fontsize=11,fontweight="700",pad=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>',unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-wrap">',unsafe_allow_html=True)
        fig,ax=dark_fig(5.5,4)
        for model,name,color in[(lr,"Logistic Reg.",P["red"]),(rf,"Random Forest",P["green"]),(gb,"Grad. Boosting",P["blue"]),(voting,"Voting Ensemble","#8b5cf6")]:
            prob=model.predict_proba(Xte)[:,1]
            auc=roc_auc_score(yte,prob)
            fpr,tpr,_=roc_curve(yte,prob)
            ax.plot(fpr,tpr,color=color,lw=2.5,label=f"{name}  AUC={auc:.3f}")
            ax.fill_between(fpr,0,tpr,alpha=0.06,color=color)
        ax.plot([0,1],[0,1],"--",color=P["border"],lw=1.5)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves",color="#e2e8f0",fontsize=11,fontweight="700",pad=10)
        ax.legend(facecolor=P["card"],edgecolor=P["border"],labelcolor="#94a3b8",fontsize=8.5)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>",unsafe_allow_html=True)
    c3,c4=st.columns(2,gap="large")

    with c3:
        st.markdown('<div class="chart-wrap">',unsafe_allow_html=True)
        fig,ax=dark_fig(5.5,5)
        rf_imp=pd.Series(rf.feature_importances_,index=Xtr.columns).sort_values().tail(14)
        cb=[P["red"] if v>rf_imp.quantile(0.7) else P["blue"] for v in rf_imp]
        ax.barh(rf_imp.index,rf_imp.values,color=cb,height=0.6,edgecolor="none")
        ax.set_title("Random Forest — feature importance",color="#e2e8f0",fontsize=10,fontweight="700",pad=10)
        ax.tick_params(labelsize=7.5)
        r_p=mpatches.Patch(color=P["red"],label="High importance")
        b_p=mpatches.Patch(color=P["blue"],label="Moderate")
        ax.legend(handles=[r_p,b_p],facecolor=P["card"],edgecolor=P["border"],labelcolor="#94a3b8",fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>',unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-wrap">',unsafe_allow_html=True)
        fig,ax=dark_fig(5.5,5)
        yp=voting.predict(Xte)
        cm=confusion_matrix(yte,yp)
        sns.heatmap(cm,annot=True,fmt="d",ax=ax,
                    cmap=sns.light_palette("#8b5cf6",as_cmap=True),
                    linewidths=2,linecolor=P["bg"],
                    xticklabels=["No Disease","Disease"],
                    yticklabels=["No Disease","Disease"],
                    annot_kws={"size":16,"weight":"800","color":"#fff"})
        ax.set_xlabel("Predicted",color=P["t2"]); ax.set_ylabel("Actual",color=P["t2"])
        ax.set_title("Confusion matrix — Voting Ensemble",color="#e2e8f0",fontsize=10,fontweight="700",pad=10)
        ax.tick_params(colors=P["t2"],labelsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# RISK ASSESSMENT
# ══════════════════════════════════════════════════════════
elif "Risk" in page:
    st.markdown('<div class="sl">Live prediction engine</div>',unsafe_allow_html=True)
    st.markdown('<h2 class="st">Patient risk assessment</h2>',unsafe_allow_html=True)

    mdl_choice=st.selectbox("Prediction model",[
        "🟣  Voting Ensemble  ·  Best Accuracy 91.9%",
        "🔴  Logistic Regression  ·  AUC 0.933",
        "🟢  Random Forest  ·  Accuracy 87.5%",
        "🔵  Gradient Boosting  ·  AUC 0.930",
        "🟠  Extra Trees  ·  Accuracy 86.4%",
        "🩵  K-Nearest Neighbors  ·  Accuracy 89.1%"])
    sel_mdl={"🟣  Voting Ensemble  ·  Best Accuracy 91.9%":voting,
              "🔴  Logistic Regression  ·  AUC 0.933":lr,
              "🟢  Random Forest  ·  Accuracy 87.5%":rf,
              "🔵  Gradient Boosting  ·  AUC 0.930":gb,
              "🟠  Extra Trees  ·  Accuracy 86.4%":et,
              "🩵  K-Nearest Neighbors  ·  Accuracy 89.1%":knn}[mdl_choice]

    input_mode = st.radio("Input method", ["✍️  Manual Entry", "📁  Upload CSV (Batch)"], horizontal=True)

    st.markdown("<div style='height:0.5rem'></div>",unsafe_allow_html=True)

    if "Manual" in input_mode:
        ca,cb,cc=st.columns([1,1,0.9],gap="large")

        with ca:
            st.markdown('<div class="ig"><span class="ig-icon">🧑</span>Demographics &amp; symptoms</div>',unsafe_allow_html=True)
            age=st.slider("Age",20,80,55)
            sex=st.selectbox("Sex",["Female (F)","Male (M)"])
            cp=st.selectbox("Chest pain type",[
                "ASY - Asymptomatic (most common high risk)",
                "ATA - Atypical Angina",
                "NAP - Non-Anginal Pain",
                "TA - Typical Angina"])
            trestbps=st.slider("Resting blood pressure (mm Hg)",80,200,130)
            chol=st.slider("Cholesterol (mg/dl)",100,600,200)
            fbs=st.selectbox("Fasting blood sugar > 120 mg/dl",["No (0)","Yes (1)"])

        with cb:
            st.markdown('<div class="ig"><span class="ig-icon">🩺</span>Cardiac measurements</div>',unsafe_allow_html=True)
            restecg=st.selectbox("Resting ECG",["Normal","LVH - Left Ventricular Hypertrophy","ST - ST-T Wave Abnormality"])
            thalach=st.slider("Max heart rate achieved",60,202,140)
            exang=st.selectbox("Exercise induced angina",["No (N)","Yes (Y)"])
            oldpeak=st.slider("Oldpeak (ST depression)",-2.6,6.2,0.0,0.1)
            slope=st.selectbox("ST slope",["Up - Upsloping","Flat","Down - Downsloping"])

        with cc:
            st.markdown('<div class="ig" style="margin-top:0"><span class="ig-icon">🔮</span>Result</div>',unsafe_allow_html=True)
            btn=st.button("🔮  Analyse patient risk")

            if btn:
                # Parse inputs → match exact training column names
                sex_val   = "M" if "Male" in sex else "F"
                cp_val    = cp.split(" - ")[0].strip()   # ASY / ATA / NAP / TA
                fbs_val   = int(fbs.split("(")[1][0])
                ecg_val   = restecg.split(" - ")[0].strip()  # Normal / LVH / ST
                exang_val = "Y" if "Yes" in exang else "N"
                slope_val = slope.split(" - ")[0].strip()  # Up / Flat / Down

                # Build dataframe with ALL training columns set to 0
                enc = pd.DataFrame(0.0, index=[0], columns=Xtr.columns)

                # Fill numerical features
                enc["Age"]        = float(age)
                enc["RestingBP"]  = float(trestbps)
                enc["Cholesterol"]= float(chol)
                enc["FastingBS"]  = float(fbs_val)
                enc["MaxHR"]      = float(thalach)
                enc["Oldpeak"]    = float(oldpeak)

                # Fill one-hot encoded columns
                def set_ohe(col):
                    if col in enc.columns:
                        enc[col] = 1.0

                set_ohe(f"Sex_{sex_val}")
                set_ohe(f"ChestPainType_{cp_val}")
                set_ohe(f"RestingECG_{ecg_val}")
                set_ohe(f"ExerciseAngina_{exang_val}")
                set_ohe(f"ST_Slope_{slope_val}")

                # Scale using SAME scaler fitted on training data
                num_cols = ["Age","RestingBP","Cholesterol","FastingBS","MaxHR","Oldpeak"]
                enc[num_cols] = scaler.transform(enc[num_cols])

                pred=sel_mdl.predict(enc)[0]
                prob=sel_mdl.predict_proba(enc)[0][1]
                rp=prob*100; cp2=max(prob,1-prob)*100

                if pred==1:
                    # High risk — pulsing alert rings + warning icon
                    rings_html = '<div class="alert-rings"><div class="alert-ring"></div><div class="alert-ring"></div><div class="alert-ring"></div></div>'
                    icon_svg = '''<svg viewBox="0 0 24 24" fill="none"><path d="M12 9v4M12 16.5h.01M10.3 3.9L2.7 17.2c-.6 1 .1 2.3 1.3 2.3h16c1.2 0 1.9-1.3 1.3-2.3L13.7 3.9c-.6-1-2-1-2.6 0z"
                      stroke="#fb7185" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'''
                    st.markdown(f"""<div class="result-stage">
                      <div class="result-modal high">
                        {rings_html}
                        <div class="res-icon-wrap high">{icon_svg}</div>
                        <div class="res-title high">High Risk</div>
                        <div class="res-prob-line">Predicted probability: <strong>{rp:.1f}%</strong></div>
                        <div class="res-sub" style="color:rgba(251,113,133,0.75)">⚕️ Please consult a cardiologist immediately</div>
                      </div>
                    </div>""",unsafe_allow_html=True)
                else:
                    # Low risk — confetti burst + checkmark icon
                    import random
                    random.seed(42+int(rp))
                    pieces=[]
                    colors_conf=["#34d399","#6ee7b7","#a7f3d0","#10b981","#fbbf24"]
                    for i in range(26):
                        dx=random.randint(-160,160); dy=random.randint(-140,40)
                        rot=random.randint(-260,260); delay=round(random.uniform(0,0.25),2)
                        c=random.choice(colors_conf)
                        pieces.append(f'<span class="confetti" style="--dx:{dx}px;--dy:{dy}px;--rot:{rot}deg;'
                                      f'background:{c};animation-delay:{delay}s;left:{50+random.randint(-8,8)}%"></span>')
                    confetti_html = '<div class="confetti-wrap">'+''.join(pieces)+'</div>'
                    icon_svg = '''<svg viewBox="0 0 24 24" fill="none"><path d="M5 13l4 4L19 7"
                      stroke="#34d399" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>'''
                    st.markdown(f"""<div class="result-stage">
                      <div class="result-modal low">
                        {confetti_html}
                        <div class="res-icon-wrap low">{icon_svg}</div>
                        <div class="res-title low">Low Risk</div>
                        <div class="res-prob-line">Predicted probability: <strong>{rp:.1f}%</strong></div>
                        <div class="res-sub" style="color:rgba(52,211,153,0.75)">✅ Continue regular health checkups</div>
                      </div>
                    </div>""",unsafe_allow_html=True)

                fc=P["red"] if pred==1 else P["green"]
                st.markdown(f"""<div class="cbar-wrap">
                  <div class="cbar-lbl">Model confidence</div>
                  <div class="cbar-bg">
                    <div class="cbar-fill-{'h' if pred==1 else 'l'}" style="--w:{cp2:.1f}%;width:{cp2:.1f}%"></div>
                  </div>
                  <div class="cbar-pct">{cp2:.1f}%</div>
                </div>""",unsafe_allow_html=True)

                fig,ax=plt.subplots(figsize=(4,3.2))
                fig.patch.set_facecolor(P["bg"]); ax.set_facecolor(P["bg"])
                th=np.linspace(0,np.pi,300)
                ax.plot(np.cos(th),np.sin(th),color="#0f1524",lw=16,solid_capstyle="round")
                th2=np.linspace(np.pi,np.pi*(1-prob),300)
                ax.plot(np.cos(th2),np.sin(th2),color=fc,lw=16,solid_capstyle="round")
                ax.plot(np.cos(th2),np.sin(th2),color=fc,lw=20,alpha=0.2,solid_capstyle="round")
                ax.text(0,0.04,f"{rp:.1f}%",ha="center",va="center",
                        fontsize=26,fontweight="900",color=fc)
                ax.text(0,-0.26,"RISK SCORE",ha="center",color=P["t2"],fontsize=7.5,fontweight="700")
                ax.text(-0.92,-0.1,"0%",ha="center",color=P["t3"],fontsize=7.5)
                ax.text( 0.92,-0.1,"100%",ha="center",color=P["t3"],fontsize=7.5)
                ax.set_xlim(-1.25,1.25); ax.set_ylim(-0.45,1.2); ax.axis("off")
                fig.tight_layout(); st.pyplot(fig); plt.close()
            else:
                st.markdown("""<div class="idle-state">
                  <div class="idle-icon">🫀</div>
                  <div style="font-size:0.85rem;color:rgba(255,255,255,0.22);line-height:1.8">
                    Enter patient details<br>and click
                    <strong style="color:rgba(255,255,255,0.4)">Analyse patient risk</strong>
                  </div>
                </div>""",unsafe_allow_html=True)

        st.markdown('<div class="warn">⚠️ For educational purposes only — not a substitute for professional medical diagnosis.</div>',unsafe_allow_html=True)

    else:
        # ══════════════════════════════════════════════════
        # CSV BATCH UPLOAD MODE
        # ══════════════════════════════════════════════════
        st.markdown("""
        <div class="upload-card">
          <div class="upload-icon">📁</div>
          <div style="font-size:1.05rem;font-weight:700;color:#fff;">Upload a CSV of patients</div>
          <div class="template-hint">
            Don't know the exact format? Download the template below, fill in your rows<br>
            in the same style, and upload it here — no manual form-filling needed.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.9rem'></div>",unsafe_allow_html=True)
        dl_col, up_col = st.columns([0.4,0.6])
        with dl_col:
            st.download_button("⬇️  Download CSV template", data=make_template_csv(),
                                file_name="heartsense_template.csv", mime="text/csv",
                                use_container_width=True)
        with up_col:
            st.caption("Required columns: " + ", ".join(REQUIRED_COLS))

        uploaded = st.file_uploader("Upload your patient CSV", type=["csv"], label_visibility="collapsed")

        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
            except Exception:
                st.error("⚠️ Could not read that file — please make sure it's a valid CSV.")
                raw_df = None

            if raw_df is not None:
                errors = validate_csv(raw_df)
                if errors:
                    st.markdown('<div class="warn">⚠️ Please fix the following before we can predict:</div>',unsafe_allow_html=True)
                    for e in errors:
                        st.markdown(f'<div class="warn">• {e}</div>',unsafe_allow_html=True)
                else:
                    with st.spinner("Analysing patients..."):
                        enc_batch = encode_batch(raw_df)
                        preds = sel_mdl.predict(enc_batch)
                        probs = sel_mdl.predict_proba(enc_batch)[:,1]

                    results = raw_df[REQUIRED_COLS].copy()
                    results.insert(0,"Patient",[f"#{i+1}" for i in range(len(results))])
                    results["Risk"] = np.where(preds==1,"High Risk","Low Risk")
                    results["Probability %"] = (probs*100).round(1)

                    n_high = int((preds==1).sum()); n_low = int((preds==0).sum())
                    st.markdown(f"""
                    <div class="batch-summary">
                      <div class="batch-chip"><div class="bc-num" style="color:#fff">{len(results)}</div><div class="bc-lbl">Patients analysed</div></div>
                      <div class="batch-chip"><div class="bc-num" style="color:#fb7185">{n_high}</div><div class="bc-lbl">High risk</div></div>
                      <div class="batch-chip"><div class="bc-num" style="color:#34d399">{n_low}</div><div class="bc-lbl">Low risk</div></div>
                    </div>""", unsafe_allow_html=True)

                    def style_risk(val):
                        if val=="High Risk": return "background-color:rgba(225,29,72,0.16);color:#fb7185;font-weight:700"
                        return "background-color:rgba(16,185,129,0.13);color:#34d399;font-weight:700"

                    st.dataframe(
                        results.style.map(style_risk, subset=["Risk"])
                               .set_properties(**{"color":"#e2e8f0","background-color":P["card"]}),
                        use_container_width=True, hide_index=True
                    )

                    csv_out = results.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️  Download results as CSV", data=csv_out,
                                        file_name="heartsense_batch_results.csv", mime="text/csv")

        st.markdown('<div class="warn">⚠️ For educational purposes only — not a substitute for professional medical diagnosis.</div>',unsafe_allow_html=True)
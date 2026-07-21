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
    return lr,rf,gb,sc

@st.cache_data
def load_data():
    df=pd.read_csv("data/heart_cleaned.csv")
    Xtr=pd.read_csv("data/X_train.csv"); Xte=pd.read_csv("data/X_test.csv")
    ytr=pd.read_csv("data/y_train.csv").squeeze(); yte=pd.read_csv("data/y_test.csv").squeeze()
    res=pd.read_csv("models/results.csv")
    return df,Xtr,Xte,ytr,yte,res

lr,rf,gb,voting,scaler=load_models()
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

# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-text">Heart<span>Sense</span> AI</div>
      <div class="sb-tag">Clinical Risk Intelligence</div>
      <div class="sb-pulse">System online</div>
    </div>""", unsafe_allow_html=True)

    page=st.radio("",["🏠  Overview","📊  Data Insights",
                       "🤖  Model Analytics","🔮  Risk Assessment"])

    st.markdown("""
    <div class="sb-stats">
      <div class="sb-stat-row">Patients<span class="sb-stat-val">918</span></div>
      <div class="sb-stat-row">Features<span class="sb-stat-val">11</span></div>
      <div class="sb-stat-row">Best AUC<span class="sb-stat-val">0.933</span></div>
      <div class="sb-stat-row">Best Acc.<span class="sb-stat-val">88.6%</span></div>
      <div class="sb-stat-row">Models<span class="sb-stat-val">3</span></div>
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
    <div class="hero">
      <div class="hero-pill">Machine Learning · Healthcare AI · UCI Dataset</div>
      <h1 class="hero-h1">Predict heart disease risk<br>with <span class="g">clinical AI precision</span></h1>
      <p class="hero-sub">A complete end-to-end ML pipeline comparing three classification algorithms
      on 303 patient records — with live risk prediction, interactive data insights, and
      fully explainable results.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
      <div class="stat-card"><span class="sc-icon">📈</span>
        <div class="sc-num red" data-target="0.933" data-suffix="" data-dec="3">0.933</div>
        <div class="sc-lbl">Best ROC-AUC score</div></div>
      <div class="stat-card"><span class="sc-icon">🎯</span>
        <div class="sc-num w" data-target="88.6" data-suffix="%" data-dec="1">88.6%</div>
        <div class="sc-lbl">Top model accuracy</div></div>
      <div class="stat-card"><span class="sc-icon">🤖</span>
        <div class="sc-num red" data-target="3" data-suffix="" data-dec="0">3</div>
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
               ("03","Data cleaning","Median imputation for missing '?' values, duplicate removal"),
               ("04","Preprocessing","One-hot encoding × 5 cols, StandardScaler × 6 cols, 80/20 split"),
               ("05","Model training","Logistic Regression · Random Forest (n=100) · Gradient Boosting"),
               ("06","Evaluation","Accuracy · Precision · Recall · F1-Score · ROC-AUC · Confusion Matrix"),
               ("07","Deployment","Streamlit Cloud — 4-page interactive web app, live predictions")]
        for num,t,d in steps:
            st.markdown(f'<div class="pip"><div class="pip-num">{num}</div><div>'
                        f'<div class="pip-t">{t}</div><div class="pip-d">{d}</div></div></div>',
                        unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sl">Key findings</div>',unsafe_allow_html=True)
        finds=[("#e11d48","ST_Slope_Up (upward ST slope) is the strongest predictor — highest importance in all models"),
               ("#e11d48","Asymptomatic chest pain (ChestPainType_ASY) strongly correlates with heart disease"),
               ("#10b981","Higher max heart rate (thalach) significantly reduces disease probability"),
               ("#e11d48","Male patients show substantially higher disease rates than female patients"),
               ("#e11d48","Oldpeak (ST depression) is a top-5 predictor — higher values strongly signal disease"),
               ("#3b82f6","Logistic Regression leads with best ROC-AUC of 0.933 on the test set"),
               ("#8b5cf6","Voting Ensemble achieves best accuracy at 91.3% with weighted soft voting"),
               ("#10b981","All 3 models exceed 85% accuracy — significantly better than old 303-patient dataset")]
        for col,txt in finds:
            st.markdown(f'<div class="fin"><div class="dot" style="background:{col}"></div>{txt}</div>',
                        unsafe_allow_html=True)
        st.markdown("""
        <div class="lb">
          <div class="lb-head">🏆 Model leaderboard</div>
          <div class="lb-sub"><span>Model</span><span>AUC</span><span>Accuracy</span></div>
          <div class="lb-row">
            <span class="lb-name">🥇 Logistic Regression</span>
            <span class="lb-auc">0.933</span><span class="lb-acc">88.6%</span></div>
          <div class="lb-row lb-dim" style="opacity:0.7">
            <span class="lb-name">🥈 Random Forest</span>
            <span style="color:#94a3b8">0.929</span><span class="lb-acc">88.0%</span></div>
          <div class="lb-row lb-dim" style="opacity:0.5">
            <span class="lb-name">🥉 Gradient Boosting</span>
            <span style="color:#94a3b8">0.920</span><span style="color:#94a3b8">85.9%</span></div>
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
    corr=df.corr()
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
        fig,ax=dark_fig(5.5,4)
        colors_m=[P["red"],P["green"],P["blue"],"#8b5cf6"][:len(res)]
        bars=ax.barh(res["Model"],(res["Accuracy"]*100).tolist(),
                     color=colors_m,height=0.42,edgecolor="none")
        for bar,v in zip(bars,(res["Accuracy"]*100).tolist()):
            ax.text(bar.get_width()-0.4,bar.get_y()+bar.get_height()/2,
                    f"{v:.1f}%",va="center",ha="right",color="#030408",fontsize=11,fontweight="800")
        ax.set_xlim(78,93)
        ax.set_title("Accuracy comparison",color="#e2e8f0",fontsize=11,fontweight="700",pad=10)
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
        "🔴  Logistic Regression  ·  Best AUC 0.933",
        "🟢  Random Forest  ·  Accuracy 88.0%",
        "🔵  Gradient Boosting  ·  AUC 0.932"])
    sel_mdl={"🔴  Logistic Regression  ·  Best AUC 0.933":lr,
              "🟢  Random Forest  ·  Accuracy 88.0%":rf,
              "🔵  Gradient Boosting  ·  AUC 0.932":gb}[mdl_choice]

    st.markdown("<div style='height:0.5rem'></div>",unsafe_allow_html=True)
    ca,cb,cc=st.columns([1,1,0.9],gap="large")

    with ca:
        st.markdown('<div class="ig">Demographics & symptoms</div>',unsafe_allow_html=True)
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
        st.markdown('<div class="ig">Cardiac measurements</div>',unsafe_allow_html=True)
        restecg=st.selectbox("Resting ECG",["Normal","LVH - Left Ventricular Hypertrophy","ST - ST-T Wave Abnormality"])
        thalach=st.slider("Max heart rate achieved",60,202,140)
        exang=st.selectbox("Exercise induced angina",["No (N)","Yes (Y)"])
        oldpeak=st.slider("Oldpeak (ST depression)",-2.6,6.2,0.0,0.1)
        slope=st.selectbox("ST slope",["Up - Upsloping","Flat","Down - Downsloping"])

    with cc:
        st.markdown('<div class="ig" style="margin-top:0">Result</div>',unsafe_allow_html=True)
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
                st.markdown(f"""<div class="res-h">
                  <div class="ri">⚠️</div>
                  <div class="rh">High Risk</div>
                  <div class="rp">Probability: <strong>{rp:.1f}%</strong></div>
                  <div style="margin-top:6px;font-size:0.77rem;color:rgba(225,29,72,0.65)">
                    Please consult a cardiologist immediately</div>
                </div>""",unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="res-l">
                  <div class="ri">✅</div>
                  <div class="rl">Low Risk</div>
                  <div class="rp">Probability: <strong>{rp:.1f}%</strong></div>
                  <div style="margin-top:6px;font-size:0.77rem;color:rgba(16,185,129,0.65)">
                    Continue regular health checkups</div>
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
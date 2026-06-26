# ============================================================
# HEART DISEASE RISK PREDICTION
# Step 8: Final Report & Model Summary
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)

# ── Load everything ────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()
df      = pd.read_csv("data/heart_cleaned.csv")

with open("models/logistic_regression.pkl", "rb") as f:
    lr = pickle.load(f)
with open("models/random_forest.pkl", "rb") as f:
    rf = pickle.load(f)
with open("models/gradient_boosting.pkl", "rb") as f:
    gb = pickle.load(f)

results = pd.read_csv("models/results.csv")

print("=" * 60)
print("      HEART DISEASE RISK PREDICTION - FINAL REPORT")
print("=" * 60)

# ── 1. Dataset Summary ─────────────────────────────────────
print("\n📋 DATASET SUMMARY")
print(f"   Total Patients     : {len(df)}")
print(f"   Features Used      : {df.shape[1] - 1}")
print(f"   No Heart Disease   : {(df['target']==0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
print(f"   Has Heart Disease  : {(df['target']==1).sum()} ({(df['target']==1).mean()*100:.1f}%)")
print(f"   Training Samples   : {len(X_train)}")
print(f"   Test Samples       : {len(X_test)}")

# ── 2. Model Comparison Table ──────────────────────────────
print("\n📊 MODEL COMPARISON TABLE")
print("=" * 60)
print(results.to_string(index=False))
print("=" * 60)

best_model_name = results.loc[results["ROC-AUC"].idxmax(), "Model"]
best_auc        = results["ROC-AUC"].max()
best_acc        = results.loc[results["ROC-AUC"].idxmax(), "Accuracy"]
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   ROC-AUC  : {best_auc:.4f}")
print(f"   Accuracy : {best_acc*100:.2f}%")

# ── 3. Final Dashboard Plot ────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Heart Disease Risk Prediction — Final Report Dashboard",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot A: Model Accuracy Bar ─────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
colors = ["#e74c3c", "#2ecc71", "#3498db"]
bars = ax1.bar(results["Model"], results["Accuracy"]*100, color=colors, edgecolor="white")
ax1.set_title("Model Accuracy Comparison", fontweight="bold")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(70, 100)
ax1.tick_params(axis='x', rotation=15)
for bar, val in zip(bars, results["Accuracy"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# ── Plot B: ROC-AUC Bar ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(results["Model"], results["ROC-AUC"], color=colors, edgecolor="white")
ax2.set_title("ROC-AUC Comparison", fontweight="bold")
ax2.set_ylabel("ROC-AUC Score")
ax2.set_ylim(0.8, 1.0)
ax2.tick_params(axis='x', rotation=15)
for bar, val in zip(bars2, results["ROC-AUC"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# ── Plot C: All Metrics Heatmap ────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
metrics_df = results.set_index("Model")[["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]]
sns.heatmap(metrics_df, annot=True, fmt=".3f", cmap="YlGn",
            ax=ax3, linewidths=0.5, vmin=0.7, vmax=1.0)
ax3.set_title("All Metrics Heatmap", fontweight="bold")
ax3.tick_params(axis='x', rotation=20)

# ── Plot D: ROC Curves ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
model_list = [("Logistic Regression", lr, "#e74c3c"),
              ("Random Forest",        rf, "#2ecc71"),
              ("Gradient Boosting",    gb, "#3498db")]
for name, model, color in model_list:
    prob = model.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, prob)
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax4.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.2f})")
ax4.plot([0,1],[0,1], "k--", alpha=0.5)
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.set_title("ROC Curves - All Models", fontweight="bold")
ax4.legend(loc="lower right")

# ── Plot E: Feature Importance (RF) ───────────────────────
ax5 = fig.add_subplot(gs[1, 2])
rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns)
rf_imp.sort_values().tail(10).plot(kind="barh", color="#2ecc71", ax=ax5)
ax5.set_title("Top 10 Features (Random Forest)", fontweight="bold")
ax5.set_xlabel("Importance")

plt.savefig("plots/15_final_dashboard.png", dpi=120, bbox_inches="tight")
plt.show()
print("\n✅ Saved: plots/15_final_dashboard.png")

# ── 4. Conclusions ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  📝 CONCLUSIONS")
print("=" * 60)
print("""
  1. Dataset: 303 patients from the UCI Heart Disease Dataset.
     The target is fairly balanced (54% no disease, 46% disease).

  2. Key Risk Factors Identified:
     • thal_7.0  — Reversible thalassemia defect (strongest predictor)
     • cp_4.0    — Asymptomatic chest pain (high risk indicator)
     • thalach   — Lower max heart rate = higher disease risk
     • oldpeak   — Higher ST depression = higher disease risk
     • ca        — More major vessels blocked = higher risk
     • sex_1.0   — Males at higher risk than females

  3. Model Performance:
     • All 3 models achieved >85% accuracy
     • Logistic Regression: Best ROC-AUC (0.96) — excellent!
     • Random Forest: Best accuracy (88.5%)
     • Gradient Boosting: Strong balanced performance

  4. Best Model Recommendation:
     → Logistic Regression for interpretability + high AUC
     → Random Forest for best accuracy

  5. Limitations:
     • Small dataset (303 samples)
     • Only Cleveland subset of UCI dataset used
     • More data could improve model robustness
""")

print("=" * 60)
print("✅ Step 8 Complete! Final report generated.")
print("   Dashboard saved to: plots/15_final_dashboard.png")
print("=" * 60)

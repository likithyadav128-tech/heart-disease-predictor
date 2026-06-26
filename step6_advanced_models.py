# ============================================================
# HEART DISEASE RISK PREDICTION
# Step 6: Advanced Models - Random Forest & Gradient Boosting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

# ── Load preprocessed data ─────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print("=" * 55)
print("   ADVANCED MODELS: RF & GRADIENT BOOSTING")
print("=" * 55)

# ── Helper: Evaluate any model ─────────────────────────────
def evaluate_model(name, model, X_test, y_test):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_prob)

    print(f"\n{'='*55}")
    print(f"  {name} RESULTS")
    print(f"{'='*55}")
    print(f"  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["No Disease", "Disease"]))

    return y_pred, y_pred_prob, acc, prec, rec, f1, auc

# ══════════════════════════════════════════════════════════
# MODEL 1: RANDOM FOREST
# ══════════════════════════════════════════════════════════
print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=5,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("   ✅ Random Forest trained!")

rf_pred, rf_prob, rf_acc, rf_prec, rf_rec, rf_f1, rf_auc = \
    evaluate_model("RANDOM FOREST", rf_model, X_test, y_test)

# Confusion Matrix - RF
cm_rf = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Random Forest - Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("plots/08_rf_confusion_matrix.png")
plt.show()
print("✅ Saved: plots/08_rf_confusion_matrix.png")

# Save RF model
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("✅ Model saved: models/random_forest.pkl")

# ══════════════════════════════════════════════════════════
# MODEL 2: GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════
print("\n⚡ Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1,
    max_depth=3, random_state=42
)
gb_model.fit(X_train, y_train)
print("   ✅ Gradient Boosting trained!")

gb_pred, gb_prob, gb_acc, gb_prec, gb_rec, gb_f1, gb_auc = \
    evaluate_model("GRADIENT BOOSTING", gb_model, X_test, y_test)

# Confusion Matrix - GB
cm_gb = confusion_matrix(y_test, gb_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_gb, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Gradient Boosting - Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("plots/09_gb_confusion_matrix.png")
plt.show()
print("✅ Saved: plots/09_gb_confusion_matrix.png")

# Save GB model
with open("models/gradient_boosting.pkl", "wb") as f:
    pickle.dump(gb_model, f)
print("✅ Model saved: models/gradient_boosting.pkl")

# ══════════════════════════════════════════════════════════
# COMBINED ROC CURVE (All 3 models)
# ══════════════════════════════════════════════════════════
# Load LR model for comparison
with open("models/logistic_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)
lr_prob = lr_model.predict_proba(X_test)[:, 1]
lr_auc  = roc_auc_score(y_test, lr_prob)

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, color="#e74c3c", lw=2,
         label=f"Logistic Regression (AUC = {lr_auc:.2f})")
plt.plot(fpr_rf, tpr_rf, color="#2ecc71", lw=2,
         label=f"Random Forest      (AUC = {rf_auc:.2f})")
plt.plot(fpr_gb, tpr_gb, color="#3498db", lw=2,
         label=f"Gradient Boosting  (AUC = {gb_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - All Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/10_roc_comparison.png")
plt.show()
print("✅ Saved: plots/10_roc_comparison.png")

# ── Update results CSV ─────────────────────────────────────
results = pd.read_csv("models/results.csv")
new_rows = pd.DataFrame({
    "Model":     ["Random Forest", "Gradient Boosting"],
    "Accuracy":  [round(rf_acc, 4), round(gb_acc, 4)],
    "Precision": [round(rf_prec, 4), round(gb_prec, 4)],
    "Recall":    [round(rf_rec, 4), round(gb_rec, 4)],
    "F1 Score":  [round(rf_f1, 4), round(gb_f1, 4)],
    "ROC-AUC":   [round(rf_auc, 4), round(gb_auc, 4)],
})
results = pd.concat([results, new_rows], ignore_index=True)
results.to_csv("models/results.csv", index=False)

print("\n📊 ALL MODELS COMPARISON:")
print(results.to_string(index=False))

print("\n" + "=" * 55)
print("✅ Step 6 Complete! All models trained & compared.")
print("=" * 55)

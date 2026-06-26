# ============================================================
# HEART DISEASE RISK PREDICTION
# Step 5: Baseline Model - Logistic Regression
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

# ── Load preprocessed data ─────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print("=" * 55)
print("     BASELINE MODEL: LOGISTIC REGRESSION")
print("=" * 55)
print(f"\n📋 Training set: {X_train.shape}")
print(f"📋 Test set:     {X_test.shape}")

# ── Train Model ────────────────────────────────────────────
print("\n🚀 Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("   ✅ Model trained!")

# ── Predictions ────────────────────────────────────────────
y_pred      = lr_model.predict(X_test)
y_pred_prob = lr_model.predict_proba(X_test)[:, 1]

# ── Evaluation Metrics ─────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print("\n" + "=" * 55)
print("         LOGISTIC REGRESSION RESULTS")
print("=" * 55)
print(f"  Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print("=" * 55)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["No Disease", "Disease"]))

# ── Confusion Matrix Plot ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Logistic Regression - Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("plots/06_lr_confusion_matrix.png")
plt.show()
print("✅ Saved: plots/06_lr_confusion_matrix.png")

# ── ROC Curve ─────────────────────────────────────────────
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#e74c3c", lw=2,
         label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/07_lr_roc_curve.png")
plt.show()
print("✅ Saved: plots/07_lr_roc_curve.png")

# ── Save Model ─────────────────────────────────────────────
with open("models/logistic_regression.pkl", "wb") as f:
    pickle.dump(lr_model, f)
print("\n✅ Model saved: models/logistic_regression.pkl")

# ── Save results for comparison later ─────────────────────
results = {
    "Model":     ["Logistic Regression"],
    "Accuracy":  [round(accuracy, 4)],
    "Precision": [round(precision, 4)],
    "Recall":    [round(recall, 4)],
    "F1 Score":  [round(f1, 4)],
    "ROC-AUC":   [round(roc_auc, 4)],
}
pd.DataFrame(results).to_csv("models/results.csv", index=False)
print("✅ Results saved: models/results.csv")

print("\n" + "=" * 55)
print("✅ Step 5 Complete! Baseline model trained & evaluated.")
print("=" * 55)

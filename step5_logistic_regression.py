# ============================================================
# Step 5: Logistic Regression (Baseline)
# ============================================================
import pandas as pd, numpy as np, pickle, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)

X_train=pd.read_csv("data/X_train.csv"); X_test=pd.read_csv("data/X_test.csv")
y_train=pd.read_csv("data/y_train.csv").squeeze(); y_test=pd.read_csv("data/y_test.csv").squeeze()

print("=" * 60)
print("   BASELINE MODEL: LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:,1]

acc=accuracy_score(y_test,y_pred); prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred);   f1=f1_score(y_test,y_pred)
auc=roc_auc_score(y_test,y_prob)

print(f"\n  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  ROC-AUC  : {auc:.4f}")
print(f"\n{classification_report(y_test,y_pred,target_names=['No Disease','Disease'])}")

# Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=["No Disease","Disease"],yticklabels=["No Disease","Disease"])
plt.title("Logistic Regression — Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout(); plt.savefig("plots/06_lr_confusion_matrix.png"); plt.show()

# ROC Curve
fpr,tpr,_=roc_curve(y_test,y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr,tpr,color="#e74c3c",lw=2,label=f"LR (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve — Logistic Regression")
plt.legend(); plt.tight_layout()
plt.savefig("plots/07_lr_roc_curve.png"); plt.show()

with open("models/logistic_regression.pkl","wb") as f: pickle.dump(lr,f)
pd.DataFrame({"Model":["Logistic Regression"],"Accuracy":[round(acc,4)],
    "Precision":[round(prec,4)],"Recall":[round(rec,4)],
    "F1 Score":[round(f1,4)],"ROC-AUC":[round(auc,4)]}).to_csv("models/results.csv",index=False)

print("✅ Step 5 Complete!")
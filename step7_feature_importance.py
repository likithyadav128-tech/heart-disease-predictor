# ============================================================
# Step 7: Feature Importance
# ============================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, pickle

X_train=pd.read_csv("data/X_train.csv")
with open("models/random_forest.pkl","rb") as f: rf=pickle.load(f)
with open("models/gradient_boosting.pkl","rb") as f: gb=pickle.load(f)
with open("models/logistic_regression.pkl","rb") as f: lr=pickle.load(f)

print("=" * 55)
print("  FEATURE IMPORTANCE & INTERPRETATION")
print("=" * 55)

features=X_train.columns.tolist()

# RF Importance
rf_imp=pd.Series(rf.feature_importances_,index=features).sort_values(ascending=False)
print("\n🌲 Top 10 Features (Random Forest):")
for i,(f,v) in enumerate(rf_imp.head(10).items(),1):
    print(f"  {i:2}. {f:30s} {v:.4f}")

plt.figure(figsize=(10,6))
rf_imp.head(15).sort_values().plot(kind="barh",color="#2ecc71")
plt.title("Random Forest — Top 15 Feature Importances")
plt.tight_layout(); plt.savefig("plots/11_rf_feature_importance.png"); plt.show()

# GB Importance
gb_imp=pd.Series(gb.feature_importances_,index=features).sort_values(ascending=False)
plt.figure(figsize=(10,6))
gb_imp.head(15).sort_values().plot(kind="barh",color="#3498db")
plt.title("Gradient Boosting — Top 15 Feature Importances")
plt.tight_layout(); plt.savefig("plots/12_gb_feature_importance.png"); plt.show()

# LR Coefficients
lr_coef=pd.Series(lr.coef_[0],index=features)
plt.figure(figsize=(10,6))
colors=["#e74c3c" if c>0 else "#3498db" for c in lr_coef.sort_values()]
lr_coef.sort_values().plot(kind="barh",color=colors)
plt.axvline(0,color="black",linewidth=0.8,linestyle="--")
plt.title("Logistic Regression — Feature Coefficients")
plt.tight_layout(); plt.savefig("plots/13_lr_coefficients.png"); plt.show()

print("\n✅ Step 7 Complete!")
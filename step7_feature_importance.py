# ============================================================
# HEART DISEASE RISK PREDICTION
# Step 7: Feature Importance & Interpretation
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ── Load models and data ───────────────────────────────────
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_test  = pd.read_csv("data/y_test.csv").squeeze()

with open("models/logistic_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open("models/random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("models/gradient_boosting.pkl", "rb") as f:
    gb_model = pickle.load(f)

feature_names = X_train.columns.tolist()

print("=" * 55)
print("     FEATURE IMPORTANCE & INTERPRETATION")
print("=" * 55)

sns.set_style("whitegrid")

# ── 1. Logistic Regression Coefficients ───────────────────
print("\n📊 Logistic Regression - Feature Coefficients:")
lr_coef = pd.Series(lr_model.coef_[0], index=feature_names)
lr_coef_sorted = lr_coef.abs().sort_values(ascending=False)

print(lr_coef.sort_values(ascending=False).to_string())

plt.figure(figsize=(10, 6))
colors = ["#e74c3c" if c > 0 else "#3498db" for c in lr_coef.sort_values()]
lr_coef.sort_values().plot(kind="barh", color=colors)
plt.title("Logistic Regression - Feature Coefficients\n(Red = increases risk, Blue = decreases risk)")
plt.xlabel("Coefficient Value")
plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig("plots/11_lr_coefficients.png")
plt.show()
print("✅ Saved: plots/11_lr_coefficients.png")

# ── 2. Random Forest Feature Importance ───────────────────
print("\n📊 Random Forest - Feature Importance:")
rf_importance = pd.Series(rf_model.feature_importances_, index=feature_names)
rf_importance_sorted = rf_importance.sort_values(ascending=False)
print(rf_importance_sorted.to_string())

plt.figure(figsize=(10, 6))
rf_importance_sorted.head(15).sort_values().plot(
    kind="barh", color="#2ecc71")
plt.title("Random Forest - Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plots/12_rf_feature_importance.png")
plt.show()
print("✅ Saved: plots/12_rf_feature_importance.png")

# ── 3. Gradient Boosting Feature Importance ───────────────
print("\n📊 Gradient Boosting - Feature Importance:")
gb_importance = pd.Series(gb_model.feature_importances_, index=feature_names)
gb_importance_sorted = gb_importance.sort_values(ascending=False)
print(gb_importance_sorted.to_string())

plt.figure(figsize=(10, 6))
gb_importance_sorted.head(15).sort_values().plot(
    kind="barh", color="#3498db")
plt.title("Gradient Boosting - Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plots/13_gb_feature_importance.png")
plt.show()
print("✅ Saved: plots/13_gb_feature_importance.png")

# ── 4. Combined Top Features Comparison ───────────────────
top_features = rf_importance_sorted.head(10).index.tolist()

comparison = pd.DataFrame({
    "Random Forest":     rf_importance[top_features],
    "Gradient Boosting": gb_importance[top_features],
})
comparison = comparison.sort_values("Random Forest", ascending=True)

plt.figure(figsize=(10, 7))
comparison.plot(kind="barh", color=["#2ecc71", "#3498db"],
                ax=plt.gca())
plt.title("Top 10 Features - RF vs Gradient Boosting Comparison")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plots/14_feature_comparison.png")
plt.show()
print("✅ Saved: plots/14_feature_comparison.png")

# ── 5. Summary: Most Important Features ───────────────────
print("\n" + "=" * 55)
print("   🏆 TOP 5 MOST IMPORTANT FEATURES (Random Forest)")
print("=" * 55)
for i, (feat, score) in enumerate(rf_importance_sorted.head(5).items(), 1):
    print(f"   {i}. {feat:30s} → {score:.4f}")

print("\n💡 INSIGHTS:")
print("   • Features with high importance strongly predict heart disease")
print("   • thalach (max heart rate) is typically a top predictor")
print("   • cp (chest pain type) is highly correlated with disease")
print("   • ca (major vessels) and thal are strong indicators")

print("\n" + "=" * 55)
print("✅ Step 7 Complete! Feature importance analyzed.")
print("=" * 55)

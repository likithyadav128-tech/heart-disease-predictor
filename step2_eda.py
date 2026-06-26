# ============================================================
# HEART DISEASE RISK PREDICTION
# Step 2: Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/heart.csv")

print("=" * 55)
print("        EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 55)

# ── 1. Basic Info ──────────────────────────────────────────
print("\n📋 DATASET SHAPE:", df.shape)
print("\n📋 COLUMN NAMES & DATA TYPES:")
print(df.dtypes)

print("\n📋 FIRST 5 ROWS:")
print(df.head())

print("\n📋 SUMMARY STATISTICS:")
print(df.describe())

# ── 2. Missing Values ──────────────────────────────────────
print("\n❓ MISSING VALUES PER COLUMN:")
print(df.isnull().sum())

# Check for '?' values (UCI dataset uses '?' for missing)
print("\n❓ '?' VALUES PER COLUMN (UCI-style missing):")
for col in df.columns:
    count = (df[col].astype(str) == '?').sum()
    if count > 0:
        print(f"   {col}: {count} missing")

# ── 3. Target Distribution ─────────────────────────────────
print("\n🎯 TARGET DISTRIBUTION:")
print(df["target"].value_counts())
print(f"   No Disease (0): {(df['target']==0).sum()} patients")
print(f"   Disease    (1): {(df['target']==1).sum()} patients")

# ── 4. PLOTS ───────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# Plot 1: Target Distribution
plt.figure(figsize=(5, 4))
sns.countplot(x="target", data=df, palette=["#2ecc71", "#e74c3c"])
plt.title("Target Distribution\n(0 = No Disease, 1 = Disease)")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Disease", "Disease"])
plt.tight_layout()
plt.savefig("plots/01_target_distribution.png")
plt.show()
print("✅ Saved: plots/01_target_distribution.png")

# Plot 2: Age Distribution by Target
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="age", hue="target", bins=20,
             palette=["#2ecc71", "#e74c3c"], kde=True)
plt.title("Age Distribution by Heart Disease")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("plots/02_age_distribution.png")
plt.show()
print("✅ Saved: plots/02_age_distribution.png")

# Plot 3: Correlation Heatmap
plt.figure(figsize=(10, 8))
# Convert to numeric to handle any '?' values
df_numeric = df.apply(pd.to_numeric, errors='coerce')
corr = df_numeric.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/03_correlation_heatmap.png")
plt.show()
print("✅ Saved: plots/03_correlation_heatmap.png")

# Plot 4: Box plots for key features
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = ["age", "trestbps", "chol", "thalach"]
titles   = ["Age", "Resting Blood Pressure", "Cholesterol", "Max Heart Rate"]
colors   = [["#2ecc71", "#e74c3c"]] * 4

for ax, feat, title in zip(axes.flatten(), features, titles):
    sns.boxplot(x="target", y=feat, data=df, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title(f"{title} vs Heart Disease")
    ax.set_xlabel("Heart Disease (0=No, 1=Yes)")
    ax.set_ylabel(title)

plt.tight_layout()
plt.savefig("plots/04_boxplots.png")
plt.show()
print("✅ Saved: plots/04_boxplots.png")

# Plot 5: Sex vs Heart Disease
plt.figure(figsize=(6, 4))
sns.countplot(x="sex", hue="target", data=df, palette=["#2ecc71", "#e74c3c"])
plt.title("Heart Disease by Sex\n(0=Female, 1=Male)")
plt.xlabel("Sex")
plt.xticks([0, 1], ["Female", "Male"])
plt.legend(title="Heart Disease", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig("plots/05_sex_vs_disease.png")
plt.show()
print("✅ Saved: plots/05_sex_vs_disease.png")

print("\n" + "=" * 55)
print("✅ Step 2 Complete! All EDA plots saved in plots/")
print("=" * 55)

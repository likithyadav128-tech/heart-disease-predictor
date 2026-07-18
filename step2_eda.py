# ============================================================
# Step 2: Exploratory Data Analysis (EDA)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/heart.csv")
sns.set_style("whitegrid")

print("=" * 60)
print("        EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"\n📋 Shape: {df.shape}")
print(f"\n📊 Summary:\n{df.describe()}")
print(f"\n🎯 Target:\n{df['HeartDisease'].value_counts()}")

# Plot 1: Target Distribution
plt.figure(figsize=(5,4))
counts = df['HeartDisease'].value_counts()
sns.barplot(x=["No Disease","Disease"], y=counts.values, palette=["#2ecc71","#e74c3c"])
for i,v in enumerate(counts.values):
    plt.text(i, v+3, str(v), ha='center', fontweight='bold', fontsize=12)
plt.title("Target Distribution"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig("plots/01_target_distribution.png"); plt.show()
print("✅ plots/01_target_distribution.png")

# Plot 2: Age Distribution
plt.figure(figsize=(8,4))
sns.histplot(data=df, x="Age", hue="HeartDisease", bins=25,
             palette=["#2ecc71","#e74c3c"], kde=True)
plt.title("Age Distribution by Heart Disease")
plt.tight_layout(); plt.savefig("plots/02_age_distribution.png"); plt.show()
print("✅ plots/02_age_distribution.png")

# Plot 3: Categorical Features vs Target
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
cat_cols = ['Sex','ChestPainType','RestingECG','ST_Slope']
for ax, col in zip(axes, cat_cols):
    ct = pd.crosstab(df[col], df['HeartDisease'])
    ct.plot(kind='bar', ax=ax, color=["#2ecc71","#e74c3c"], edgecolor='none', legend=False)
    ax.set_title(f"{col} vs Disease"); ax.set_xlabel(""); ax.tick_params(axis='x', rotation=30)
fig.suptitle("Categorical Features vs Heart Disease", y=1.02)
plt.tight_layout(); plt.savefig("plots/03_categorical_features.png"); plt.show()
print("✅ plots/03_categorical_features.png")

# Plot 4: Correlation Heatmap (numeric only)
plt.figure(figsize=(8,6))
num_df = df.select_dtypes(include=[np.number])
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True)
plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig("plots/04_correlation_heatmap.png"); plt.show()
print("✅ plots/04_correlation_heatmap.png")

# Plot 5: Box plots
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
num_feats = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
for ax, feat in zip(axes.flatten(), num_feats):
    sns.boxplot(x='HeartDisease', y=feat, data=df,
                palette=["#2ecc71","#e74c3c"], ax=ax)
    ax.set_xlabel("Heart Disease (0=No, 1=Yes)")
axes.flatten()[-1].axis('off')
plt.tight_layout(); plt.savefig("plots/05_boxplots.png"); plt.show()
print("✅ plots/05_boxplots.png")

print("\n✅ Step 2 Complete! All EDA plots saved.")
# ============================================================
# HEART DISEASE RISK PREDICTION
# Step 4: Feature Preprocessing
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load cleaned dataset
df = pd.read_csv("data/heart_cleaned.csv")

print("=" * 55)
print("         FEATURE PREPROCESSING")
print("=" * 55)

print(f"\n📋 Loaded cleaned dataset: {df.shape}")

# ── 1. Define Features & Target ────────────────────────────
X = df.drop("target", axis=1)
y = df["target"]

print(f"\n🎯 Features (X): {X.shape[1]} columns")
print(f"   {list(X.columns)}")
print(f"\n🎯 Target (y): {y.shape[0]} rows")
print(f"   Classes: {y.unique()}")

# ── 2. Identify Categorical vs Numerical Columns ───────────
# Categorical: sex, cp, fbs, restecg, exang, slope, ca, thal
# Numerical:   age, trestbps, chol, thalach, oldpeak

categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
numerical_cols   = ["age", "trestbps", "chol", "thalach", "oldpeak"]

print(f"\n📊 Categorical columns: {categorical_cols}")
print(f"📊 Numerical columns:   {numerical_cols}")

# ── 3. One-Hot Encode Categorical Columns ──────────────────
print("\n🛠️  One-hot encoding categorical columns...")
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"   Shape before encoding: {X.shape}")
print(f"   Shape after encoding:  {X_encoded.shape}")
print(f"   New columns: {list(X_encoded.columns)}")

# ── 4. Scale Numerical Columns ─────────────────────────────
print("\n🛠️  Scaling numerical columns with StandardScaler...")
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
print("   ✅ Numerical features scaled (mean=0, std=1)")

print("\n📋 Preprocessed feature preview:")
print(X_encoded.head())

# ── 5. Train-Test Split ────────────────────────────────────
print("\n🛠️  Splitting into train (80%) and test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✅ Training set:   {X_train.shape[0]} samples")
print(f"   ✅ Test set:       {X_test.shape[0]} samples")
print(f"\n   Train target distribution:\n{y_train.value_counts().to_string()}")
print(f"\n   Test target distribution:\n{y_test.value_counts().to_string()}")

# ── 6. Save Preprocessed Data ──────────────────────────────
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv",  index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",  index=False)

# Save scaler for later use
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Saved:")
print("   data/X_train.csv, data/X_test.csv")
print("   data/y_train.csv, data/y_test.csv")
print("   models/scaler.pkl")

print("\n" + "=" * 55)
print("✅ Step 4 Complete! Data is preprocessed and split.")
print("=" * 55)

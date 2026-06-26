# ============================================================
# HEART DISEASE RISK PREDICTION
# Step 3: Data Cleaning
# ============================================================

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/heart.csv")

print("=" * 55)
print("            DATA CLEANING")
print("=" * 55)

print(f"\n📋 Original shape: {df.shape}")

# ── 1. Check for '?' missing values (UCI dataset style) ────
print("\n🔍 Checking for '?' values (UCI-style missing data)...")
question_mark_counts = {}
for col in df.columns:
    count = (df[col].astype(str) == '?').sum()
    if count > 0:
        question_mark_counts[col] = count
        print(f"   ⚠️  Column '{col}': {count} missing values ('?')")

if not question_mark_counts:
    print("   ✅ No '?' values found!")

# ── 2. Replace '?' with NaN ────────────────────────────────
df.replace('?', np.nan, inplace=True)

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\n🔍 NaN values after replacing '?':")
print(df.isnull().sum())

# ── 3. Handle Missing Values ───────────────────────────────
print("\n🛠️  Handling missing values with median imputation...")
missing_before = df.isnull().sum().sum()

for col in df.columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   ✅ '{col}': filled {df[col].isnull().sum()} NaN → median ({median_val})")

missing_after = df.isnull().sum().sum()
print(f"\n   Missing values before: {missing_before}")
print(f"   Missing values after:  {missing_after}")

# ── 4. Check & Remove Duplicates ───────────────────────────
print(f"\n🔍 Checking for duplicate rows...")
duplicates = df.duplicated().sum()
print(f"   Duplicate rows found: {duplicates}")

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"   ✅ Duplicates removed. New shape: {df.shape}")
else:
    print("   ✅ No duplicates found!")

# ── 5. Check Data Types ────────────────────────────────────
print("\n🔍 Data types after cleaning:")
print(df.dtypes)

# ── 6. Check Value Ranges ──────────────────────────────────
print("\n🔍 Checking value ranges for anomalies...")

checks = {
    "age":      (20, 100),
    "trestbps": (80, 220),
    "chol":     (100, 600),
    "thalach":  (60, 220),
    "oldpeak":  (0, 10),
}

for col, (low, high) in checks.items():
    out = df[(df[col] < low) | (df[col] > high)]
    if len(out) > 0:
        print(f"   ⚠️  '{col}': {len(out)} values outside [{low}, {high}]")
    else:
        print(f"   ✅ '{col}': all values in normal range [{low}, {high}]")

# ── 7. Final Cleaned Dataset ───────────────────────────────
print(f"\n📋 Final cleaned shape: {df.shape}")
print("\n📋 Final dataset preview:")
print(df.head())

# Save cleaned dataset
df.to_csv("data/heart_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved to: data/heart_cleaned.csv")

print("\n" + "=" * 55)
print("✅ Step 3 Complete! Data is clean and ready.")
print("=" * 55)

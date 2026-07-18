# ============================================================
# Step 3: Data Cleaning
# ============================================================
import pandas as pd
import numpy as np

df = pd.read_csv("data/heart.csv")
print("=" * 60)
print("            DATA CLEANING")
print("=" * 60)
print(f"\n📋 Original shape: {df.shape}")
print(f"\n❓ Missing values:\n{df.isnull().sum()}")

# Fix Cholesterol = 0 (medically impossible)
chol_zero = (df['Cholesterol'] == 0).sum()
print(f"\n⚠️  Cholesterol = 0: {chol_zero} rows — replacing with median")
median_chol = df[df['Cholesterol'] > 0]['Cholesterol'].median()
df['Cholesterol'] = df['Cholesterol'].where(df['Cholesterol'] > 0, median_chol)
print(f"   ✅ Replaced with median: {median_chol}")

# Fix RestingBP = 0
bp_zero = (df['RestingBP'] == 0).sum()
if bp_zero > 0:
    print(f"\n⚠️  RestingBP = 0: {bp_zero} rows — replacing with median")
    median_bp = df[df['RestingBP'] > 0]['RestingBP'].median()
    df['RestingBP'] = df['RestingBP'].where(df['RestingBP'] > 0, median_bp)
    print(f"   ✅ Replaced with median: {median_bp}")

# Duplicates
dupes = df.duplicated().sum()
print(f"\n🔍 Duplicates found: {dupes}")
if dupes > 0:
    df = df.drop_duplicates()

print(f"\n✅ Final NaN count: {df.isnull().sum().sum()}")
print(f"📋 Final shape: {df.shape}")

df.to_csv("data/heart_cleaned.csv", index=False)
print("✅ Saved: data/heart_cleaned.csv")
print("✅ Step 3 Complete!")
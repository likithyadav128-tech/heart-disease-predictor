# ============================================================
# HEART DISEASE RISK PREDICTION - NEW DATASET
# Step 1: Setup & Dataset Overview
# ============================================================
import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/heart.csv")

print("=" * 60)
print("   HEART DISEASE RISK PREDICTION - NEW DATASET")
print("=" * 60)
print(f"\n📋 Shape       : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"✅ Missing vals : {df.isnull().sum().sum()}")
print(f"\n📋 Columns:\n{list(df.columns)}")
print(f"\n📋 Data types:\n{df.dtypes}")
print(f"\n📋 First 5 rows:\n{df.head()}")
print(f"\n🎯 Target distribution:\n{df['HeartDisease'].value_counts()}")
print(f"   No Disease (0): {(df['HeartDisease']==0).sum()}")
print(f"   Disease    (1): {(df['HeartDisease']==1).sum()}")
print("\n✅ Step 1 Complete!")
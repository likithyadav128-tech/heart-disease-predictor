# ============================================================
# Step 4: Feature Preprocessing
# ============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("data/heart_cleaned.csv")
print("=" * 60)
print("         FEATURE PREPROCESSING")
print("=" * 60)
print(f"\n📋 Loaded: {df.shape}")

# Features & Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Categorical columns
cat_cols = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
num_cols = ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']

print(f"\n📊 Categorical: {cat_cols}")
print(f"📊 Numerical:   {num_cols}")

# One-hot encode
print("\n🛠️  One-hot encoding categorical columns...")
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
print(f"   Before: {X.shape} → After: {X_enc.shape}")
print(f"   Columns: {list(X_enc.columns)}")

# Scale numerical
print("\n🛠️  Scaling numerical columns...")
scaler = StandardScaler()
X_enc[num_cols] = scaler.fit_transform(X_enc[num_cols])
print("   ✅ StandardScaler applied")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"   Train target: {y_train.value_counts().to_dict()}")
print(f"   Test target:  {y_test.value_counts().to_dict()}")

# Save
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv",   index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",   index=False)

with open("models/scaler.pkl","wb") as f: pickle.dump(scaler, f)
print("\n✅ Saved all splits + scaler")
print("✅ Step 4 Complete!")
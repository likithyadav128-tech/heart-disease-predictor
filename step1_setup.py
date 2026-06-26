# ============================================================
# HEART DISEASE RISK PREDICTION
# Step 1: Project Setup & Dataset Download
# ============================================================

import urllib.request
import os

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("✅ Folders created: data/, plots/, models/")

# Download UCI Heart Disease Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
save_path = "data/heart.csv"

print("\n📥 Downloading UCI Heart Disease Dataset...")
urllib.request.urlretrieve(url, save_path)
print(f"✅ Dataset saved to: {save_path}")

# Add column names (UCI dataset has no header)
import pandas as pd

columns = [
    "age",           # Age in years
    "sex",           # 1 = male, 0 = female
    "cp",            # Chest pain type (0-3)
    "trestbps",      # Resting blood pressure (mm Hg)
    "chol",          # Serum cholesterol (mg/dl)
    "fbs",           # Fasting blood sugar > 120 mg/dl (1=true, 0=false)
    "restecg",       # Resting ECG results (0-2)
    "thalach",       # Max heart rate achieved
    "exang",         # Exercise induced angina (1=yes, 0=no)
    "oldpeak",       # ST depression induced by exercise
    "slope",         # Slope of peak exercise ST segment
    "ca",            # Number of major vessels (0-3)
    "thal",          # Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)
    "target"         # Diagnosis (0 = no disease, 1-4 = disease present)
]

df = pd.read_csv(save_path, header=None, names=columns)

# Binarize target: 0 = no disease, 1 = disease
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Save the clean-named version
df.to_csv("data/heart.csv", index=False)

print("\n📋 Dataset Preview:")
print(df.head())
print(f"\n📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"🎯 Target distribution:\n{df['target'].value_counts()}")
print("\n✅ Step 1 Complete! Dataset is ready in data/heart.csv")

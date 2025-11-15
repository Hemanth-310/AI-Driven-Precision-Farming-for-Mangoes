import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import os
# === Load Dataset ===
df = pd.read_csv(r"C:\Users\sabin\Downloads\archive (10)\Train_Data.csv")  # Replace with your actual file path


# === Features and Targets ===
features = ['Storage Time', 'DAFS', 'W', 'V', 'W/C', 'R', 'G', 'B']
targets = ['TA', 'TSS', 'TSS/TA']

X = df[features]
y = df[targets]

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Create output folder ===
os.makedirs("saved_models", exist_ok=True)

# === Train and Save CatBoost for Each Target ===
for target in targets:
    print(f"📦 Training CatBoost for: {target}")
    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_state=42
    )
    model.fit(X_train, y_train[target])

    # Save model
    model_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\ripeness\catboost_{target}.cbm"
    model.save_model(model_path)
    print(f"✅ Saved to: {model_path}\n")

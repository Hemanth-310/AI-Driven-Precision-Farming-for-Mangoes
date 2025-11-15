import pandas as pd
import shap
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt

# === Load Dataset ===
df = pd.read_csv(r"C:\Users\sabin\Downloads\archive (10)\Train_Data.csv")
features = ['Storage Time', 'DAFS', 'W', 'V', 'W/C', 'R', 'G', 'B']
X = df[features]

# === Load Saved Model (e.g., for TSS) ===
model = CatBoostRegressor()
model.load_model(r"D:\CS\AI\PROJECT\ML_agri project\new_models\ripeness\catboost_{target}.cbm")  # Path to saved model

# === Create SHAP Explainer ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# === Plot 1: Global Feature Importance (bar plot) ===
shap.summary_plot(shap_values, X, plot_type="bar")

# === Plot 2: SHAP Summary Plot (beeswarm) ===
shap.summary_plot(shap_values, X)

# === Optional: Plot 3: Single Sample Explanation ===
sample_idx = 5  # Pick a sample row to explain
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[sample_idx], X.iloc[sample_idx], matplotlib=True)

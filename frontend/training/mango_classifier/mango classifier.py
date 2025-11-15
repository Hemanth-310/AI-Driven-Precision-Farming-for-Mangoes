import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data setup
data = {
    'Target': ['TA', 'TA', 'TA', 'TSS', 'TSS', 'TSS', 'TSS/TA', 'TSS/TA', 'TSS/TA'],
    'Model': ['XGBoost', 'AdaBoost', 'CatBoost'] * 3,
    'MAE': [0.5831, 0.5439, 0.5026, 1.0366, 1.0724, 0.9972, 1.3634, 2.1058, 1.4507],
    'RMSE': [0.8611, 0.8010, 0.7694, 1.4430, 1.4558, 1.3568, 2.2572, 3.7557, 2.0753],
    'R2': [0.2710, 0.3693, 0.4181, 0.7151, 0.7101, 0.7482, 0.5412, -0.2701, 0.6122]
}

df = pd.DataFrame(data)

# Reshape for Seaborn
df_melted = df.melt(id_vars=["Target", "Model"],
                    value_vars=["MAE", "RMSE", "R2"],
                    var_name="Metric", value_name="Value")

# Use catplot to facet by Metric
g = sns.catplot(
    data=df_melted,
    kind="bar",
    x="Target",
    y="Value",
    hue="Model",
    col="Metric",
    palette="Set2",
    height=5,
    aspect=1
)

g.set_titles("{col_name}")
g.set_axis_labels("Target", "Score")
g._legend.set_title("Model")
plt.tight_layout()
plt.show()

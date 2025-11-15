from scipy.stats import f_oneway

# Validation accuracies
repvgg_accs = [0.991, 0.989, 0.991]
convnext_accs = [0.983, 0.986, 0.991]
tinyvit_accs = [0.892, 0.983, 0.983]

# Run ANOVA
f_stat, p_value = f_oneway(repvgg_accs, convnext_accs, tinyvit_accs)

print("\n📊 One-way ANOVA Results:")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("✅ Statistically significant difference between models (p < 0.05)")
else:
    print("❌ No significant difference between models (p ≥ 0.05)")


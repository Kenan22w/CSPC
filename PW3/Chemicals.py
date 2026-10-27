#PW3 - Chemicals and Cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("chemicals_cancer.csv")
print(df.shape)
print(df.head())

# 
print("\n--- NAIVE: correlation with malignancy (all 1000 patients) ---")
for chem in ["benzene", "cadmium"]:
    r, p = stats.pearsonr(df[chem], df["malignancy"])
    print(f"{chem}: r = {r:.3f}, p = {p:.1e}")


# Step A: do the chemicals correlate with pollution_index?
print("\n--- Full correlation matrix ---")
print(df.corr().round(3))

# Step B: restrict to patients with SIMILAR pollution, recompute
print("\n--- CONTROLLING for pollution (restrict to similar pollution) ---")
for lo, hi in [(40, 60), (45, 55)]:
    sub = df[(df["pollution_index"] > lo) & (df["pollution_index"] < hi)]
    print(f"pollution in ({lo}, {hi}), n = {len(sub)}")
    for chem in ["benzene", "cadmium"]:
        r, p = stats.pearsonr(sub[chem], sub["malignancy"])
        print(f"   {chem}: r = {r:.3f}, p = {p:.1e}")

# Step C: five bands, to show the pattern is consistent
df["band"] = pd.cut(df["pollution_index"], [0, 20, 40, 60, 80, 100])
bands = df.groupby("band", observed=True).apply(
    lambda s: pd.Series({"n": len(s),
                         "r_benzene": s["benzene"].corr(s["malignancy"]),
                         "r_cadmium": s["cadmium"].corr(s["malignancy"])}))
print(bands.round(3))

# Step D: partial correlation = remove pollution's linear effect from both
def residuals(y, x):
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)

print("\n--- Partial correlation (pollution removed) ---")
for chem in ["benzene", "cadmium"]:
    r = np.corrcoef(residuals(df[chem], df["pollution_index"]),
                    residuals(df["malignancy"], df["pollution_index"]))[0, 1]
    print(f"{chem}: partial r = {r:.3f}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].scatter(df["cadmium"], df["pollution_index"], s=8, alpha=0.5)
axes[0].set(xlabel="cadmium", ylabel="pollution_index",
            title="Cadmium just tracks pollution (r = 0.98)")

sub = df[(df["pollution_index"] > 45) & (df["pollution_index"] < 55)]
for ax, chem, col in [(axes[1], "benzene", "crimson"), (axes[2], "cadmium", "navy")]:
    ax.scatter(df[chem], df["malignancy"], s=6, alpha=0.2, color="grey", label="all")
    ax.scatter(sub[chem], sub["malignancy"], s=18, color=col, label="pollution 45-55")
    ax.set(xlabel=chem, ylabel="malignancy", title=f"{chem} vs malignancy")
    ax.legend()
fig.tight_layout()
fig.savefig("confounding.png", dpi=150)
# PW3 - Session 2:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("heart.csv")

disease = df.loc[df["target"] == 1, "thalach"]
healthy = df.loc[df["target"] == 0, "thalach"]
print(f"n disease = {len(disease)}, n healthy = {len(healthy)}")


print("Shapiro disease:", stats.shapiro(disease))
print("Shapiro healthy:", stats.shapiro(healthy))
print("Levene (equal variances?):", stats.levene(disease, healthy))


t, p = stats.ttest_ind(disease, healthy, equal_var=False)
print(f"Welch t-test: t = {t:.3f}, p = {p:.2e}")


u, p_u = stats.mannwhitneyu(disease, healthy, alternative="two-sided")
print(f"Mann-Whitney U: U = {u:.0f}, p = {p_u:.2e}")


pooled = np.sqrt((disease.var(ddof=1) + healthy.var(ddof=1)) / 2)
d = (disease.mean() - healthy.mean()) / pooled
print(f"Mean difference = {disease.mean() - healthy.mean():.2f} bpm, Cohen's d = {d:.2f}")


summary = {}
for name, g in [("disease", disease), ("healthy", healthy)]:
    mean = g.mean()
    sem = g.std(ddof=1) / np.sqrt(len(g))       # standard error of the mean
    ci = stats.t.ppf(0.975, len(g) - 1) * sem   # half-width of 95% CI
    summary[name] = (mean, sem, ci)
    print(f"{name}: mean = {mean:.2f}, SD = {g.std(ddof=1):.2f}, "
          f"SEM = {sem:.2f}, 95% CI = [{mean - ci:.2f}, {mean + ci:.2f}]")

fig, ax = plt.subplots(figsize=(5, 5))
labels = list(summary)
means = [summary[k][0] for k in labels]
cis = [summary[k][2] for k in labels]
ax.errorbar(labels, means, yerr=cis, fmt="o", capsize=8, markersize=9,
            color="navy", ecolor="crimson", elinewidth=2)
ax.set_ylabel("mean thalach (max heart rate, bpm)")
ax.set_title("Mean thalach with 95% confidence interval")
ax.set_xlim(-0.5, 1.5)
fig.tight_layout()
fig.savefig("thalach_means_ci.png", dpi=150)


r, p_r = stats.pearsonr(df["age"], df["thalach"])
rho, p_rho = stats.spearmanr(df["age"], df["thalach"])
print(f"Pearson r = {r:.3f} (p = {p_r:.1e}); Spearman rho = {rho:.3f} (p = {p_rho:.1e})")

slope, intercept, *_ = stats.linregress(df["age"], df["thalach"])
print(f"Fit: thalach = {slope:.2f} * age + {intercept:.1f}")


# Plot age vs thalach 
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df["age"], df["thalach"],     alpha=0.6, color="steelblue")
xs = np.array([df["age"].min(), df["age"].max()])
ax.plot(xs, slope * xs + intercept, color="red", label=f"fit, r = {r:.2f}")
ax.set_xlabel("age (years)")
ax.set_ylabel("thalach (bpm)")
ax.set_title("Age vs maximum heart rate")
ax.legend()
fig.tight_layout()
fig.savefig("age_vs_thalach.png", dpi=150)

# How uncertain are we about the proportions of disease and healthy 
for col in ["target", "sex"]:
    p = df[col].value_counts(normalize=True)
    entropy = -(p * np.log2(p)).sum()   # Shannon entropy in bits
    gini = 1 - (p ** 2).sum()           # Gini impurity
    max_h = np.log2(len(p))
    print(f"{col}: proportions = {p.round(3).to_dict()}, "
          f"entropy = {entropy:.3f} bits (max {max_h:.2f}), Gini = {gini:.3f}")
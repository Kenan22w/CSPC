# PW3 - Session 1:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1.Loading the data
df = pd.read_csv("heart.csv") # CSV to dataframe
print("Shape (rows, columns):", df.shape)
print(df.head())             #first 5 rows
print("Missing values:", df.isna().sum().sum()) # df.isna() is a boolean data frame

cont = df[["age", "chol", "trestbps", "thalach"]] # the 4 continuous columns
print(cont.describe().round(2)) # it shows count, mean, std, min, 25%, 50%, 75%, max for each variable

# 2. Shape of each variable (histograms)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), cont.columns):
    ax.hist(cont[col], bins=20, color="steelblue", edgecolor="white")
    ax.axvline(cont[col].mean(), color="red", ls="--", label="mean")
    ax.axvline(cont[col].median(), color="orange", ls="-", label="median")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.legend()
fig.suptitle("Distributions of the four continuous variables (n = %d)" % len(df)) # % is a placeholder for the number of rows in df
fig.tight_layout()
fig.savefig("distributions.png", dpi=150)
 
# 3. Normality of each variable
# (a) Graphical: Q-Q plot. Points on the red line  =>  approximately normal.
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), cont.columns):
    stats.probplot(cont[col], dist="norm", plot=ax)
    ax.set_title("Q-Q plot: " + col)
fig.tight_layout()
fig.savefig("qq_plots.png", dpi=150)
 
# (b) Numerical: skewness, excess kurtosis, and Shapiro-Wilk test.
# Shapiro-Wilk test (H0: data are normal; small p => reject normality)
rows = []
for col in cont.columns:
    x = cont[col]
    w, p = stats.shapiro(x)
    rows.append({"variable": col,
                 "skewness": stats.skew(x),
                 "excess_kurtosis": stats.kurtosis(x),
                 "shapiro_W": w,
                 "shapiro_p": p})
res = pd.DataFrame(rows).set_index("variable") 
print(res.to_string(float_format=lambda x: f"{x:.4g}"))
res.to_csv("normality_table.csv", float_format="%.4g")
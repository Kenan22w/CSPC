# CSPC - Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup

Create the environment for a given lab:

    conda env create -f PW<n>/Lab\ <X>/environment.yml

    conda activate cspc

---

## PW1 - Lab A: Reproducible Foundations

**What I built:**

- Created the CSPC repository and PW1/Lab A structure.
- Set up the Conda environment, Git repository, tests, and GitHub repository.

**Speed comparison (loop vs NumPy):**

- loop : 1.009213 s
- numpy : 0.000186 s
- speed-up: 5416.82x faster

**Tests:** all passing? yes

**Conclusion:**

- I created a reproducible Python environment and learned how to organize
  my coursework using Git and GitHub. I also learned how to write automated
  tests and verify that the decay simulation behaves correctly.

- The NumPy vectorised implementation was much faster than the pure-Python
  loop, showing the advantage of vectorisation for this type of simulation.





---

## PW1 - Lab B: Observed vs Analytical Decay

**What I built:**

- Read the observed decay data from `decay_observed.csv`.
- Compared the observed data with the analytical decay law
  `N(t) = N0 exp(-lambda t)` using `lambda = 0.3`.
- Created a 1x2 figure showing the observed data and analytical curve
  on shared axes.
- Created a Snakemake workflow to automate generation of `figure.png`.

**What the data showed:**

- The observed count decreases over time and follows the general
  exponential decay trend of the analytical model.
- The observed data has some fluctuations around the analytical curve.

**Conclusion:**

- The observed data generally follows the expected analytical decay
  behaviour, while showing fluctuations compared with the smooth
  analytical curve.
- Snakemake automates the workflow by checking the input and output
  files and running `plot.py` when the output needs to be generated or
  updated.



## PW2 — Lab A

### Motion from Tracking Data

The position data was differentiated twice using `numpy.gradient` to calculate velocity and acceleration.

The mean acceleration was approximately **-8.58 m/s²**, reasonably close to the theoretical free-fall acceleration of **-9.81 m/s²**.

The acceleration was much noisier than the position because differentiation amplifies measurement noise. Differentiating twice amplifies the noise further. The acceleration standard deviation was approximately **28.72 m/s²**.

The noisy acceleration was integrated back to velocity and position using `scipy.integrate.cumulative_trapezoid`. The maximum difference between the recovered and original position was approximately **0.78 m**, within about one metre.

This demonstrates that differentiation amplifies noise, while integration can suppress random noise.

### Bonus — 2D Trajectory

The x and y coordinates were differentiated separately to calculate the velocity components. The speed was calculated using `sqrt(vx**2 + vy**2)` and plotted against time. The 2D path was also plotted using the x and y coordinates.



## PW2 — Lab B: Optimization in Chemistry

### Optimization Methods

I compared Gradient Descent, Newton's Method, and the SLSQP
optimizer from SciPy using two functions.

For the first function, f(x) = (x-3)² + 1, all three methods found
the minimum at approximately x = 3.

For the second function, g(x) = x⁴ - 3x² + x + 5, the methods did
not all agree, and the results depended on the starting point.
From x0 = 0, Gradient Descent and SLSQP reached the global minimum
at about **x = -1.30**, but Newton's Method converged to
**x = 0.170**. The second derivative there was **-5.65**, so this
point is a maximum, not a minimum. Newton's Method only solves
g'(x) = 0, so it can stop at any stationary point. From x0 = 2,
Gradient Descent and Newton's Method found the local minimum at
**x = 1.131** (second derivative **+9.35**), while SLSQP still
reached x = -1.30. This showed that the starting point and the
method can both affect the result of an optimization.

### Kinetics

The first-order decay model

`C(t) = C0 * exp(-k*t)`

was fitted to the concentration data from `kinetics.csv`.

I used SLSQP to minimize the sum of squared differences between
the measured concentrations and the model predictions.

The fitted rate constant was approximately **0.262**, close to the
expected value of 0.25. The measured data and fitted curve were
plotted together in `kinetics.png` to compare the model with the
experimental results, and the curve passes through the data.

### Chemical Equilibrium

For the reaction

`H2 + I2 <-> 2HI`

with K = 15.6, I calculated the equilibrium composition using
Newton's Method and SLSQP.

Both methods gave an extent of reaction of approximately
**0.664 mol**, so they agree. The equilibrium amounts were
approximately **0.336 mol of H2**, **0.336 mol of I2**, and
**1.328 mol of HI**.

The graph (`equilibrium.png`) shows the amounts of the reactants
decreasing and the amount of HI increasing as the reaction proceeds
towards equilibrium, with the equilibrium marked by a vertical line.

### Bonus — Titration

I used `numpy.gradient` to calculate the slope of the pH curve
with respect to the volume of added base.

The equivalence point was estimated by finding the volume
corresponding to the maximum slope. The calculated value was
approximately **50 mL**.

The pH curve and its slope were plotted side by side in
`titration.png`, with a line at the equivalence point, to show
how the equivalence point can be identified numerically.



# PW3 — Data, Distributions, Testing, and Causation

## 1. Heart Disease Dataset

I used pandas, NumPy, Matplotlib and SciPy to analyse the Cleveland heart disease dataset. It contains 303 rows and 14 columns, with no missing values.

### Distribution of the Variables

I examined age, cholesterol (`chol`), resting blood pressure (`trestbps`) and maximum heart rate (`thalach`) using histograms, Q-Q plots and the Shapiro-Wilk normality test.

- **Age:** The distribution is fairly balanced, with a slight left skew.
- **Cholesterol:** The distribution is right-skewed, with some high values.
- **Resting blood pressure:** The distribution is moderately right-skewed.
- **Maximum heart rate:** The distribution has a slight left skew.

At the 0.05 significance level, the Shapiro-Wilk tests rejected strict normality for all four variables. The graphical checks are also useful: age and maximum heart rate look more balanced than cholesterol. I therefore considered both the plots and the statistical tests when choosing methods for Session 2.

## 2. Comparing Maximum Heart Rate

I compared maximum heart rate between patients with heart disease (`target = 1`) and healthy patients (`target = 0`).

- Disease group: mean = **139.26 bpm**
- Healthy group: mean = **158.38 bpm**
- Mean difference = **−19.12 bpm**

I used Welch's t-test to compare the means without assuming equal group variances. The result was statistically significant (\(t=-7.858,\ p<0.001\)). I also used the Mann-Whitney U test as a non-parametric comparison; it also showed a significant difference (\(p<0.001\)).

The 95% confidence intervals were approximately 135.47–143.05 bpm for the disease group and 155.42–161.34 bpm for the healthy group. The plot shows the same clear difference as the tests.

**Conclusion:** In this dataset, patients with heart disease had a lower average maximum heart rate than healthy patients. This is an association in the dataset, not proof that heart disease itself causes the difference.

## 3. Age and Maximum Heart Rate

I used a scatter plot, Pearson's correlation and Spearman's rank correlation to examine the relationship between age and maximum heart rate.

- Pearson correlation: \(r=-0.394\)
- Spearman correlation: \(\rho=-0.392\)

Both correlations were statistically significant (\(p<0.001\)) and negative.

**Conclusion:** Older patients tended to have lower maximum heart rates in this dataset. The similar Pearson and Spearman results support a negative relationship.

## 4. Bonus: Entropy and Gini Impurity

I calculated entropy and Gini impurity for the categorical columns `target` and `sex`, using the proportions of their categories.

- `target`: entropy = **0.995 bits**, Gini impurity = **0.497**
- `sex`: entropy = **0.905 bits**, Gini impurity = **0.435**

The `target` categories are more balanced than the `sex` categories in this dataset, so `target` has higher entropy and Gini impurity.

## 5. Chemical Exposure and Confounding

I analysed the synthetic chemical-exposure dataset to compare benzene and cadmium in relation to malignancy.

### Initial analysis

In the initial correlations, cadmium had a stronger positive correlation with malignancy (\(r\approx0.880\)) than benzene (\(r\approx0.385\)). Looking only at these results, cadmium would appear to be the stronger candidate.

However, cadmium was also very strongly correlated with the pollution index (\(r\approx0.982\)). This suggested that pollution could be a confounding variable.

### Accounting for pollution

I compared observations with similar pollution levels and calculated partial correlations while accounting for the pollution index.

- Benzene remained positively associated with malignancy. Its partial correlation was approximately **0.755**.
- Cadmium's partial correlation fell to approximately **0.013**, indicating almost no remaining linear association after accounting for pollution.

**Conclusion:** In this synthetic dataset, benzene is the chemical intended to drive malignancy, while cadmium appears strongly associated in the initial analysis mainly because it is related to pollution. This demonstrates why correlation alone is not enough to identify a cause. The conclusion is supported by the dataset's design and the confounding analysis; correlation analysis by itself does not prove causation in real-world data.

## 6. Files and How to Run

The `PW3` folder contains the Python scripts, input CSV files, plots and the normality results table.

Activate the `cspc` Conda environment, then open a terminal in the `PW3` folder and run:

```bash
python session1_heart.py
python session2_heart.py
python chemicals.py
```

The scripts use pandas, NumPy, Matplotlib and SciPy.


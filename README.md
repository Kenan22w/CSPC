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

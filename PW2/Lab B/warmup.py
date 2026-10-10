"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

# TODO 2A: minimise f three ways from x0=0 and print each result:
#   (1) gradient descent by hand (loop x = x - lr*df(x) until the step is tiny)
#   (2) scipy.optimize.newton(df, x0, fprime=d2f)
#   (3) scipy.optimize.minimize(f, x0, method="SLSQP")

# Gradient descent by hand:

x = 0.0
learning_rate = 0.01
step = 1.0
while abs(step) > 1e-8:
    step = learning_rate * df(x)
    x = x - step

print("Gradient Descent: ", x)

# Newton:

newton_result = newton(df, 0.0, fprime=d2f)

print("Newton's Method: ", newton_result)


# SLSQP: 
result = minimize(f, x0=[0.0], method="SLSQP")

print("SLSQP: ", result.x[0])


# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

# TODO 2B: run the same three methods on g, from x0=0 AND from x0=2.
#   For Newton (which solves dg(x)=0), also check the sign of d2g at the answer:
#   d2g > 0 means a minimum, d2g < 0 means a maximum.
#   In your README note: do the methods agree? did Newton find a minimum or
#   another stationary point? how did the starting point change the result?

for x0 in [0.0, 2.0]:
    # Gradient Descent
    x = x0
    step = 1.0
    while abs(step) > 1e-8:
        step = learning_rate * dg(x)
        x = x - step

    print("Gradient Descent from", x0, ":", x)

    # Newton's Method
    newton_result = newton(dg, x0, fprime=d2g)

    print("Newton's method from", x0, ":", newton_result)
    curv = d2g(newton_result)
    print("Second derivative:", curv, "->", "minimum" if curv > 0 else "maximum")

    # SLSQP
    result = minimize(g, x0=[x0], method="SLSQP")

    print("SLSQP from", x0, ":", result.x[0])

    print()
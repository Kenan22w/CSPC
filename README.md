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
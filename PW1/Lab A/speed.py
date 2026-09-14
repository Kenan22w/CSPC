import time 
from decay import simulate_loop, simulate

N0 = 200000
lam = 0.4
dt = 0.05
steps = 20
seed = 0


#pure pythom version

start = time.perf_counter()

simulate_loop(N0, lam, dt, steps, seed)

end = time.perf_counter()

py_time = end - start


#numpy version
start = time.perf_counter()

simulate(N0, lam, dt, steps, seed)

end = time.perf_counter()

numpy_time = end - start


speedup = py_time / numpy_time

print(f"Python loop: {py_time:.6f} s")
print(f"NumPy:       {numpy_time:.6f} s")
print(f"Speed-up:    {speedup:.2f}x")
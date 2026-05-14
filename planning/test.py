import numpy as np, time
from RaTLsnake import DPPlanner
from RaTLsnake.utils import contacts_plot

nc = 10
P = np.vstack((np.arange(0, nc), np.zeros(nc))).T
m = 5
na = 16
n_iter = 4
planner = DPPlanner(P, window_size=m, n_angles=na, refinement_steps=n_iter)

N = 10
avg = 0
for i in range(N):
    t0 = time.perf_counter()
    normals, pfcm = planner.solve()
    t1 = time.perf_counter()
    computation_time = t1 - t0
    avg += computation_time / N
    print(computation_time)
print(f"avg={avg}")
# print(f"Solved in {computation_time:.2f}s  (nc={nc}, m={m}, n_a={na}, n_iter={n_iter})")
# fcms = [planner.compute_metric(P[k : k + m], normals[k : k + m]) for k in range(nc - 4)]
# print(f"PFCM = {pfcm:.4f}, average FCM = {sum(fcms) / len(fcms):.4f}")

# sides = list("r" if n[1] > 0 else "l" for n in normals)

# contacts_plot(P, normals, interpolate=True, sides=sides, R=0.2, l=0.2)

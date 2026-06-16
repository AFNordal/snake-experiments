import numpy as np, time
from RaTLsnake import DPPlanner
from RaTLsnake.utils import contacts_plot, contacts_to_dict
from json import dump

nc = 10
P = np.vstack((np.arange(0, nc), np.zeros(nc))).T
m = 4
na = 16
n_iter = 5
planner = DPPlanner(P, window_size=m, n_angles=na, refinement_steps=n_iter)


t0 = time.perf_counter()
normals, pfcm = planner.solve()
t1 = time.perf_counter()
computation_time = t1 - t0
print(f"Solved in {computation_time:.2f}s  (nc={nc}, m={m}, n_a={na}, n_iter={n_iter})")
fcms = [planner.compute_metric(P[k : k + m], normals[k : k + m]) for k in range(nc - 4)]
print(f"PFCM = {pfcm:.4f}, average FCM = {sum(fcms) / len(fcms):.4f}")

sides = []
for i in range(nc):
    sides.append("r" if normals[i, 1] > 0 else "l")

path_dict = contacts_to_dict(P, normals)
with open("configs/path_description.json", "w") as file:
    dump(path_dict, file)
contacts_plot(P, normals, interpolate=True, sides=sides, R=0.2, l=0.2)


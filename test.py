import pathlib

import numpy as np, time
from path_planner import FCMPathPlanner, compute_fcm
from path_planner.fcm import compute_fcm_batch


def contacts_to_json(P, N, planner):
    """
    P: np.ndarray of shape (2, M)  - points as columns
    N: np.ndarray of shape (2, K)  - normals as columns

    Returns a dict ready to be serialized as JSON.
    """
    num_contacts = N.shape[1]
    contacts = []

    for i in range(num_contacts):
        point = P[i, :].tolist()
        normal = N[i, :].tolist()

        side = "r" if normal[1] > 0 else "l"

        contacts.append({"point": point, "normal": normal, "side": side})

    return {"contacts": contacts, "planner": planner}


# Warm JIT
# _ = compute_fcm_batch(np.random.randn(5,2), np.random.randn(10,5,2))
# print("compiled")

# np.random.seed(42)
# nc = 20
# angles = np.linspace(0, 2*np.pi, nc, endpoint=False) + np.random.randn(nc)*0.3
# P = np.column_stack([np.cos(angles)*2 + np.random.randn(nc)*0.1,
#                      np.sin(angles)*2 + np.random.randn(nc)*0.1])
# planner = FCMPathPlanner(P, m=4, n_a=18, n_iter=3)
P = np.array(
    [
        [
            -2,
            -1.5,
            -1,
            -0.5,
            0.0,
            0.5,
            0.75,
            1.0,
            1.5,
            1.75,
            2.0,
            2.5,
            2.75,
            3.0,
            3.5,
            3.75,
            4.0,
            4.5,
        ],
        [
            0,
            0.3,
            0.3,
            0,
            1.0,
            1.0,
            0.5,
            0.0,
            0.0,
            0.5,
            1.0,
            1.0,
            0.5,
            0.0,
            0.0,
            0.5,
            1.0,
            1.0,
        ],
    ]
).T
nc = P.shape[0]
N = np.array(
    [
        [np.sqrt(2.0), 0.0, 0.0, -np.sqrt(2.0), np.sqrt(2.0)],
        [-np.sqrt(2.0), 1.0, 1.0, -np.sqrt(2.0), -np.sqrt(2.0)],
    ]
)
m = 5
na = 24
planner = FCMPathPlanner(P, m=m, n_a=na, n_iter=3)
# planner = FCMPathPlanner(P, m=m, n_a=na, n_iter=3, initial_normals=N.T[:m])

t0 = time.perf_counter()
normals = planner.solve()
computation_time = time.perf_counter() - t0
print(f"Solved in {computation_time:.2f}s  (nc={nc}, m={m}, n_a={na}, n_iter=3)")
print(f"all unit: {np.allclose(np.linalg.norm(normals, axis=1), 1.0)}")
# print(compute_fcm(P[:m], normals[:m]))
fcms = [compute_fcm(P[k : k + m], normals[k : k + m]) for k in range(nc - 4)]
pfcm = min(fcms)
print(f"PFCM = {pfcm:.4f}  total={sum(fcms):.4f}  all>0: {all(f>0 for f in fcms)}")
root_dir = pathlib.Path(__file__).parent.resolve()
import json

path_dir = root_dir / "path_planner" / "paths"
planner_info = {
    "method": "DP with FCM1",
    "window_size": m,
    "num_directions": na,
    "PFCM1": pfcm,
    "computation_time": computation_time,
}
with open(path_dir / f"path_{m}_{na}.json", "w") as file:
    json.dump(contacts_to_json(P, normals, planner_info), file)

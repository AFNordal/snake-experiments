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

def contacts_plot(points: np.ndarray, normals: np.ndarray):
    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.use("WebAgg")

    assert points.shape == normals.shape
    P = points
    N = normals / np.linalg.norm(normals, axis=0)

    fig, ax = plt.subplots()
    arrs = []
    for i in range(P.shape[1]):
        arrs.append(
            ax.annotate(
                "",
                xytext=P[:, i],
                xy=P[:, i] + N[:, i] / 5,
                arrowprops=dict(arrowstyle="->", color="m"),
                annotation_clip=False,
            )
        )

    ax.scatter(P[0, :], P[1, :], color="b")

    ax.set_aspect("equal")
    ax.set_xlim((np.min(P[0,:])-1, np.max(P[0,:])+1))
    ax.set_ylim((np.min(P[1,:])-1, np.max(P[1,:])+1))
    plt.show()


# P = np.array(
#     [
#         [
#             -2,
#             -1.5,
#             -1,
#             -0.5,
#             0.0,
#             0.5,
#             0.75,
#             1.0,
#             1.5,
#             1.75,
#             2.0,
#             2.5,
#             2.75,
#             3.0,
#             3.5,
#             3.75,
#             4.0,
#             4.5,
#         ],
#         [
#             0,
#             0.3,
#             0.3,
#             0,
#             1.0,
#             1.0,
#             0.5,
#             0.0,
#             0.0,
#             0.5,
#             1.0,
#             1.0,
#             0.5,
#             0.0,
#             0.0,
#             0.5,
#             1.0,
#             1.0,
#         ],
#     ]
# ).T

nc = 100
P = np.vstack((np.arange(0, nc*0.5, 0.5), np.zeros(nc))).T
# N = np.array(
#     [
#         [np.sqrt(2.0), 0.0, 0.0, -np.sqrt(2.0), np.sqrt(2.0)],
#         [-np.sqrt(2.0), 1.0, 1.0, -np.sqrt(2.0), -np.sqrt(2.0)],
#     ]
# )
m = 5
na = 8
n_iter = 10
planner = FCMPathPlanner(P, m=m, n_a=na, n_iter=n_iter)

# planner = FCMPathPlanner(P, m=m, n_a=na, n_iter=3, initial_normals=N.T[:m])

t0 = time.perf_counter()
normals = planner.solve()
computation_time = time.perf_counter() - t0
print(f"Solved in {computation_time:.2f}s  (nc={nc}, m={m}, n_a={na}, n_iter={n_iter})")
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
compute_fcm(P[:m], normals[:m])
contacts_plot(P.T, normals.T)

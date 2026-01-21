import numpy as np
from scipy.optimize import linprog
import cvxpy as cp


def _get_intersections(P: np.ndarray, N: np.ndarray, tol: float = 1e-10):
    n = P.shape[1]

    # Lists to store results
    intersection_pts = []
    intersection_indices = []

    # Track which lines have been checked for parallelism
    checked = set()
    parallel_groups = []
    parallel_reps = []

    # Check all pairs of lines
    for i in range(n):
        for j in range(i + 1, n):
            p1, n1 = P[:, i], N[:, i]
            p2, n2 = P[:, j], N[:, j]

            if abs(np.dot(n1, n2)) > 1 - tol:
                # Lines are parallel
                if i not in checked:
                    # Start a new parallel group
                    group = [i, j]
                    parallel_groups.append(group)
                    parallel_reps.append(n1.copy())
                    checked.add(i)
                    checked.add(j)
                else:
                    # Add to existing group
                    for group in parallel_groups:
                        if i in group:
                            group.append(j)
                            checked.add(j)
                            break
            else:
                # Lines intersect - solve for intersection point
                A = np.column_stack([n1, -n2])
                b = p2 - p1
                t = np.linalg.solve(A, b)
                intersection = p1 + t[0] * n1

                intersection_pts.append(intersection)
                intersection_indices.append(set((i, j)))

    # Convert to ndarray
    if intersection_pts:
        intersections = np.column_stack(intersection_pts)
        indices = np.array(intersection_indices)
    else:
        intersections = np.empty((2, 0))
        indices = np.empty((0, 2), dtype=int)

    if parallel_reps:
        parallel_vectors = np.column_stack(parallel_reps)
    else:
        parallel_vectors = np.empty((2, 0))

    return intersections, indices, parallel_vectors, parallel_groups


def grasp_matrix(P, N):
    p0 = np.array([0, 0]) # Arbitrary?
    Gi_list = []
    nc = P.shape[1]
    for i in range(nc):
        theta = np.arctan2(N[1, i], N[0, i])
        Ri = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        r = P[:, i] - p0
        Pi = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [-r[1], r[0], 1],
            ]
        )
        Gi_list.append(Pi @ Ri)
    G_ = np.hstack(Gi_list)
    H = np.kron(np.eye(nc), np.array([1, 0, 0]))
    G = G_ @ H.T
    return G


def FC_G(P, N):
    G = grasp_matrix(P, N)
    m, n = G.shape

    # Objective doesn't matter (feasibility problem)
    c = np.zeros(n)

    # Equality constraints: Gy = 0 and sum(y) = 1
    A_eq = np.vstack([G, np.ones(n)])
    b_eq = np.zeros(m + 1)
    b_eq[-1] = 1

    # y >= 0
    bounds = [(0, None)] * n

    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    return res.success

def FCM2(P, N):
    G = grasp_matrix(P, N)
    m, n = G.shape

    y = cp.Variable(n)
    t = cp.Variable()

    constraints = [
        G @ y == 0,
        y >= t,
        cp.norm(y, 2) <= 1
    ]

    problem = cp.Problem(cp.Maximize(t), constraints)
    problem.solve()

    return t.value

def FCM1(points: np.ndarray, normals: np.ndarray):
    assert points.shape == normals.shape
    P = points
    N = normals / np.linalg.norm(normals, axis=0)
    n = P.shape[1]

    intersections, indices, parallel_dirs, parallel_groups = _get_intersections(P, N)

    min_PTI = np.inf
    for i in range(intersections.shape[1]):
        S_min = np.inf
        S_max = -np.inf
        for j in range(n):
            if j not in indices[i]:
                r = P[:, j] - intersections[:, i]
                dr = np.array((-r[1], r[0])) / np.linalg.norm(r)
                S_val = np.dot(dr, N[:, j])
                S_min = min(S_val, S_min)
                S_max = max(S_val, S_max)
        min_PTI = min(-S_min * S_max, min_PTI)

    for i in range(parallel_dirs.shape[1]):
        S_min = 0
        S_max = 0
        for j in range(n):
            if j not in parallel_groups[i]:
                r = parallel_dirs[:, i]
                dr = np.array((-r[1], r[0]))
                S_val = np.dot(dr, N[:, j])
                S_min = min(S_val, S_min)
                S_max = max(S_val, S_max)

        min_PTI = min(-S_min * S_max, min_PTI)
    return min_PTI


def FC_plot(points: np.ndarray, normals: np.ndarray):
    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.use("WebAgg")

    assert points.shape == normals.shape
    P = points
    N = normals / np.linalg.norm(normals, axis=0)

    color_counter = 0
    colors = (
        "red",
        "lime",
        "blue",
        "yellow",
        "fuchsia",
        "cyan",
        "orange",
        "springgreen",
        "deeppink",
        "gold",
    )
    ix, _, _, parallel_groups = _get_intersections(P, N)

    fig, ax = plt.subplots()
    for i in range(P.shape[1]):
        ax.axline(P[:, i], P[:, i] + N[:, i], color="k")
        ax.annotate(
            "",
            xytext=P[:, i],
            xy=P[:, i] + N[:, i],
            arrowprops=dict(arrowstyle="->", color="m"),
        )

    for i, group in enumerate(parallel_groups):
        for idx in group:
            ax.axline(
                P[:, idx],
                P[:, idx] + N[:, idx],
                color=colors[color_counter],
                linestyle="--",
            )
        color_counter += 1
    ax.scatter(P[0, :], P[1, :], color="b")
    ax.scatter(ix[0, :], ix[1, :], color="r")

    ax.set_aspect("equal")
    plt.show()


def FC_animation(points, normals, t0=0, t1=10, steps=500):
    from matplotlib import pyplot as plt
    from matplotlib.widgets import Slider
    import matplotlib

    matplotlib.use("WebAgg")

    T = np.linspace(t0, t1, steps)
    Ps, Ns, intersections = [], [], []
    FCMs = [[], []]
    for t in T:
        P = points(t)
        N = normals(t)
        N = N / np.linalg.norm(N, axis=0)
        ix, _, _, _ = _get_intersections(P, N)
        Ps.append(P)
        Ns.append(N)
        intersections.append(ix)
        FCMs[0].append(FCM1(P, N))
        FCMs[1].append(FCM2(P, N))

    contact_lines = []
    contact_arrows = []

    fig, (FCM_ax, diagram_ax, slider_ax) = plt.subplots(3, 1, height_ratios=(2, 8, 1))
    for i in range(P.shape[1]):
        contact_lines.append(diagram_ax.axline(Ps[0][:, i], Ps[0][:, i] + Ns[0][:, i], color="k"))
        contact_arrows.append(
            diagram_ax.annotate(
                "",
                xytext=Ps[0][:, i],
                xy=Ps[0][:, i] + Ns[0][:, i],
                arrowprops=dict(arrowstyle="->", color="m"),
            )
        )
    contact_points = diagram_ax.scatter(Ps[0][0, :], Ps[0][1, :], color="b")
    intersection_points = diagram_ax.scatter(
        intersections[0][0, :], intersections[0][1, :], color="r"
    )
    diagram_ax.set_aspect("equal")
    FCM_ax.plot(T, FCMs[0], label="FCM1")
    FCM_ax.plot(T, FCMs[1], label="FCM2")
    FCM_ax.legend(loc="right")
    FCM_timeline = FCM_ax.axvline(t0, color="r")

    t_slider = Slider(
        ax=slider_ax,
        label="t [s]",
        valmin=t0,
        valmax=t1,
        valinit=t0,
    )

    def update(t):
        idx = min(steps - 1, int((t - t0) / (t1 - t0) * steps))
        P = Ps[idx]
        N = Ns[idx]
        
        print(f"FC_G: {int(FC_G(P, N))}, FCM1: {FCM1(P, N):.4f}, FCM2: {FCM2(P, N):.4f}")
        intersection_points.set_offsets(intersections[idx].T)
        contact_points.set_offsets(P.T)
        FCM_timeline.set_xdata((t, t))
        for i in range(P.shape[1]):
            l = contact_lines[i]
            l.set_xy1(P[:, i])
            l.set_xy2(P[:, i] + N[:, i])
            a = contact_arrows[i]
            a.xy = P[:, i] + N[:, i]
            a.set_position(P[:, i])

    t_slider.on_changed(update)
    plt.show()


if __name__ == "__main__":

    def rocking_normals(frame: int):
        t = frame / 10.0
        x = 1.5 * np.sin(t)
        phi1 = -np.asin(2 / np.sqrt(8 + 4 * x + x**2))
        phi2 = -np.asin(2 / np.sqrt(8 - 4 * x + x**2))
        return np.array([[-np.sin(phi1), 0, 0, np.sin(phi2)], [-np.cos(phi1), 1, 1, -np.cos(phi2)]])

    def rocking_points(frame: int):
        return np.array([[-4, -1.5, 1.5, 4], [-2, 0, 0, -2 + frame * 0.01]])

    FC_animation(rocking_points, rocking_normals, t0=0, t1=40)

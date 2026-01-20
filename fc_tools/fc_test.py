import numpy as np


def get_intersections(P: np.ndarray, N: np.ndarray, tol: float = 1e-10):
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

                # Check if coincides with other intersection
                for k, other in enumerate(intersection_pts):
                    if np.linalg.norm(other - intersection) < tol:
                        intersection_indices[k].add(i)
                        intersection_indices[k].add(j)
                        break
                else:
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


def FCM(points: np.ndarray, normals: np.ndarray):
    assert points.shape == normals.shape
    P = points
    N = normals / np.linalg.norm(normals, axis=0)
    n = P.shape[1]

    intersections, indices, parallel_dirs, parallel_groups = get_intersections(P, N)
    # pole_twist_indices = []
    min_PTI = np.inf
    for i in range(intersections.shape[1]):
        S_min = 0
        S_max = 0
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
    ix, _, _, parallel_groups = get_intersections(P, N)

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


def FC_animation(points, normals):
    from matplotlib import pyplot as plt
    from matplotlib import animation as animation
    import matplotlib

    matplotlib.use("WebAgg")

    P = points(0)
    N = normals(0) / np.linalg.norm(normals(0), axis=0)

    ix, _, _, _ = get_intersections(P, N)

    contact_lines = []
    contact_arrows = []

    fig, (fcm_ax, ax) = plt.subplots(2, 1, height_ratios=(1, 4))
    for i in range(P.shape[1]):
        contact_lines.append(ax.axline(P[:, i], P[:, i] + N[:, i], color="k"))
        contact_arrows.append(
            ax.annotate(
                "",
                xytext=P[:, i],
                xy=P[:, i] + N[:, i],
                arrowprops=dict(arrowstyle="->", color="m"),
            )
        )
    contact_points = ax.scatter(P[0, :], P[1, :], color="b")
    intersection_points = ax.scatter(ix[0, :], ix[1, :], color="r")
    ax.set_aspect("equal")
    fcm_vals = [FCM(P, N)]
    fcm_times = [0]

    fcm_plot = fcm_ax.plot(fcm_times, fcm_vals)[0]
    fcm_ax.set_xlim(0, 314)
    fcm_ax.set_ylim(-0.2, 0.2)
    
    def update(frame):
        P = points(frame)
        N = normals(frame) / np.linalg.norm(normals(frame), axis=0)
        
        if len(fcm_times) > frame + 10:
            fcm_vals.clear()
            fcm_times.clear()
        fcm_vals.append(FCM(P, N))
        fcm_times.append(frame)
        fcm_plot.set_xdata(fcm_times)
        fcm_plot.set_ydata(fcm_vals)
        ix, _, _, _ = get_intersections(P, N)
        intersection_points.set_offsets(ix.T)
        for i in range(P.shape[1]):
            l = contact_lines[i]
            l.set_xy1(P[:, i])
            l.set_xy2(P[:, i] + N[:, i])
            a = contact_arrows[i]
            a.xy = P[:, i] + N[:, i]
            a.set_position(P[:, i])


    anim = animation.FuncAnimation(fig, update, 314, interval=100)
    anim.save(filename="./fcm1.mp4", writer="ffmpeg")
    plt.show()


if __name__ == "__main__":
    import pathlib
    import json

    def rocking_normals(frame: int):
        t = frame / 10.0
        x = 1.5 * np.sin(t)
        phi1 = -np.asin(2 / np.sqrt(8 + 4 * x + x**2))
        phi2 = -np.asin(2 / np.sqrt(8 - 4 * x + x**2))
        return np.array(
            [[-np.sin(phi1), 0, 0, np.sin(phi2)], [-np.cos(phi1), 1, 1, -np.cos(phi2)]]
        )

    def rocking_points(frame: int):
        return np.array([[-4, -1.5, 1.5, 4], [-2, 0, 0, -2]])

    root_dir = pathlib.Path(__file__).parent.resolve()
    # config_root = root_dir / "configs/"
    with open(root_dir / "path_description.json") as f:
        path_description = json.load(f)
    P = np.column_stack([c["point"] for c in path_description["contacts"]])
    N = np.column_stack([c["normal"] for c in path_description["contacts"]])
    # FCM(P, N)
    FC_animation(rocking_points, rocking_normals)
    # FC_plot(P, N)

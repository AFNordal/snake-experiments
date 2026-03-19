"""
Generalized contact Jacobian and wrench basis matrices.

Parameters
----------
n  : number of joint angles (phi0 … phi_{n-1}), giving n+1 link endpoints
     (base + n distal endpoints).  The link lengths are l0 … l_{n-1}.
nc : number of contacts.  Contact i attaches at a chosen link endpoint
     (specified by `contact_links[i]`, 0-based index into the n endpoints)
     with reach k_i along the outward direction and contact-frame angle phic_i.

Generalisation logic vs. the original MATLAB
---------------------------------------------
Original has n=3 (phi0, phi1, phi2), nc=4 contacts, with contacts 1 & 2 on
link 1 and contacts 3 & 4 on link 2/3.  Here every contact specifies which
link endpoint it sits on via `contact_links`.

Outputs
-------
qe  : generalised coordinate vector  [x, y, phi0…phi_{n-1}, k0…k_{nc-1},
                                       phic0…phic_{nc-1}]
X   : stacked constraint vector  (3*nc,)
J   : Jacobian  dX/dqe
N   : block-diagonal normal-vector matrix  (3*nc  x  nc)
T   : block-diagonal tangent-vector matrix (3*nc  x  nc)
G   : wrench basis matrix  (3  x  nc)
"""

import sympy as sp


def build_model(n: int, nc: int, contact_links: list[int], link_length: float):
    """
    Parameters
    ----------
    n            : number of link angles / links
    nc           : number of contacts
    contact_links: list of length nc; contact_links[i] is the 0-based index of
                   the link endpoint that contact i is attached to.
                   Endpoint index j means the tip of link j (0 = first link,
                   n-1 = last link).
    """

    assert len(contact_links) == nc, "contact_links must have length nc"
    assert all(0 <= j <= n - 1 for j in contact_links), (
        "contact_links entries must be in [0, n-1]"
    )

    # ------------------------------------------------------------------ #
    #  Symbolic variables
    # ------------------------------------------------------------------ #
    x, y = sp.symbols("x y", real=True)

    # Joint angles: phi_0 … phi_{n-1}
    phis = [sp.Symbol(f"φ{i}", real=True) for i in range(n)]

    # Link length
    l = link_length

    # Contact reaches and angles: k_0…k_{nc-1}, phic_0…phic_{nc-1}
    ks   = [sp.Symbol(f"k{i}",   real=True) for i in range(nc)]
    phics = [sp.Symbol(f"φ꜀_{i}", real=True) for i in range(nc)]


    # Generalised coordinate vector
    qe = sp.Matrix([x, y] + phis + ks + phics)

    # ------------------------------------------------------------------ #
    #  Cumulative angle sums:  theta_j = phi_0 + phi_1 + … + phi_j
    # ------------------------------------------------------------------ #
    theta = []
    for j in range(n):
        theta.append(sum(phis[: j + 1]))

    # ------------------------------------------------------------------ #
    #  Link-endpoint positions
    #  p[j] = (px, py) of the tip of link j (0-based)
    # ------------------------------------------------------------------ #
    px = {-1: x}   # base
    py = {-1: y}
    for j in range(n):
        px[j] = px[j - 1] + l * sp.cos(theta[j])
        py[j] = py[j - 1] + l * sp.sin(theta[j])

    # ------------------------------------------------------------------ #
    #  Contact constraints  c_i  (3-vector each)
    # ------------------------------------------------------------------ #
    #  Contact i sits at the tip of link contact_links[i], displaced by
    #  k_i in the direction theta[contact_links[i]], with angle constraint
    #  theta[contact_links[i]] = phic_i.
    # ------------------------------------------------------------------ #
    constraints = []
    for i in range(nc):
        j = contact_links[i]        # which link endpoint
        th = theta[j]               # cumulative angle up to that link

        cx = px[j - 1] + ks[i] * sp.cos(th)   # note: reach k replaces the
        cy = py[j - 1] + ks[i] * sp.sin(th)   # last link length for contact
        c_angle = th - phics[i]

        constraints.append(sp.Matrix([cx, cy, c_angle]))

    # Stack into a single column vector
    X = sp.Matrix([entry for c in constraints for entry in c])

    # ------------------------------------------------------------------ #
    #  Jacobian  J = dX / dqe
    # ------------------------------------------------------------------ #
    J = X.jacobian(qe)

    # ------------------------------------------------------------------ #
    #  Normal and tangent matrices  N, T  (block-diagonal)
    # ------------------------------------------------------------------ #
    zero = sp.Integer(0)

    def normal_vec(i):
        return sp.Matrix([-sp.sin(phics[i]), sp.cos(phics[i]), zero])

    def tangent_vec(i):
        return sp.Matrix([sp.cos(phics[i]), sp.sin(phics[i]), zero])

    # Build block-diagonal matrices
    N_mat = sp.zeros(3 * nc, nc)
    T_mat = sp.zeros(3 * nc, nc)
    for i in range(nc):
        n_vec = normal_vec(i)
        t_vec = tangent_vec(i)
        for row in range(3):
            N_mat[3 * i + row, i] = n_vec[row]
            T_mat[3 * i + row, i] = t_vec[row]

    # ------------------------------------------------------------------ #
    #  Wrench basis  G  (3 x nc)
    # ------------------------------------------------------------------ #
    G = sp.zeros(3, nc)
    for i in range(nc):
        cx  = X[3 * i]
        cy  = X[3 * i + 1]
        nix = N_mat[3 * i,     i]
        niy = N_mat[3 * i + 1, i]
        G[:, i] = sp.Matrix([nix, niy, cx * niy - cy * nix])

    N_func = sp.lambdify(qe, N_mat, "numpy")
    T_func = sp.lambdify(qe, T_mat, "numpy")
    J_func = sp.lambdify(qe, J, "numpy")
    G_func = sp.lambdify(qe, G, "numpy")

    return N_func, T_func, J_func, G_func
    # return dict(
    #     qe=qe,
    #     phis=phis,
    #     ks=ks,
    #     phics=phics,
    #     l=l,
    #     X=X,
    #     J=J,
    #     N=N_mat,
    #     T=T_mat,
    #     G=G,
    # )




# from SimSerpent.control.path.utils import contacts_to_obstacle_centers
# import json
# import pathlib

# # Load configs
# root_dir = pathlib.Path(__file__).parent.resolve()
# config_root = root_dir / "configs/"
# with open(config_root / "snake_description.json") as f:
#     snake_description = json.load(f)
# with open(config_root / "path_description.json") as f:
#     path_description = json.load(f)
# with open(config_root / "obstacles_description.json") as f:
#     obstacles_description = json.load(f)
# with open(config_root / "simulator_config.json") as f:
#     simulator_config = json.load(f)

# # Find obstacles from contacts
# obstacle_centers = contacts_to_obstacle_centers(
#     path_description["contacts"],
#     snake_description["link_radius_m"] + obstacles_description["default"]["radius"],
# )

# def rocking_references(dx: float):
#     xc1 = np.array(obstacle_centers[0])
#     xc2 = np.array(obstacle_centers[1])
#     xc3 = np.array(obstacle_centers[2])
#     xc4 = np.array(obstacle_centers[3])
#     l = snake_description["link_length_m"]

#     jc2 = np.array([-l/2 + dx, 0])
#     jc3 = np.array([l/2 + dx, 0])

#     phi1 = -np.arctan2(jc2[1] - xc1[1], jc2[0] - xc1[0])
#     phi2 =  np.arctan2(xc4[1] - jc3[1], xc4[0] - jc3[0])

#     phi0 = -phi1
#     jc1 = jc2 - np.array([l * np.cos(phi0), l * np.sin(phi0)])
#     x = jc1[0]
#     y = jc1[1]

#     k1 = np.linalg.norm(xc1 - jc1)
#     k2 = np.linalg.norm(xc2 - jc2)
#     k3 = np.linalg.norm(xc3 - jc2)
#     k4 = np.linalg.norm(xc4 - jc3)

#     phic1 = phi0 + np.pi
#     phic2 = 0.0
#     phic3 = 0.0
#     phic4 = phi2 + np.pi

#     return {
#         "x": x, "y": y,
#         "phi0": phi0, "phi1": phi1, "phi2": phi2,
#         "k1": k1, "k2": k2, "k3": k3, "k4": k4,
#         "phic1": phic1, "phic2": phic2, "phic3": phic3, "phic4": phic4
#     }


# if __name__ == "__main__":
#     from matplotlib import pyplot as plt
#     import matplotlib
#     matplotlib.use("WebAgg")
#     from matplotlib.widgets import Slider

#     t = []
#     dx = []
#     x = []
#     y = []
#     phi0, phi1, phi2 = [], [], []
#     k1, k2, k3, k4 = [], [], [], []
#     phic1, phic2, phic3, phic4 = [], [], [], []

#     for i in range(1000):
#         t.append(i / 100.)
#         dx.append(np.sin(t[-1]))

#         refs = rocking_references(dx[-1])
#         x.append(refs["x"])
#         y.append(refs["y"])
#         phi0.append(refs["phi0"])
#         phi1.append(refs["phi1"])
#         phi2.append(refs["phi2"])
#         k1.append(refs["k1"])
#         k2.append(refs["k2"])
#         k3.append(refs["k3"])
#         k4.append(refs["k4"])
#         phic1.append(refs["phic1"])
#         phic2.append(refs["phic2"])
#         phic3.append(refs["phic3"])
#         phic4.append(refs["phic4"])

    
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.25)

#     plt.plot(t, dx)
#     plt.show()

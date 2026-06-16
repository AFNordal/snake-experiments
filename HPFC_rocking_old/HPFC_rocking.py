from SimSerpent.control.controllers import PIDControllerArray
from SimSerpent.simulation import Simulator
from SimSerpent.control.path.utils import contacts_to_obstacle_centers
import json
import pathlib
import numpy as np
from symmath import build_model
import osqp
import scipy.sparse as sp
from tqdm import trange

class ForceQPSolver:
    def __init__(self):
        self._prob = None
        self._prev_shape = None
        self._prev_x = None
        self._prev_y = None

    def solve(self, G: np.ndarray, f_min: float, discontinuous: bool = False) -> np.ndarray:
        if G.shape != self._prev_shape or discontinuous:
            # Re-setup when dimensions change (rare, accepts cold start)
            self._setup(G, f_min)
        else:
            # Cheap in-place value update, avoids re-factorization
            n = G.shape[1]
            A = sp.vstack([sp.csc_matrix(G), sp.eye(n)], format="csc")
            l = np.concatenate([np.zeros(G.shape[0]), f_min * np.ones(n)])
            self._prob.update(Ax=A.data, l=l)
            self._prob.warm_start(x=self._prev_x, y=self._prev_y)

        res = self._prob.solve()
        if res.info.status != "solved":
            raise RuntimeError(f"OSQP failed: {res.info.status}")

        self._prev_x, self._prev_y = res.x, res.y
        self._prev_shape = G.shape
        return res.x

    def _setup(self, G: np.ndarray, f_min: float):
        n = G.shape[1]
        # A stacks equality (Gf=0) and inequality (f >= f_min) constraints
        A = sp.vstack([sp.csc_matrix(G), sp.eye(n)], format="csc")
        l = np.concatenate([np.zeros(G.shape[0]), f_min * np.ones(n)])
        u = np.concatenate([np.zeros(G.shape[0]), np.inf * np.ones(n)])
        self._prob = osqp.OSQP()
        self._prob.setup(
            sp.eye(n, format="csc"),
            np.zeros(n),
            A,
            l,
            u,
            verbose=False,
            warm_starting=True,
            eps_abs=1e-8,
            eps_rel=1e-8,
        )

# Load configs
root_dir = pathlib.Path(__file__).parent.resolve()
config_root = root_dir / "configs/"
with open(config_root / "snake_description.json") as f:
    snake_description = json.load(f)
with open(config_root / "path_description.json") as f:
    path_description = json.load(f)
with open(config_root / "obstacles_description.json") as f:
    obstacles_description = json.load(f)
with open(config_root / "simulator_config.json") as f:
    simulator_config = json.load(f)

# Find obstacles from contacts
obstacle_centers = contacts_to_obstacle_centers(
    path_description["contacts"],
    snake_description["link_radius_m"] + obstacles_description["default"]["radius"],
)
for c in obstacle_centers:
    obstacles_description["obstacles"].append({"position": c})
contact_points = list([c["point"] for c in path_description["contacts"]])


def rocking_references(dx: float):
    xc1 = np.array(contact_points[0])
    xc2 = np.array(contact_points[1])
    xc3 = np.array(contact_points[2])
    xc4 = np.array(contact_points[3])
    l = snake_description["link_length_m"]

    jc2 = np.array([-l / 2 + dx, 0])
    jc3 = np.array([l / 2 + dx, 0])

    phi1 = -np.arctan2(jc2[1] - xc1[1], jc2[0] - xc1[0])
    phi2 = np.arctan2(xc4[1] - jc3[1], xc4[0] - jc3[0])

    phi0 = -phi1
    jc1 = jc2 - np.array([l * np.cos(phi0), l * np.sin(phi0)])
    x = jc1[0]
    y = jc1[1]

    k1 = np.linalg.norm(xc1 - jc1)
    k2 = np.linalg.norm(xc2 - jc2)
    k3 = np.linalg.norm(xc3 - jc2)
    k4 = np.linalg.norm(xc4 - jc3)

    phic1 = phi0 + np.pi
    phic2 = 0.0
    phic3 = 0.0
    phic4 = phi2 + np.pi

    return np.array([x, y, phi0, phi1, phi2, k1, k2, k3, k4, phic1, phic2, phic3, phic4])


# Initialize snake on "path"
r0 = rocking_references(0)
pose = r0[0:5]
snake_description["initial_pose"] = pose
N_func, T_func, J_func, G_func = build_model(3, 4, [0, 1, 1, 2], snake_description["link_length_m"])

simulator = Simulator(
    simulator_config,
    snake_description,
    obstacles_description,
    video_output_path=root_dir / "out.mp4",
)
sim_duration = 50.0  # seconds
dt = simulator.dt
prev_u_f = None
shape_controller = PIDControllerArray(1000, 0, 5, n=snake_description["n_links"] - 1, derivative_filter_tc=0.001)
force_controllers = PIDControllerArray(2, 3, 0.0001, n=4, input_filter_tc=dt*50, derivative_filter_tc=0)

max_effort = 0.5
min_effort = 0.

plot_t = []
plot_f = []
plot_fd = []
plot_phi = []
plot_phid = []
plot_tau = []
plot_nc = []

sat_hi = None
sat_lo = None

qpsolver = ForceQPSolver()

for step in trange(int(sim_duration / dt)):

    # qe = rocking_references(np.sin(step * dt / 10) * 0.3)
    qe = rocking_references(0.2*np.sin(max(0, (step*dt-0.5) / 10)))
    # print(0.3*np.sin(max(0, (step*dt-0.5) / 10)))
    # print(qe[3], qe[4], simulator.get_joint_angles())
    N = N_func(*qe)
    J = J_func(*qe)
    T = T_func(*qe)
    G = G_func(*qe)

    G_null = np.eye(4) - np.linalg.pinv(G) @ G
    # f_d = -G_null @ np.ones(4) * 0.1
    f_d = -qpsolver.solve(G, 0.1)
    # print(f_d)
    force_controllers.set_reference(f_d)
    f = -np.array(
        [np.linalg.norm(simulator.get_obstacle_contact_force(f"obstacle_{i}")) for i in range(4)]
    )
    force_efforts = force_controllers.tick(f, dt, sat_lo, sat_hi)
    sat_lo = force_efforts < -max_effort
    sat_hi = force_efforts > -min_effort
    force_efforts = np.maximum(np.minimum(-min_effort, force_efforts), -max_effort)

    tau_fb = (J.T @ N @ G_null @ force_efforts)[3:5]
    # print(tau_ctrl[3:5], f_d-f)

    f_Fd = N @ (f_d)# + G_null @ force_efforts)

    tau_ff = (J.T @ f_Fd)[3:5]

    phi_d = qe[3:5]
    shape_controller.set_reference(phi_d)
    phi = simulator.get_joint_angles()
    shape_torques = shape_controller.tick(phi, dt)

    motion_projector = (np.linalg.pinv(J) @ J)[3:5, 3:5]
    tau_s = motion_projector @ shape_torques

    tau = tau_fb + tau_ff + tau_s
    simulator.step(tau)

    plot_t.append(step*dt)
    plot_f.append(f)
    plot_fd.append(f_d)
    plot_phi.append(np.array(phi))
    plot_phid.append(phi_d)
    plot_tau.append(tau)
    plot_nc.append(simulator.get_n_contacts())

    if simulator.should_close():
        break

simulator.close_window()

import matplotlib
matplotlib.use("WebAgg")
from matplotlib import pyplot as plt

plot_t = np.array(plot_t)
plot_f = np.array(plot_f)
plot_fd = np.array(plot_fd)
plot_phi = np.array(plot_phi)
plot_phid = np.array(plot_phid)
plot_tau = np.array(plot_tau)
plot_nc = np.array(plot_nc)

fig, axes = plt.subplots(4, 2)

for i in range(4):
    axes[i, 0].plot(plot_t, -plot_f[:, i])
    axes[i, 0].plot(plot_t, -plot_fd[:, i], ":")

    axes[i, 0].set_ylim((-0.1, 2))
    axes[i, 0].set_title(f"Contact {i} force")
    
for i in range(2):
    axes[i, 1].plot(plot_t, -plot_phi[:, i])
    axes[i, 1].plot(plot_t, -plot_phid[:, i], ":")
    axes[i, 1].set_title(f"Joint {i} angle")

# for i in range(2):
#     axes[6+i].plot(plot_t, plot_tau[:, i])
#     axes[6+i].set_title(f"Joint {i} torque")

axes[-1, 1].plot(plot_t, plot_nc)
axes[-1, 1].set_title("Number of contacts")

fig.tight_layout()

plt.show()
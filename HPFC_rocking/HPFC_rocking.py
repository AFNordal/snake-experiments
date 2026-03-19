from SimSerpent.control.controllers import PIDControllerArray
from SimSerpent.simulation import Simulator
from SimSerpent.control.path.utils import contacts_to_obstacle_centers
import json
import pathlib
import numpy as np
from symmath import build_model

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
pose = r0[0:4]
snake_description["initial_pose"] = pose
N_func, T_func, J_func, G_func = build_model(3, 4, [0, 1, 1, 2], snake_description["link_length_m"])

controller = PIDControllerArray(40000, 0, 10, snake_description["n_links"] - 1, 0.001)
force_controllers = PIDControllerArray(10, 0, 0.000000, n=4)
force_controllers.set_reference(np.ones(4) * 0.1)
simulator = Simulator(
    simulator_config,
    snake_description,
    obstacles_description,
    video_output_path=root_dir / "out.mp4",
)
sim_duration = 50.0  # seconds
dt = simulator.dt
prev_u_f = None
for step in range(int(sim_duration / dt)):

    # qe = rocking_references(np.sin(step * dt / 10) * 0.3)
    qe = rocking_references(0)
    # print(qe[3], qe[4], simulator.get_joint_angles())
    N = N_func(*qe)
    J = J_func(*qe)
    T = T_func(*qe)
    G = G_func(*qe)

    G_null = np.eye(4) - np.linalg.pinv(G) @ G
    wPef = N @ G_null @ np.linalg.pinv(N)
    col_norms = np.linalg.norm(wPef, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1
    wPef = wPef / col_norms
    J_pinv = np.linalg.pinv(J)
    jPef = J.T @ wPef @ J_pinv.T
    print(jPef>0.01)
    jPaf = (J_pinv @ J).T
    jFf = jPaf @ np.linalg.pinv(jPef @ jPaf) @ jPef
    fd = np.ones(4) * 0.1
    f = np.array([np.linalg.norm(simulator.get_obstacle_contact_force(f"obstacle_{i}")) for i in range(4)])
    tau_f = jFf @ J.T @ N @ (fd - f)
    print(tau_f)
    controller.set_reference([qe[3], qe[4]])
    torques = controller.tick(simulator.get_joint_angles(), dt)

    simulator.step(torques)

    if simulator.should_close():
        break

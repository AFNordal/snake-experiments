from SimSerpent.control.controllers import HPFCController
from SimSerpent.simulation import Simulator
from SimSerpent.control.path import SnakePath
from SimSerpent.control.path.utils import s_ref_sin
import json
import pathlib
from tqdm import tqdm, trange  # Loading bar
from time import time_ns

import numpy as np

# Load configs
root_dir = pathlib.Path(__file__).parent.resolve()
config_root = root_dir / "configs"
with open(config_root / "snake_description.json") as f:
    snake_description = json.load(f)
with open(config_root / "path_description.json") as f:
    path_description = json.load(f)
with open(config_root / "obstacles_description.json") as f:
    obstacles_description = json.load(f)
with open(config_root / "simulator_config.json") as f:
    simulator_config = json.load(f)

path = SnakePath(path_description, snake_description, dir="backward")

# Find obstacles from contacts
obstacle_centers = path.calculate_obstacle_centers(
    snake_description["link_radius_m"] + obstacles_description["default"]["radius"],
)
for c in obstacle_centers:
    obstacles_description["obstacles"].append({"position": c})

# Initialize snake on path
s_ddot = 0.5
t_settle = 0.2


sim_duration = 10  # seconds
dt = simulator_config["timestep"]
N = int(sim_duration / dt)


deltas = s_ref_sin(dt=dt, N=N, t_settle=t_settle, amplitude=0.05, acceleration=s_ddot)
S = []
for d in tqdm(deltas):
    S.append(path.delta_to_s(d))

print(path.curve.length())

pose = path.to_snake_pose(S[0])
print(pose)
# dist = 0.0
# pose[7] += dist
# pose[8] -= dist
snake_description["initial_pose"] = pose
simulator = Simulator(
    simulator_config,
    snake_description,
    obstacles_description,
    # video_output_path=root_dir / "out.mp4",
)
simulator.set_display_curve(path.curve, z=0)
hpfc_controller = HPFCController(path, f_min=0.1)


ncons = []
times = []


steps_per_step = 2
for step in range(N // steps_per_step):
    # Find reference pose
    path_param = S[step*steps_per_step]
    phi = simulator.get_joint_angles()

    joint_centers = simulator.get_joint_center_coords()
    s, _ = path.curve.closest_point(joint_centers[-1])
    contact_obstacles, contact_links = path.planned_contacts(s)
    f = np.array(
        [simulator.get_obstacle_contact_force(f"obstacle_{i}") for i in contact_obstacles]
    )
    err = hpfc_controller._inner_propulsion_controller.prev_error
    if err is None:
        err = 0
    torques = hpfc_controller.tick(dt * steps_per_step, path_param, s, phi, f)

    ncons.append(simulator.get_n_contacts())
    times.append(step*steps_per_step*dt)
    
    simulator.step(torques, iterations=steps_per_step)
    if simulator.should_close():
        break
    
simulator.close_window()

from matplotlib import pyplot as plt

plt.plot(times, ncons)
plt.show()
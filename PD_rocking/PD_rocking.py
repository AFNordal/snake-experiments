from SimSerpent.control.controllers import PIDController
from SimSerpent.simulation import Simulator

# from SimSerpent.control.path import SnakePath
from SimSerpent.control.path.utils import contacts_to_obstacle_centers
import json
import pathlib
import numpy as np

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
    print(c)

# Initialize snake on path
pose = [-2 - 2 * np.sqrt(2), -2 * np.sqrt(2), np.pi / 4, -np.pi / 4, -np.pi / 4]
snake_description["initial_pose"] = pose


def rocking_references(t: float):
    x = 0.2*np.sin(t / 10)
    phi1 = -np.asin(2 / np.sqrt(8 + 4*x + x**2))
    phi2 = -np.asin(2 / np.sqrt(8 - 4*x + x**2))
    return np.array([phi1, phi2])


controller = PIDController(2000, 0, 10, snake_description["n_links"] - 1, 0)
simulator = Simulator(simulator_config, snake_description, obstacles_description)
# simulator.set_display_curve(path.curve, z=0.1)
sim_duration = 50.0  # seconds
dt = simulator.dt
for step in range(int(sim_duration / dt)):
    # pose = path.to_snake_pose(max(0, step * dt / 15 - 0.1), unit="rad")
    phi_r = rocking_references(step*dt)
    controller.set_reference(phi_r)
    torques = controller.tick(simulator.get_joint_angles(), dt)
    print(controller.prev_error, torques)
    simulator.step(torques)
    if simulator.should_close():
        break

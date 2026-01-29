from SimSerpent.control.controllers import PIDController
from SimSerpent.simulation import Simulator
from SimSerpent.control.path import SnakePath
from SimSerpent.control.path.utils import contacts_to_obstacle_centers
import json
import pathlib
from tqdm import trange  # Loading bar

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

# Find obstacles from contacts
obstacle_centers = contacts_to_obstacle_centers(
    path_description["contacts"],
    snake_description["link_radius_m"] + obstacles_description["default"]["radius"],
)
for c in obstacle_centers:
    obstacles_description["obstacles"].append({"position": c})

# Initialize snake on path
path = SnakePath(path_description, snake_description)
path_param = 0
pose = path.to_snake_pose(path_param, unit="rad")
snake_description["initial_pose"] = pose

controller = PIDController(6000, 0, 25, snake_description["n_links"] - 1, 0)
simulator = Simulator(simulator_config, snake_description, obstacles_description, video_output_path="")
simulator.set_display_curve(path.curve, z=0)
sim_duration = 30.0  # seconds
dt = simulator.dt

errors = []
tau_max = []

for step in trange(int(sim_duration / dt)):
    if (step % 200) == 0:
        rms_error = path.rms_tracking_error(simulator.get_joint_center_coords())
        errors.append(rms_error)
    # Find reference pose
    path_param = min(max(0, (step * dt - 0.2) / 4), path.curve.length() - 4.5)
    pose = path.to_snake_pose(path_param, unit="rad")
    # Compute and apply control
    controller.set_reference(pose[3:])
    torques = controller.tick(simulator.get_joint_angles(), dt)
    tau_max.append(max(torques))
    simulator.step(torques)
    if simulator.should_close():
        break
simulator.close_window()

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("WebAgg")
# plt.plot(errors, label="rms_to_closest")
plt.plot(tau_max, label="tau_max")
plt.legend()
plt.show()
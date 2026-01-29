from SimSerpent.control.controllers import PIDController
from SimSerpent.simulation import Simulator
from SimSerpent.control.path import SnakePath
from SimSerpent.control.path.utils import contacts_to_obstacle_centers
import json
import pathlib

# Load configs
root_dir = pathlib.Path(__file__).parent.resolve()
config_root = root_dir / "configs/example_experiment/"
with open(config_root / "snake_description.json") as f:
    snake_description = json.load(f)
with open(config_root / "out_path2.json") as f:
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
pose = path.to_snake_pose(0, unit="rad")
snake_description["initial_pose"] = pose

controller = PIDController(400000, 0, 1500, snake_description["n_links"] - 1, 0)
simulator = Simulator(simulator_config, snake_description, obstacles_description, video_output_path="")#root_dir/"out.mp4")
# simulator.set_display_curve(path.curve, z=0.1)
sim_duration = 40.0  # seconds
dt = simulator.dt
for step in range(int(sim_duration / dt)):
    pose = path.to_snake_pose(min(max(0, step * dt/4 - 0.1), path.curve.length()-4.5), unit="rad")
    controller.set_reference(pose[3:])
    torques = controller.tick(simulator.get_joint_angles(), dt)
    simulator.step(torques)
    if simulator.should_close():
        break
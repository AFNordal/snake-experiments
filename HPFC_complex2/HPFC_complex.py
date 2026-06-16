from SimSerpent.control.controllers import HPFCController
from SimSerpent.simulation import Simulator
from SimSerpent.control.path import SnakePath
from SimSerpent.control.path.utils import s_ref_linear
import json
import pathlib
from tqdm import tqdm, trange  # Loading bar
import numpy as np
import traceback

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


sim_duration = 400.0  # seconds
dt = simulator_config["timestep"]
N = int(sim_duration / dt)
path = SnakePath(path_description, snake_description, dir="backward")
s0 = path.opposite_end_s(0.05, dir="forward")
# s0 = 7.1
S = s_ref_linear(dt=dt, N=N, s0=s0, t_settle=0.2, s_ddot=0.1, s_dot=0.1)


# Initialize snake on path
pose = path.to_snake_pose(S[0])
snake_description["initial_pose"] = pose

# Find obstacles from contacts
obstacle_centers = path.calculate_obstacle_centers(
    snake_description["link_radius_m"] + obstacles_description["default"]["radius"]
)
for c in obstacle_centers:
    obstacles_description["obstacles"].append({"position": c})

simulator = Simulator(
    simulator_config,
    snake_description,
    obstacles_description,
    video_output_path=root_dir / "out.mp4",
)
simulator.set_display_curve(path.curve, z=2 * snake_description["link_radius_m"])
hpfc_controller = HPFCController(path, f_min=0.1)
forces = []
times = []
ncons = []
sms = []
steps_per_step = 2
try:
    for step in range(N // steps_per_step):
        # while 1:
        #     simulator.step()
        # Find reference pose
        path_param = S[step]
        phi = simulator.get_joint_angles()
        # torques = pos_controllers.tick(phi, dt)
        actual_s = 0
        joint_centers = simulator.get_joint_center_coords()
        for i_ in range(3):
            i = int((i_ + 1) * (path.n_links + 1) / 5)
            jc = joint_centers[i]
            s_proj, _ = path.curve.closest_point(jc)
            _, params = path.curve.equally_spaced_points(
                path.link_length, path.n_links + 1 - i, s_proj, dir="forward"
            )
            actual_s += params[-1]
        actual_s /= 3
        # s, _ = path.curve.closest_point(joint_centers[-1])
        contact_obstacles, contact_links = path.planned_contacts(actual_s)
        # t0 = time_ns()
        f = np.array(
            [simulator.get_obstacle_contact_force(f"obstacle_{i}") for i in contact_obstacles]
        )

        torques = hpfc_controller.tick(steps_per_step * dt, path_param, actual_s, phi, f)
        forces.append(np.min(f))
        times.append(step * steps_per_step * dt)
        ncons.append(simulator.get_n_contacts())
        sms.append(actual_s)

        simulator.step(torques, iterations=steps_per_step)

        if simulator.should_close():
            break
except Exception as e:
    traceback.print_exc()

simulator.close_window()
if step * dt > 1:
    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.use("WebAgg")
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(times, ncons)
    axes[1].plot(times, forces)
    axes[2].plot(times, S[: len(times)], "r:")
    axes[2].plot(times, sms)
    plt.show()
    

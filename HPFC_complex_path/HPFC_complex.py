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

S = list(0.6 + s for s in s_ref_linear(dt=dt, N=N, s0=6.6, t_settle=0.2, s_ddot=0.05, s_dot = 0.01))
# S = []
# for d in tqdm(deltas):
#     S.append(path.delta_to_s(d))

# import numpy as np
# import matplotlib

# matplotlib.use("webagg")
# from matplotlib import pyplot as plt

# t = np.linspace(0, sim_duration, N)
# plt.plot(t, deltas)
# plt.show()


# Initialize snake on path
pose = path.to_snake_pose(S[0])
dist = 0.0
pose[8] += dist
pose[9] -= 2 * dist
pose[10] += dist
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
# pos_controllers = PIDControllerArray(6000, 0, 20, n=snake_description["n_links"] - 1, derivative_filter_tc=2*dt)
hpfc_controller = HPFCController(path, f_min=0.1)
forces = []
times = []
ncons = []
sms = []

try:
    for step in range(N):
        # while 1:
        #     simulator.step()
        # Find reference pose
        path_param = S[step]
        phi = simulator.get_joint_angles()
        # torques = pos_controllers.tick(phi, dt)

        joint_centers = simulator.get_joint_center_coords()
        s, _ = path.curve.closest_point(joint_centers[-1])
        contact_obstacles, contact_links = path.planned_contacts(s)
        # t0 = time_ns()
        f = np.array([simulator.get_obstacle_contact_force(f"obstacle_{i}") for i in contact_obstacles])
            
        torques = hpfc_controller.tick(dt, path_param, s, phi, f)
        forces.append(np.min(f))
        times.append(step * dt)
        ncons.append(simulator.get_n_contacts())
        sms.append(s)

        simulator.step(torques)
        
        if simulator.should_close():
            break
except Exception as e:
    traceback.print_exc()

    # if (step % 20) == 0:
    #     planned_contacts.append(len(path.planned_contacts(path_param)[0]))
    #     real_contacts.append(simulator.get_n_contacts())
    # j = simulator.get_contact_jacobian([0,0], 3)
    # for r in j:
    #     print(" ".join(list(f"{e:2.1f}" for e in r)))
    # print()
hpfc_controller.print_profile()
simulator.close_window()

from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("WebAgg")
fig, axes = plt.subplots(3, 1)
axes[0].plot(times, ncons)
axes[1].plot(times, forces)
axes[2].plot(times, S[: len(times)], "r:")
axes[2].plot(times, sms)
plt.show()
# # plt.plot(errors, label="rms_to_closest")
# # plt.plot(tau_max, label="tau_max")
# plt.plot(list(range(len(planned_contacts))), planned_contacts)
# plt.plot(list(range(len(real_contacts))), real_contacts)
# plt.show()

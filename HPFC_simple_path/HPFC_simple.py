from SimSerpent.control.controllers import HPFCController
from SimSerpent.simulation import Simulator
from SimSerpent.control.path import SnakePath
from SimSerpent.control.path.utils import s_ref_sin
from SimSerpent.control.path.interpolation.rocking import abgk_to_contacts
import json
import pathlib
from tqdm import tqdm, trange  # Loading bar
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


sim_duration = 60.0  # seconds
dt = simulator_config["timestep"]
N = int(sim_duration / dt)
contacts = abgk_to_contacts(
    path_description["alpha"],
    path_description["beta"],
    path_description["gamma"],
    1.0,
)
_contacts = abgk_to_contacts(
    path_description["alpha"],
    path_description["beta"],
    path_description["gamma"]*2,
    1.0,
)
contacts = [_contacts[0]] + contacts + [_contacts[-1]]
path_description["contacts"] = contacts
path = SnakePath(path_description, snake_description, dir="forward")

S = list(0.6 + s for s in s_ref_sin(dt=dt, N=N, t_settle=0.5, amplitude=0.05, acceleration=0.05))
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
forces = list([] for i in range(4))
times = []
ncons = []
sms = []


for step in range(N):
    # while 1:
    #     simulator.step()
    # Find reference pose
    path_param = S[step]
    # print(path_param, path.s_to_delta(path_param))
    # pose = path.to_snake_pose(path_param)
    # Compute and apply control
    # pos_controllers.set_reference(pose[3:])
    phi = simulator.get_joint_angles()
    # torques = pos_controllers.tick(phi, dt)

    joint_centers = simulator.get_joint_center_coords()
    s, _ = path.curve.closest_point(joint_centers[0])
    # print(s, path.s_to_delta(s))
    # print(f"{s:.5f}\t{path_param:.5f}")
    # print(f"{path.s_to_delta(s):.5f}\t{deltas[step]:.5f}")
    contact_obstacles, contact_links = path.planned_contacts(s)
    # t0 = time_ns()
    f = np.array([simulator.get_obstacle_contact_force(f"obstacle_{i}") for i in contact_obstacles])
    # print(", \t".join(f"{i:.2f}" for i in f))
    err = hpfc_controller._inner_controller.prev_error
    if err is None:
        err = 0
    # print(f"{simulator.get_n_contacts()}/{len(contact_obstacles)}\t{np.linalg.norm(err):.3f}\t{path_param:.3f}\t{s:.3f}")
    torques = hpfc_controller.tick(dt, path_param, s, phi, f)
    for i in range(4):
        forces[i].append(f[i])
    times.append(step * dt)
    ncons.append(simulator.get_n_contacts())
    sms.append(s)

    # t1 = time_ns()
    # print(f"{(t1-t0)*1e-6:.2f} ms")

    # contact_set = set(contact_indices)
    # for new in contact_set - prev_contacts:
    #     force_controllers.push_controller(new_reference=1)
    #     if prev_u_f is not None:
    #         prev_u_f = np.append(prev_u_f, 0)
    # for gone in prev_contacts - contact_set:
    #     force_controllers.pop_controller()
    #     if prev_u_f is not None:
    #         prev_u_f = prev_u_f[1:]
    # prev_contacts = contact_set

    # contact_points = list(path.contact_points[i] for i in contact_indices)
    # contact_normals_T = block_diag(*list(path.contact_normals[i] for i in contact_indices))
    # contact_jacobians = np.vstack(
    #     [simulator.get_contact_jacobian(p, l) for p, l in zip(contact_points, contact_links)]
    # )
    # J_N = contact_normals_T @ contact_jacobians
    # motion_projector = (
    #     np.eye(J_N.shape[1]) - J_N.T @ np.linalg.inv(J_N @ J_N.T + np.eye(J_N.shape[0])) @ J_N
    # )
    # # print(J_N.shape)
    # contact_forces = np.array(
    #     [
    #         np.linalg.norm(simulator.get_obstacle_contact_force(f"obstacle_{i}"))
    #         for i in contact_indices
    #     ]
    # )
    # sat_hi = np.zeros(len(contact_set), dtype=bool)
    # sat_lo = contact_forces <= 0
    # if prev_u_f is not None:
    #     sat_lo = np.logical_or(sat_lo, prev_u_f < -10)
    #     sat_hi = np.logical_or(sat_hi, prev_u_f > 10)
    # u_f = force_controllers.tick(contact_forces, dt, sat_lo, sat_hi)
    # # u_f = force_controllers.tick(contact_forces, dt)#, sat_lo, sat_hi)
    # prev_u_f = u_f
    # print(" ".join(f"{f:.2f}" for f in contact_forces))
    # print(" ".join(f"{f:.2f}" for f in u_f))

    # force_torques = - J_N.T @ np.maximum(0, u_f)

    simulator.step(torques)
    # simulator.step()
    # while 1:
    #     pass
    if simulator.should_close():
        break

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
for i in range(4):
    axes[1].plot(times, forces[i])
axes[2].plot(times, S[: len(times)], "r:")
axes[2].plot(times, sms)
plt.show()
# # plt.plot(errors, label="rms_to_closest")
# # plt.plot(tau_max, label="tau_max")
# plt.plot(list(range(len(planned_contacts))), planned_contacts)
# plt.plot(list(range(len(real_contacts))), real_contacts)
# plt.show()

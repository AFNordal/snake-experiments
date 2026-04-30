from SimSerpent.control.path import SnakePath
import json
import pathlib
import numpy as np
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

sim_duration = 5.0  # seconds
dt = 0.01
N = int(sim_duration / dt)

# Initialize snake on path
path = SnakePath(path_description, snake_description, dir="backward")
pose = path.to_snake_pose(path.delta_to_s(-0.0))

vals = []
ders = []
ss = []
for step in trange(int(sim_duration / dt)):
    # Find reference pose
    try:
        path_param = path.delta_to_s(0. + step*dt*0.1 / sim_duration)
        pose = path.to_snake_pose(path_param)
        _der = path.reference_derivative(path_param)
        vals.append(pose[-1])
        ders.append(_der[-1])
        ss.append(path_param)
    except:
        break

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
matplotlib.use("WebAgg")
fig, (s_ax, p_ax, m_ax) = plt.subplots(3, 1, height_ratios=(1,10,20))
S = Slider(
    ax = s_ax,
    label="s",
    valmin=min(ss),
    valmax=max(ss),
    valinit=min(ss)
)
num_ders = list((vals[i]-vals[i-1])/(ss[i]-ss[i-1]) for i in range(1, len(vals)))
p_ax.plot(ss, vals)
p_ax.plot(ss, ders)
p_ax.plot(ss[1:], num_ders)

disp_points = list(path.curve.point_at(s) for s in np.arange(0, path.curve.length(), 0.01))
m_ax.plot(list(i[0] for i in disp_points), list(i[1] for i in disp_points))
centers, dists = path.curve.equally_spaced_points(snake_description["link_length_m"], snake_description["n_links"]+1, min(ss), path.dir)
circles = []
for c in centers:
    c_ = plt.Circle(c, snake_description["link_length_m"], fill=False)
    m_ax.add_patch(c_)
    circles.append(c_)
snake, = m_ax.plot(list(c[0] for c in centers), list(c[1] for c in centers), "o:r")
t = path.curve.tangent_at(dists[-1])
vec, = m_ax.plot([centers[-1][0], centers[-1][0]+t[0]], [centers[-1][1], centers[-1][1]+t[1]], color="g")
m_ax.set_aspect("equal")
P = path.contact_points
m_ax.scatter(list(i[0] for i in P), list(i[1] for i in P))


scan = p_ax.axvline(x=min(ss), color="red")
def update(val):
    scan.set_xdata([val, val])
    centers, dists = path.curve.equally_spaced_points(snake_description["link_length_m"], snake_description["n_links"]+1, val, path.dir)
    snake.set_xdata(list(c[0] for c in centers))
    snake.set_ydata(list(c[1] for c in centers))
    for crc, c in zip(circles, centers):
        crc.set(center=c)
    t = path.curve.tangent_at(dists[-1])
    vec.set_xdata([centers[-1][0], centers[-1][0]+t[0]])
    vec.set_ydata([centers[-1][1], centers[-1][1]+t[1]])
S.on_changed(update)
plt.show()
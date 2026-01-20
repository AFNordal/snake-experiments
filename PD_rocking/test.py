import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("WebAgg")


def rocking_references(t: float):
    x = 0.5*np.sin(t / 10)
    phi1 = -np.asin(2 / np.sqrt(8 + 4*x + x**2))
    phi2 = -np.asin(2 / np.sqrt(8 - 4*x + x**2))
    return [phi1, phi2]

t = np.arange(0, 100, 0.005)
angles = [rocking_references(i) for i in t]
plt.plot(t, list(i[0] for i in angles), "r")
plt.plot(t, list(i[1] for i in angles), "b")
plt.show()
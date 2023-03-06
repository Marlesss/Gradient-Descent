from typing import Callable
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from vector import Vector

GRID_SIZE = 10


def wrap_func(func: Callable[[Vector], float]) -> Callable[[[float]], float]:
    def wrapped_func(*args) -> float:
        x = Vector(*args)
        return func(x)

    return wrapped_func


def unzip_dots(dots: [Vector]) -> [[float]]:
    if len(dots) == 0:
        return None
    n = len(dots[0])
    ans = [[] for _ in range(n)]
    for dot in dots:
        for i in range(n):
            ans[i].append(dot[i])
    return ans


def show_2d_plot(func: Callable[[Vector], float], dots: [Vector]):
    subplot = plt.subplot()
    min_x = min(map(lambda dot: dot[0], dots))
    max_x = max(map(lambda dot: dot[0], dots))
    t = np.linspace(min_x, max_x, GRID_SIZE)
    s = np.vectorize(wrap_func(func))(t)
    subplot.plot(t, s)
    subplot.plot(*unzip_dots(dots))
    subplot.grid()
    plt.show()


def func(x: Vector) -> float:
    assert len(x) == 1
    return x[0] ** 2


# dots = [Vector(-10, 1), Vector(-2, 2), Vector(10, 100)]
#
# show_2d_plot(func, dots)

dots = [[20, 20], [5.318943338572119, 2.6002291420114005], [2.3215306705179453, 4.285782032031778],
        [2.0592856709478977, 3.974998164989072], [2.00574344289354, 4.005104460614513],
        [2.001059008973999, 3.9995534327991282], [2.0001025940580357, 4.000091172695428],
        [2.0000189168814164, 3.9999920236948685], [2.0000018319421162, 4.00000162885025],
        [2.0000003378495244, 3.999999857576605]]


def test_1_function(x: Vector) -> float:
    assert len(x) == 2
    return 2 * ((x[0] - 2) ** 2) + 4 * ((x[1] - 4) ** 2)


fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
# zline = np.linspace(0, 15, len(dots))
# xline, yline = unzip_dots(dots)
# ax.plot3D(xline, yline, zline, 'gray')

# # Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, cmap='Greens')

# x_dots, y_dots = unzip_dots(dots)
# xline = np.linspace(min(x_dots), max(x_dots), GRID_SIZE)
# yline = np.linspace(min(y_dots), max(y_dots), GRID_SIZE)
# zline = np.vectorize(test_1_function)(dots)
# zline = np.vectorize(wrap_func(test_1_function))(*dots)
# print(zline)

# plt.show()

gridd = np.mgrid[0:10:0.5, 0:10:0.5].reshape(2, -1).T
zline = np.array(list(map(test_1_function, gridd)))
for i in range(len(zline)):
    print(gridd.T[0][i], gridd.T[1][i], zline[i])
ax.scatter3D(gridd.T[0], gridd.T[1], zline)
plt.show()

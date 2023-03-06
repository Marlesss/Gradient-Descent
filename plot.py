from typing import Callable
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from vector import Vector

GRID_SIZE = 100


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


def show_1arg_func(func: Callable[[Vector], float], dots: [Vector]):
    subplot = plt.subplot()
    min_x = min(map(lambda dot: dot[0], dots))
    max_x = max(map(lambda dot: dot[0], dots))
    t = np.linspace(min_x, max_x, GRID_SIZE)
    s = np.vectorize(wrap_func(func))(t)
    subplot.plot(t, s)
    subplot.plot(*unzip_dots(dots))
    subplot.grid()
    plt.show()


def show_2arg_func(func: Callable[[Vector, Vector], float], dots: [Vector]):
    x_line, y_line = unzip_dots(dots)
    k = 2
    grid = np.mgrid[min(x_line) / k:max(x_line) * k:complex(0, GRID_SIZE),
           min(y_line) / k:max(y_line) * k:complex(0, GRID_SIZE)].reshape(2, -1).T
    print(grid)
    func_values = np.array(list(map(func, grid)))
    minz, maxz = min(func_values), max(func_values)
    medz = (minz + maxz) / 5

    subplot = plt.subplot()
    for i in range(GRID_SIZE ** 2):
        z = func_values[i]
        if z < medz:
            c = (z - minz) / (medz - minz)
            subplot.plot([grid.T[0][i]], [grid.T[1][i]], "ro", color=(0, 1 - c, c))
        else:
            c = (z - medz) / (maxz - medz)
            subplot.plot([grid.T[0][i]], [grid.T[1][i]], "ro", color=(c, 0, 1 - c))
    subplot.plot(x_line, y_line, "ro", color=(1, 1, 1))
    subplot.plot(x_line, y_line, linewidth=1, color=(1, 1, 1))
    plt.show()

# def func(x: Vector) -> float:
#     assert len(x) == 1
#     return x[0] ** 2


# dots = [Vector(-10, 1), Vector(-2, 2), Vector(10, 100)]


# dots = [[20, 20], [5.318943338572119, 2.6002291420114005], [2.3215306705179453, 4.285782032031778],
#         [2.0592856709478977, 3.974998164989072], [2.00574344289354, 4.005104460614513],
#         [2.001059008973999, 3.9995534327991282], [2.0001025940580357, 4.000091172695428],
#         [2.0000189168814164, 3.9999920236948685], [2.0000018319421162, 4.00000162885025],
#         [2.0000003378495244, 3.999999857576605]]
#
#
# def test_1_function(x: Vector) -> float:
#     assert len(x) == 2
#     return 2 * ((x[0] - 2) ** 2) + 4 * ((x[1] - 4) ** 2)
#
#
# show_2arg_func(test_1_function, dots)

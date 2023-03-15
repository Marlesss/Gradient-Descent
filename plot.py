from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from vector import Vector

GRID_SIZE = 1000


def show_2arg_func(f: Callable[[np.ndarray], float], dots: np.ndarray, dots_show: bool = True, levels: bool = False,
                   contour: bool = False, show=True, label: str = None, clabel: bool = False, color: tuple = (1, 0, 0)):
    x_min, x_max = min(dots[:, 0]), max(dots[:, 0])
    y_min, y_max = min(dots[:, 1]), max(dots[:, 1])
    x_space = np.linspace(x_min - (x_max - x_min) / 10 - 100, x_max + (x_max - x_min) / 10 + 100, GRID_SIZE)
    y_space = np.linspace(y_min - (y_max - y_min) / 10 - 100, y_max + (y_max - y_min) / 10 + 100, GRID_SIZE)
    if dots_show:
        if label:
            plt.plot(dots[:, 0], dots[:, 1], 'o-')
        else:
            x, y = dots[:, 0], dots[:, 1]
            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1,
                       color=color)
            # plt.plot(dots[:, 0], dots[:, 1], 'o-')
    if levels:
        contour_set = plt.contour(x_space, y_space, [[f(np.array([x, y])) for x in x_space] for y in y_space],
                                  levels=sorted(list(set([f(dot) for dot in dots]))))
        if clabel:
            plt.clabel(contour_set)
    if contour:
        contour_set = plt.contour(x_space, y_space, [[f(np.array([x, y])) for x in x_space] for y in y_space])
        if clabel:
            plt.clabel(contour_set)
    if show:
        plt.show()


def show_2arg_func_slice(func: Callable[[np.ndarray], float], x_min=-100, x_max=100, y_min=-100, y_max=100, **kwargs):
    grid = np.mgrid[x_min:x_max:complex(0, GRID_SIZE),
           y_min:y_max:complex(0, GRID_SIZE)].reshape(2, -1).T
    show_2arg_func(func, grid, **kwargs)

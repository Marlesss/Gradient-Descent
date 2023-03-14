from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from vector import Vector

GRID_SIZE = 100



def show_2arg_func(f: Callable[[np.ndarray], float], dots: np.ndarray, dots_show: bool = True, levels: bool = False):
    x_min, x_max = min(dots[:, 0]), max(dots[:, 0])
    y_min, y_max = min(dots[:, 1]), max(dots[:, 1])
    x_space = np.linspace(x_min - (x_max - x_min) / 10, x_max + (x_max - x_min) / 10, GRID_SIZE)
    y_space = np.linspace(y_min - (y_max - y_min) / 10, y_max + (y_max - y_min) / 10, GRID_SIZE)
    if dots_show:
        plt.plot(dots[:, 0], dots[:, 1], 'o-')
    if levels:
        plt.contour(x_space, y_space, [[f(np.array([x, y])) for x in x_space] for y in y_space],
                    levels=sorted([f(dot) for dot in dots]))
    else:
        plt.contour(x_space, y_space, [[f(np.array([x, y])) for x in x_space] for y in y_space])


def show_2arg_func_contour(func: Callable[[np.ndarray], float], x_min=-100, x_max=100, y_min=-100, y_max=100):
    dots = np.mgrid[x_min:x_max:complex(0, GRID_SIZE),
           y_min:y_max:complex(0, GRID_SIZE)].reshape(2, -1).T
    show_2arg_func(func, dots, dots_show=False, levels=True)

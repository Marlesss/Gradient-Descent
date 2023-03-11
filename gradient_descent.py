from typing import Callable
from vector import Vector
import numpy as np
from math import sqrt

ITER_LIMIT = 50
EPS = 10 ** -3
PHI = (1 + 5 ** 0.5) / 2


def golden_section_method(x: np.ndarray, grad: Callable[[np.ndarray], np.ndarray],
                          f: Callable[[np.ndarray], float]) -> float:
    left = 0
    right = 1
    med_left = left + (right - left) / (PHI + 1)
    med_right = right - (right - left) / (PHI + 1)
    f_left = f(x - med_left * grad(x))
    f_right = f(x - med_right * grad(x))

    while abs(left - right) > EPS:
        if f_left < f_right:
            right = med_right
            med_right = med_left
            f_right = f_left
            med_left = left + (right - left) / (PHI + 1)
            f_left = f(x - med_left * grad(x))
        else:
            left = med_left
            med_left = med_right
            f_left = f_right
            med_right = right - (right - left) / (PHI + 1)
            f_right = f(x - med_right * grad(x))

    return left


def gradient_descent_linear(x0: np.ndarray, grad: Callable[[np.ndarray], np.ndarray],
                            f: Callable[[np.ndarray], float]) -> (bool, np.ndarray):
    dots = np.array([x0])
    prev_x = x0
    for _ in range(ITER_LIMIT):
        grad_value = grad(prev_x)
        if sqrt((grad_value ** 2).sum()) < EPS:
            return True, dots
        new_x = prev_x - golden_section_method(prev_x, grad, f) * grad_value
        dots = np.append(dots, [new_x], 0)
        prev_x = new_x
    return False, dots


def gradient_descent_constant(x0: np.ndarray, alph: float,
                              grad: Callable[[np.ndarray], np.ndarray]) -> (bool, [np.ndarray]):
    dots = np.array([x0])
    prev_x = x0
    for _ in range(ITER_LIMIT):
        grad_value = grad(prev_x)
        if sqrt((grad_value ** 2).sum()) < EPS:
            return True, dots
        new_x = prev_x - alph * grad_value
        dots = np.append(dots, [new_x], 0)
        prev_x = new_x
    return False, dots

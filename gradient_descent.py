from typing import Callable
from vector import Vector
import numpy as np
from math import sqrt

ITER_LIMIT = 50
EPS = 10 ** -3
PHI = (1 + 5 ** 0.5) / 2


def first_wolfe_condition(x: np.ndarray, antigradient: np.ndarray, alpha: float,
                          c1: float, grad: Callable[[np.ndarray], np.ndarray],
                          f: Callable[[np.ndarray], float]) -> bool:
    return f(x + alpha * antigradient) <= f(x) + c1 * alpha * np.dot(grad(x), antigradient)


def second_wolfe_condition(x: np.ndarray, antigradient: np.ndarray, alpha: float,
                           c2: float, grad: Callable[[np.ndarray], np.ndarray]) -> bool:
    return np.dot(grad(x + alpha * antigradient), antigradient) >= c2 * np.dot(grad(x), antigradient)


def golden_section_method_with_wolfe_conditions(x: np.ndarray, grad: Callable[[np.ndarray], np.ndarray],
                                                f: Callable[[np.ndarray], float],
                                                c1: float, c2: float) -> float:
    left = 0
    right = 1
    gradient = grad(x)
    med_left = left + (right - left) / (PHI + 1)
    med_right = right - (right - left) / (PHI + 1)
    f_left = f(x - med_left * gradient)
    f_right = f(x - med_right * gradient)

    while abs(left - right) > EPS:
        if f_left < f_right:
            right = med_right
            med_right = med_left
            f_right = f_left
            med_left = left + (right - left) / (PHI + 1)
            f_left = f(x - med_left * gradient)
        else:
            left = med_left
            med_left = med_right
            f_left = f_right
            med_right = right - (right - left) / (PHI + 1)
            f_right = f(x - med_right * gradient)
        checkpoint = (left + right) / 2
        if (first_wolfe_condition(x, -gradient, checkpoint, c1, grad, f)
                and second_wolfe_condition(x, -gradient, checkpoint, c2, grad)):
            return checkpoint

    return left


def golden_section_method(x: np.ndarray, grad: Callable[[np.ndarray], np.ndarray],
                          f: Callable[[np.ndarray], float]) -> float:
    left = 0
    right = 1
    gradient = grad(x)
    med_left = left + (right - left) / (PHI + 1)
    med_right = right - (right - left) / (PHI + 1)
    f_left = f(x - med_left * gradient)
    f_right = f(x - med_right * gradient)

    while abs(left - right) > EPS:
        if f_left < f_right:
            right = med_right
            med_right = med_left
            f_right = f_left
            med_left = left + (right - left) / (PHI + 1)
            f_left = f(x - med_left * gradient)
        else:
            left = med_left
            med_left = med_right
            f_left = f_right
            med_right = right - (right - left) / (PHI + 1)
            f_right = f(x - med_right * gradient)
    return left


def gradient_descent_linear_with_wolfe_condition(x0: np.ndarray, grad: Callable[[np.ndarray], np.ndarray],
                                                 f: Callable[[np.ndarray], float], c1: float, c2: float) -> (
bool, np.ndarray):
    dots = np.array([x0])
    prev_x = x0
    for _ in range(ITER_LIMIT):
        grad_value = grad(prev_x)
        if sqrt((grad_value ** 2).sum()) < EPS:
            return True, dots
        new_x = prev_x - golden_section_method_with_wolfe_conditions(prev_x, grad, f, c1, c2) * grad_value
        dots = np.append(dots, [new_x], 0)
        prev_x = new_x
    return False, dots


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

from typing import Callable
from vector import Vector

ITER_LIMIT = 50
EPS = 10 ** (-5)
PHI = (1 + 5 ** 0.5) / 2


def golden_section_method(x: Vector, grad: Callable[[Vector], Vector], f: Callable[[Vector], float]) -> [Vector]:
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

    return x - left * grad(x)


def gradient_descent_linear(x0: Vector, grad: Callable[[Vector], Vector],
                            f: Callable[[Vector], float]) -> (bool, [Vector]):
    dots = [x0]
    prev_x = x0
    for _ in range(ITER_LIMIT):
        new_x = golden_section_method(prev_x, grad, f)
        dots.append(new_x)

        if (new_x - prev_x).vector_length() < EPS:
            return True, dots
        prev_x = new_x
    return False, dots


def gradient_descent_constant(x0: Vector, alph: float, grad: Callable[[Vector], Vector]) -> (bool, [Vector]):
    dots = [x0]
    prev_x = x0
    for _ in range(ITER_LIMIT):
        new_x = prev_x - alph * grad(prev_x)
        dots.append(new_x)

        if (new_x - prev_x).vector_length() < EPS:
            return True, dots
        prev_x = new_x
    return False, dots

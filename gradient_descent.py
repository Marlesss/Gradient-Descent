from typing import Callable
from vector import Vector

ITER_LIMIT = 50
EPS = 10 ** (-5)


def gradient_descent(x0: Vector, alph: float, grad: Callable[[Vector], Vector]) -> (bool, [Vector]):
    eps_dot = Vector([EPS for _ in range(len(x0))])
    dots = [x0]
    prev_x = x0
    for _ in range(ITER_LIMIT):
        new_x = prev_x - alph * grad(prev_x)
        dots.append(new_x)

        # изменить на сравнение длины вектора
        if abs(new_x - prev_x) < eps_dot:
            return True, dots
        prev_x = new_x
    return False, dots

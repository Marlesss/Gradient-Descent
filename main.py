from gradient_descent import *
from vector import Vector


def test_1_function(x: Vector) -> float:
    assert len(x) == 2
    return 2 * ((x[0] - 2) ** 2) + 4 * ((x[1] - 4) ** 2)


def test_1_grad(x: Vector) -> Vector:
    assert len(x) == 2
    return Vector((x[0] - 2) * 6, (x[1] - 4) * 8)


def main():
    # ans, dots = gradient_descent_linear(Vector(20, 20), test_1_grad, test_1_function)
    ans, dots = gradient_descent_constant(Vector(20, 20), 0.1, test_1_grad)
    if ans:
        print(f"Сошлось за {len(dots)} шагов")
        print(dots)
    else:
        print("Не сошлось")
        print(dots)


if __name__ == '__main__':
    main()

from gradient_descent import gradient_descent
from vector import Vector


def test_1(x: Vector) -> Vector:
    assert len(x) == 2
    return Vector([(x[0] - 2) * 6, (x[1] - 4) * 8])


def main():
    ans, dots = gradient_descent(Vector([0, 0]), 0.1, test_1)
    if ans:
        print(f"Сошлось за {len(dots)} шагов")
        print(dots)
    else:
        print("Не сошлось")
        print(dots)


if __name__ == '__main__':
    main()

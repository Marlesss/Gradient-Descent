from gradient_descent import *
from square_function import generate_function
from plot import *


def main():
    func, grad = generate_function(2, 2)
    start_dot = np.array([20, 20])
    # ans, dots = gradient_descent_linear(start_dot, grad, func)
    ans, dots = gradient_descent_constant(start_dot, 0.03, grad)
    print(f"{'Сошлось' if ans else 'Не сошлось'} за {len(dots)} шагов")
    print(dots)
    # show_2arg_func(func, dots, levels=False)
    show_2arg_func(func, dots, levels=True)
    # show_2arg_func_contour(func)


if __name__ == '__main__':
    main()

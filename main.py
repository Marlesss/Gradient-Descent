from matplotlib.patches import Patch

from gradient_descent import *
from square_function import generate_function
from plot import *
from pprint import pprint
import math


def get_short_func():
    def ploho(x: np.ndarray):
        return x[0] ** 2


    def ploho_grad(x: np.ndarray):
        return np.array([2 * x[0], 0])

    return f"f(x_1, x_2) = x_1^2", ploho, ploho_grad



def get_simple_func(k: float):
    def ploho(x: np.ndarray) -> float:
        return k * x[0] ** 2 + x[1] ** 2

    def ploho_grad(x: np.ndarray) -> np.ndarray:
        return np.array([2 * k * x[0], 2 * x[1]])

    return f"f(x_1, x_2) = {k} * x_1^2 + x_2^2", ploho, ploho_grad


def get_gradient_descent_constant(alph: float):
    def wrapped(func: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x: float, y: float):
        return gradient_descent_constant(np.array([x, y]), alph, grad)

    return f"Градиентный спуск с постоянным шагом({alph})", wrapped


def get_gradient_descent_linear():
    def wrapped(func: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x: float, y: float):
        return gradient_descent_linear(np.array([x, y]), grad, func)

    return "Градиентный спуск с одномерным поиском (метод золотого сечения)", wrapped


def main():
    for func_name, func, grad in [get_short_func()]:
        for gradient_descent_name, gradient_descent in [
            #get_gradient_descent_constant(0.01),
            #get_gradient_descent_constant(0.05),
            #get_gradient_descent_constant(0.3)
            get_gradient_descent_linear()
        ]:
            print("=========================")
            print(func_name)
            print(gradient_descent_name)
            print("_________________________")

            x_min, x_max, y_min, y_max = -100, 100, -100, 100

            print("Сходимость относительно точки старта")
            plt.title(f"Сходимость относительно точки старта\nдля градиентного спуска")
            x_space = np.linspace(x_min, x_max, x_max - x_min + 1)
            y_space = np.linspace(y_min, y_max, x_max - x_min + 1)
            convergence = [[gradient_descent(func, grad, x, y)[0] for x in x_space] for y in y_space]
            plt.contourf(x_space, y_space, convergence)
            blue_patch = Patch(color='royalblue', label='Мн-во точек, в которых гр. с. не сошёлся')
            green_patch = Patch(color='limegreen', label='Мн-во точек, в которых гр. с. сошёлся')
            plt.legend(handles=[blue_patch, green_patch], loc='upper right', fontsize="xx-small")
            plt.show()

            x, y = 0, 20
            print(f"Градиентный спуск из точки ({x}, {y})")
            conv, dots = gradient_descent(func, grad, x, y)
            print(conv, dots[-1])
            plt.title(f"Градиентный спуск\nиз точки ({x}, {y})")
            show_2arg_func(func, dots, levels=False, contour=True, label="Траектория градиентного спуска")

            for r in [5, 20, 50]:
                cnt = 8
                all_dots = None
                print(
                    f"Шаги градиентного спуска относительно точки старта ({cnt} точек старта на расстоянии {r} от минимума")
                plt.title(
                    f"Шаги градиентного спуска относительно\nточки старта ({cnt} точек старта на расстоянии {r} от минимума)")
                for i in range(cnt):
                    color = (i / cnt, 1 - i / cnt, 0)
                    x = r * math.cos(math.pi * i / cnt * 2)
                    y = r * math.sin(math.pi * i / cnt * 2)
                    _, dots = gradient_descent(func, grad, x, y)
                    all_dots = np.concatenate((all_dots, dots)) if all_dots is not None else dots
                    show_2arg_func(func, dots, show=False, color=color,
                                   label=f"Траектория гр. с. из т. ({'%.2f' % x}, {'%.2f' % y})")
                show_2arg_func(func, all_dots, dots_show=False, levels=False, contour=True)
            print("=========================")

    # show_2arg_func_slice(func, dots_show=False)
    # show_2arg_func(func, gradient_descent_constant(np.array([20, 20]), 0.20, grad)[1], contour=False, show=True)
    # plt.show()

    # alph = 0.055
    # shodimost = [[gradient_descent_constant(np.array([i, j]), alph, grad)[0] for j in range(-100, 100, 10)]
    #        for i in range(-100, 100, 10)]
    # shodimost.insert(0, [-100 + 10 * i for i in range(len(shodimost))])
    # for i, line in zip(range(len(shodimost)), shodimost):
    #     line.insert(0, -100 + 10 * i)
    # for line in shodimost:
    #     for elem in line:
    #         print(str(elem).ljust(5, " "), end=" ")
    #     print()

    # func, grad = get_simple_func(90)
    # func, grad = generate_function(2, 2)
    # start_dot = np.array([20, 20])
    # ans, dots = gradient_descent_linear_with_wolfe_condition(start_dot, grad, func, 0.1, 0.9)
    # ans, dots = gradient_descent_linear(start_dot, grad, func)
    # start_dot = np.array([-20, -20])
    # ans1, dots1 = gradient_descent_linear(start_dot, grad, func)

    # ans, dots = gradient_descent_constant(start_dot, 0.03, grad)
    # print(f"{'Сошлось' if ans else 'Не сошлось'} за {len(dots)} шагов")
    # show_2arg_func(func, dots, levels=False)
    # show_2arg_func(func, dots, levels=False, contour=False, show=False)
    # show_2arg_func(func, dots1, levels=False, contour=False, show=False)
    # show_2arg_func(func, np.concatenate([dots, dots1]), dots_show=False, levels=False, contour=True)
    # show_2arg_func_levels(func, np.concatenate([dots, dots1]))
    # show_2arg_func_contour(func, show=True)


if __name__ == '__main__':
    main()

import math

import matplotlib.pyplot as plt
import numpy as np

from tasks.data_generator import generate_linear_noisy_data


def approx_fun_1(x, a, b):
    return a * x + b


def approx_fun_2(x, a, b):
    return a / (1 + b * x)


def least_squares(x, y, approx_fun, search_fun, e=0.001):
    return search_fun(lambda a, b: np.sum((approx_fun(x, a, b) - y) ** 2), e)


def exhaustive_search_1(fun, boundaries, e):
    a, b = boundaries
    n = math.ceil((b - a) / e)

    min_val_x, min_val = a, fun(a)

    for k in range(n + 1):
        x = a + k * (b - a) / n
        val = fun(x)
        if val < min_val:
            min_val_x, min_val = x, val

    return min_val_x, min_val


def exhaustive_search_2(fun, e, boundaries=None):
    if boundaries is None:
        boundaries = (-2, 2, -2, 2)

    a0, b0, a1, b1 = boundaries
    n0 = math.ceil((b0 - a0) / e)
    n1 = math.ceil((b1 - a1) / e)

    min_val_x0, min_val_x1, min_val = a0, a1, fun(a0, a1)

    for k0 in range(n0 + 1):
        print(k0)
        x0 = a0 + k0 * (b0 - a0) / n0
        for k1 in range(n1 + 1):
            x1 = a1 + k1 * (b1 - a1) / n1
            val = fun(x0, x1)
            if val < min_val:
                min_val_x0, min_val_x1, min_val = x0, x1, val

    return min_val_x0, min_val_x1, min_val


def gauss(fun, e, initial=None, boundaries=None):
    if boundaries is None:
        boundaries = (-2, 2, -2.05, 2)

    if initial is None:
        x_1 = 1
        x_2 = 1
    else:
        x_1, x_2 = initial

    x_1_min, x_1_max, x_2_min, x_2_max = boundaries

    x_1_next, _ = exhaustive_search_1(lambda x: fun(x, x_2), (x_1_min, x_1_max), e)
    x_2_next, _ = exhaustive_search_1(lambda x: fun(x_1_next, x), (x_2_min, x_2_max), e)

    while abs(fun(x_1_next, x_2_next) - fun(x_1, x_2)) >= e:
        x_1 = x_1_next
        x_2 = x_2_next
        x_1_next, _ = exhaustive_search_1(lambda x: fun(x, x_2), (x_1_min, x_1_max), e)
        x_2_next, _ = exhaustive_search_1(lambda x: fun(x_1_next, x), (x_2_min, x_2_max), e)

    return x_1_next, x_2_next, fun(x_1_next, x_2_next)


def nelder_mead(fun, e, alpha=1, beta=0.5, gamma=2, initial=None):
    if initial is None:
        x_1 = np.array([-2, -2])
        x_2 = np.array([2, 2])
        x_3 = np.array([2, -2])
    else:
        x_1, x_2, x_3 = initial

    x_l, f_l = x_1, fun(*x_1)
    x_g, f_g = x_2, fun(*x_2)
    x_h, f_h = x_3, fun(*x_3)

    def sort():
        nonlocal x_l, f_l, x_g, f_g, x_h, f_h
        s = [(x_l, f_l), (x_g, f_g), (x_h, f_h)]
        s.sort(key=lambda t: t[1])
        x_l, f_l = s[0]
        x_g, f_g = s[1]
        x_h, f_h = s[2]
        return

    def shrink():
        nonlocal x_h, f_h, x_g, f_g
        x_s = beta * x_h + (1 - beta) * x_c
        f_s = fun(*x_s)

        if f_s < f_h:
            x_h, f_h = x_s, f_s
        else:
            x_g = x_l + (x_g - x_l) / 2
            x_h = x_l + (x_h - x_l) / 2
            f_g = fun(*x_g)
            f_h = fun(*x_h)
        return

    while (np.std([x_l, x_g, x_h], axis=0) >= e).all():
        sort()

        x_c = 0.5 * (x_l + x_g)
        x_r = (1 + alpha) * x_c - alpha * x_h
        f_r = fun(*x_r)

        if f_r < f_l:
            x_e = (1 - gamma) * x_c + gamma * x_r
            f_e = fun(*x_e)
            if f_e < f_r:
                x_h, f_h = x_e, f_e
                continue
            else:
                x_h, f_h = x_r, f_r
                continue

        if f_l < f_r < f_g:
            x_h, f_h = x_r, f_r
            continue

        if f_g < f_r < f_h:
            x_r, x_h = x_h, x_r
            f_r, f_h = f_h, f_r
            shrink()
            continue

        if f_h < f_r:
            shrink()
            continue

    result_x = (x_l + x_g + x_h) / 3
    return result_x[0], result_x[1], fun(*result_x)


def dichotomy(fun, boundaries, e, delta_fun=lambda e: e / 2):
    a, b = boundaries
    delta = delta_fun(e)
    x_1 = (a + b - delta) / 2
    x_2 = (a + b + delta) / 2

    while abs(a - b) >= e:
        if fun(x_1) <= fun(x_2):
            b = x_2
        else:
            a = x_1

        x_1 = (a + b - delta) / 2
        x_2 = (a + b + delta) / 2

    x_star = (a + b) / 2
    return x_star, fun(x_star)


def golden(fun, boundaries, e):
    a, b = boundaries
    x_1 = a + (3 - math.sqrt(5)) / 2 * (b - a)
    x_2 = b + (math.sqrt(5) - 3) / 2 * (b - a)

    while abs(a - b) >= e:
        if fun(x_1) <= fun(x_2):
            b = x_2
            x_2 = x_1
            x_1 = a + (3 - math.sqrt(5)) / 2 * (b - a)
        else:
            a = x_1
            x_1 = x_2
            x_2 = b + (math.sqrt(5) - 3) / 2 * (b - a)

    x_star = (a + b) / 2
    return x_star, fun(x_star)


def fun_1(x):
    return x * x * x


def fun_2(x):
    return np.abs(x - 0.2)


def fun_3(x):
    return x * np.sin(1 / x)


def task_1():
    funcs = {
        fun_1: ([0, 1]),
        fun_2: ([0, 1]),
        fun_3: ([0.01, 1])
    }

    for fun, (boundaries) in funcs.items():
        res0 = exhaustive_search_1(fun, boundaries, 0.001)
        res1 = dichotomy(fun, boundaries, 0.001)
        res2 = golden(fun, boundaries, 0.001)


if __name__ == '__main__':
    x_exp, y_exp = generate_linear_noisy_data(100)

    a, b, _ = least_squares(x_exp, y_exp, approx_fun_1, nelder_mead)

    plt.plot(x_exp, y_exp, label="Experimental")
    plt.plot(x_exp, approx_fun_1(x_exp, a, b), label="Approximation")
    plt.legend()
    plt.show()

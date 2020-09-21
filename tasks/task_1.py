import random
import time
from decimal import Decimal, localcontext

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def generate_vec(n):
    return [random.uniform(0, 1) for _ in range(n)]


def generate_mat(n):
    return [[random.uniform(0, 1) for _ in range(n)] for _ in range(n)]


def partition(v, low, high):
    i = (low - 1)
    pivot = v[high]

    for j in range(low, high):
        if v[j] <= pivot:
            i = i + 1
            v[i], v[j] = v[j], v[i]

    v[i + 1], v[high] = v[high], v[i + 1]
    return i + 1


def quick_sort(v, low, high):
    if len(v) == 1:
        return v

    if low < high:
        pi = partition(v, low, high)
        quick_sort(v, low, pi - 1)
        quick_sort(v, pi + 1, high)

    return v


def fun_1(v):
    return 1


def fun_2(v):
    result = 0
    for e in v:
        result = result + e
    return result


def fun_3(v):
    result = 1
    for e in v:
        result = result * e
    return result


def fun_4_1(v):
    with localcontext() as ctx:
        ctx.prec = 100
        x = Decimal(1.5)
        p = Decimal(0)

        for i, e in enumerate(v):
            p += Decimal(e) * (x ** i)

        return p


def fun_4_2(v):
    with localcontext() as ctx:
        ctx.prec = 100
        x = Decimal(1.5)
        p = Decimal(v[-1])
        i = len(v) - 2
        while i >= 0:
            p = p * x + Decimal(v[i])
            i -= 1
        return p


def fun_5(v):
    n = len(v)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if v[j] > v[j + 1]:
                v[j], v[j + 1] = v[j + 1], v[j]

    return v


def fun_6(v):
    n = len(v)
    return quick_sort(v, 0, n - 1)


def fun_7(v):
    return v.sort()


def mat_product(args):
    a, b = args
    return np.matmul(a, b)


def black_hole(arg):
    return


def dummy_benchmark(fun, generator, n, runs):
    stats = [0] * n

    for i in range(n):
        args = [generator(i + 1) for _ in range(runs)]

        t1 = time.perf_counter_ns()
        for j in range(runs):
            black_hole(fun(args[j]))
        t2 = time.perf_counter_ns()

        stats[i] = (t2 - t1) / runs

        if i % (n / 10) == 0:
            print('Step ', i, 'of ', fun, ' execution completed!')

    return stats


def fit_const(n, c):
    return np.full(n.shape, c)


def fit_linear(n, a, b):
    return a * n + b


def fit_quadratic(n, a, b, c):
    return a * n ** 2 + b * n + c


def fit_n_log(n, a, b):
    return a * n * np.log(n) + b


def fit_cubic(n, a, d):
    return a * n * n * n + d


if __name__ == '__main__':
    N = 2000
    n_range = np.arange(1, N + 1)
    benchmarks = {
        'fun_1': (fun_1, generate_vec, fit_const, [0]),
        'fun_2': (fun_2, generate_vec, fit_linear, [1, 0]),
        'fun_3': (fun_3, generate_vec, fit_linear, [1, 0]),
        'fun_4_1': (fun_4_1, generate_vec, fit_n_log, [1, 1]),
        'fun_4_2': (fun_4_2, generate_vec, fit_linear, [1, 0]),
        'fun_5': (fun_5, generate_vec, fit_quadratic, [1, 1, 0]),
        'fun_6': (fun_6, generate_vec, fit_n_log, [1, 1]),
        'fun_7': (fun_7, generate_vec, fit_n_log, [1, 1]),
        'mat_product': (mat_product, lambda n: (np.random.rand(n, n), np.random.rand(n, n)), fit_cubic, [1, 0])
    }

    for key, (fun, generator, fitting_func, initial_guess) in benchmarks.items():
        benchmarks_stats = dummy_benchmark(fun, generator, N, 5)
        benchmarks_stats = np.array(benchmarks_stats)

        pars, _ = curve_fit(f=fitting_func,
                            xdata=n_range, ydata=benchmarks_stats,
                            p0=initial_guess
                            )

        plt.plot(n_range, benchmarks_stats, label="Experimental")
        plt.plot(n_range, fitting_func(n_range, *pars), label="Approximation")
        plt.title(key)
        plt.legend()
        plt.show()

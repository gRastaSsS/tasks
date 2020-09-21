import random

import numpy as np


def generate_linear_noisy_data(n=100):
    a = random.uniform(0, 1)
    b = random.uniform(0, 1)
    x = [k / n for k in range(n)]
    y = [a * x[k] + b + np.random.normal() for k in range(n)]
    return np.array(x), np.array(y)


def generate_noisy_data(n=1000):
    def f(x):
        return 1 / (x**2 - 3*x + 2)

    def y_f(x):
        if f(x) < -100:
            return -100 + np.random.normal()
        elif -100 <= f(x) <= 100:
            return f(x) + np.random.normal()
        else:
            return 100 + np.random.normal()

    x = [3 * k / n for k in range(n)]
    y = [y_f(x_k) for x_k in x]
    return np.array(x), np.array(y)


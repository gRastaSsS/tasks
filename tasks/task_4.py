import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import dual_annealing
from scipy.optimize import differential_evolution

from tasks.data_generator import generate_noisy_data


def a_fun(x, p):
    a, b, c, d = p
    return (a * x + b) / (x ** 2 + c * x + d)


def a_fun_unpacked(x, a, b, c, d):
    return (a * x + b) / (x ** 2 + c * x + d)


def get_ls_fun(x, y, fun):
    return lambda p: np.sum((fun(x, p) - y) ** 2)


if __name__ == '__main__':
    x_exp, y_exp = generate_noisy_data(1000)

    p_nm = minimize(
        get_ls_fun(x_exp, y_exp, a_fun),
        np.array([1, 1, 1, 1]),
        method='Nelder-Mead', options={'fatol': 0.001, 'maxiter': 1000}
    ).x

    p_lm, _ = curve_fit(
        a_fun_unpacked, x_exp, y_exp,
        p0=np.array([1, 1, 1, 1]),
        method='lm', xtol=0.001, maxfev=1000
    )

    p_sa = dual_annealing(
        get_ls_fun(x_exp, y_exp, a_fun),
        x0=np.array([1, 1, 1, 1]),
        bounds=np.array([[-100, 100], [-100, 100], [-100, 100], [-100, 100]]),
        maxiter=1000
    ).x

    p_de = differential_evolution(
        get_ls_fun(x_exp, y_exp, a_fun),
        bounds=np.array([[-100, 100], [-100, 100], [-100, 100], [-100, 100]]),
        maxiter=1000
    ).x

    plt.plot(x_exp, y_exp, label="Experimental")
    plt.plot(x_exp, a_fun(x_exp, p_nm), label="Nelder-Mead")
    plt.plot(x_exp, a_fun(x_exp, p_lm), label="LM")
    plt.plot(x_exp, a_fun(x_exp, p_sa), label="Annealing")
    plt.plot(x_exp, a_fun(x_exp, p_de), label="Differential Evolution")
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares
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


def get_ls_fun_sqrt(x, y, fun):
    return lambda p: (fun(x, p) - y)


if __name__ == '__main__':
    x_exp, y_exp = generate_noisy_data(1000)

    stats_nm = minimize(
        get_ls_fun(x_exp, y_exp, a_fun),
        np.array([1, 1, 1, 1]),
        method='Nelder-Mead', options={'fatol': 0.001, 'maxiter': 1000}
    )

    stats_lm = least_squares(
        get_ls_fun_sqrt(x_exp, y_exp, a_fun),
        np.array([1, 1, 1, 1]),
        xtol=0.001,
        method='lm'
    )

    stats_sa = dual_annealing(
        get_ls_fun(x_exp, y_exp, a_fun),
        x0=np.array([1, 1, 1, 1]),
        bounds=np.array([[-200, 200], [-200, 200], [-200, 200], [-200, 200]]),
        maxiter=1000
    )

    stats_de = differential_evolution(
        get_ls_fun(x_exp, y_exp, a_fun),
        tol=0.001,
        bounds=np.array([[-2, 2], [-2, 2], [-2, 2], [-2, 2]]),
        maxiter=1000
    )

    print("Statistics\n")
    print('Nelder-Mead\n', stats_nm)
    print('LM\n', stats_lm)
    print('Annealing\n', stats_sa, 'Precision =', np.sum((stats_nm.x - stats_sa.x)**2))
    print('Differential Evolution\n', stats_de, 'Precision =', np.sum((stats_nm.x - stats_de.x)**2))

    plt.plot(x_exp, y_exp, label="Experimental")
    plt.plot(x_exp, a_fun(x_exp, stats_nm.x), label="Nelder-Mead")
    plt.plot(x_exp, a_fun(x_exp, stats_lm.x), label="LM")
    plt.legend()
    plt.show()

    plt.plot(x_exp, y_exp, label="Experimental")
    plt.plot(x_exp, a_fun(x_exp, stats_sa.x), label="Annealing")
    plt.plot(x_exp, a_fun(x_exp, stats_de.x), label="Differential Evolution")
    plt.legend()
    plt.show()

    plt.plot(x_exp, y_exp, label="Experimental")
    plt.plot(x_exp, a_fun(x_exp, stats_nm.x), label="Nelder-Mead")
    plt.plot(x_exp, a_fun(x_exp, stats_lm.x), label="LM")
    plt.plot(x_exp, a_fun(x_exp, stats_sa.x), label="Annealing")
    plt.plot(x_exp, a_fun(x_exp, stats_de.x), label="Differential Evolution")
    plt.legend()
    plt.show()

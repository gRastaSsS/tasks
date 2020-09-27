import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

from tasks.data_generator import generate_linear_noisy_data
from tasks.task_2 import least_squares_fun, gauss, nelder_mead


def a_fun_1(x, p):
    return p[0] * x + p[1]


def a_fun_2(x, p):
    return p[0] / (1 + p[1] * x)


def get_ls_functions_1(x, y):
    def fun(p):
        return np.sum((a_fun_1(x, p) - y) ** 2)

    def dfun(p):
        x_0 = 2 * np.sum((p[0] * x + p[1] - y) * x)
        x_1 = 2 * np.sum(p[0] * x + p[1] - y)
        return np.array([x_0, x_1])

    def fun_least_squares(p):
        return a_fun_1(x, p) - y

    def dfun_least_squares(p):
        x_0 = x
        x_1 = np.full(x.shape, 1)
        return np.array([x_0, x_1]).transpose()

    def unpacked(x, a, b):
        return a_fun_1(x, [a, b])

    return fun, dfun, fun_least_squares, dfun_least_squares, unpacked


def get_ls_functions_2(x, y):
    def fun(p):
        return np.sum((a_fun_2(x, p) - y) ** 2)

    def dfun(p):
        a, b = p[0], p[1]
        x_0 = -2 * np.sum((-a+b*x*y+y)/(b*x+1)**2)
        x_1 = -2 * np.sum((a*x*(a-y*(b*x+1)))/(b*x+1)**3)
        return np.array([x_0, x_1])

    def fun_least_squares(p):
        return a_fun_2(x, p) - y

    def dfun_least_squares(p):
        x_0 = 1 / (p[1]*x + 1)
        x_1 = - (p[0] * x) / (p[1] * x + 1)**2
        return np.array([x_0, x_1]).transpose()

    def unpacked(x, a, b):
        return a_fun_2(x, [a, b])

    return fun, dfun, fun_least_squares, dfun_least_squares, unpacked


def grad(fun, dfun, a0, e=0.001):
    pre_a = a0
    res = minimize_scalar(lambda p: fun(pre_a - p * dfun(pre_a)))
    rate = res.x
    cur_a = a0 - rate * dfun(a0)

    f_calc_counter = res.nfev
    df_calc_counter = 1
    iter_counter = 0

    while norm(pre_a - cur_a) >= e:
        pre_a = cur_a

        res = minimize_scalar(lambda p: fun(pre_a - p * dfun(pre_a)))
        rate = res.x
        cur_a = cur_a - rate * dfun(pre_a)

        f_calc_counter += res.nfev
        df_calc_counter += 1
        iter_counter += 1

    return {
        'x': cur_a,
        'f_calcs': f_calc_counter,
        'df_calcs': df_calc_counter,
        'iters': iter_counter
    }


if __name__ == '__main__':
    x_exp, y_exp = generate_linear_noisy_data(100)

    for a_fun_real, ls_funs, label in [(a_fun_1, get_ls_functions_1, 'Linear approximation'), (a_fun_2, get_ls_functions_2, 'Rational approximation')]:
        a_fun, a_dfun, a_fun_cv, a_dfun_cv, a_fun_unpacked = ls_funs(x_exp, y_exp)

        stat_newton = minimize(a_fun, np.array([0, 0]), jac=a_dfun, method='Newton-CG', options={'xtol': 0.001})
        stat_cg = minimize(a_fun, np.array([0, 0]), jac=a_dfun, method='CG', tol=0.001)
        stat_lm = least_squares(a_fun_cv, np.array([0, 0]), jac=a_dfun_cv, xtol=0.001, method='lm')
        stat_grad = grad(a_fun, a_dfun, np.array([0, 0]), e=0.001)

        a_ga, b_ga, _ = least_squares_fun(x_exp, y_exp, a_fun_unpacked, gauss)
        a_nm, b_nm, _ = least_squares_fun(x_exp, y_exp, a_fun_unpacked, nelder_mead)

        print("Statistics", label)
        print("LM:", stat_lm)
        print("Grad:", stat_grad)
        print("Newton:", stat_newton)
        print("CG:", stat_cg)
        print('\n')

        plt.plot(x_exp, y_exp, label="Experimental")
        plt.plot(x_exp, a_fun_real(x_exp, stat_lm.x), label="LM")
        plt.plot(x_exp, a_fun_real(x_exp, stat_grad['x']), label="Grad")
        plt.plot(x_exp, a_fun_real(x_exp, stat_newton.x), label="Newton")
        plt.plot(x_exp, a_fun_real(x_exp, stat_cg.x), label="CG")
        #plt.plot(x_exp, a_fun_unpacked(x_exp, a_nm, b_nm), label="Nelder-Mead")
        #plt.plot(x_exp, a_fun_unpacked(x_exp, a_ga, b_ga), label="Gauss")
        plt.legend()
        plt.title(label)
        plt.show()

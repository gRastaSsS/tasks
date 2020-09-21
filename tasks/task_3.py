import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

from tasks.data_generator import generate_linear_noisy_data


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

    def fun_curve_fit(x, a, b):
        return a_fun_1(x, [a, b])

    def dfun_curve_fit(x, a, b):
        x_0 = x
        x_1 = np.full(x.shape, 1)
        return np.array([x_0, x_1]).transpose()

    return fun, dfun, fun_curve_fit, dfun_curve_fit


def get_ls_functions_2(x, y):
    def fun(p):
        return np.sum((a_fun_2(x, p) - y) ** 2)

    def dfun(p):
        a, b = p[0], p[1]
        x_0 = -2 * np.sum((-a+b*x*y+y)/(b*x+1)**2)
        x_1 = -2 * np.sum((a*x*(a-y*(b*x+1)))/(b*x+1)**3)
        return np.array([x_0, x_1])

    def fun_curve_fit(x, a, b):
        return a_fun_2(x, [a, b])

    def dfun_curve_fit(x, a, b):
        x_0 = 1 / (b*x + 1)
        x_1 = - (a * x) / (b * x + 1)**2
        return np.array([x_0, x_1]).transpose()

    return fun, dfun, fun_curve_fit, dfun_curve_fit


def grad(fun, dfun, a0, e=0.001):
    pre_a = a0
    rate = minimize_scalar(lambda p: fun(pre_a - p * dfun(pre_a))).x
    cur_a = a0 - rate * dfun(a0)

    while norm(pre_a - cur_a) >= e:
        pre_a = cur_a
        rate = minimize_scalar(lambda p: fun(pre_a - p * dfun(pre_a))).x
        cur_a = cur_a - rate * dfun(pre_a)

    return cur_a


if __name__ == '__main__':
    x_exp, y_exp = generate_linear_noisy_data(100)

    a_fun, a_dfun, a_fun_cv, a_dfun_cv = get_ls_functions_2(x_exp, y_exp)

    p_newton = minimize(a_fun, np.array([1, 1]), jac=a_dfun, method='Newton-CG', options={'xtol': 0.001}).x
    p_cg = minimize(a_fun, np.array([1, 1]), jac=a_dfun, method='CG', tol=0.001).x
    p_lm, _ = curve_fit(a_fun_cv, x_exp, y_exp, method='lm', p0=[1, 1], jac=a_dfun_cv, xtol=0.001)
    p_grad = grad(a_fun, a_dfun, np.array([1, 1]), e=0.001)

    print(p_newton, p_cg, p_lm, p_grad)

    plt.plot(x_exp, y_exp, label="Experimental")
    plt.plot(x_exp, a_fun_2(x_exp, p_lm), label="LM")
    plt.plot(x_exp, a_fun_2(x_exp, p_grad), label="Grad")
    plt.plot(x_exp, a_fun_2(x_exp, p_newton), label="Newton")
    plt.plot(x_exp, a_fun_2(x_exp, p_cg), label="CG")
    plt.legend()
    plt.show()

    #params = [p_newton, p_cg, p_lm]

    #for p in params:
    #    plt.plot(x_exp, y_exp, label="Experimental")
    #    plt.plot(x_exp, a_fun_1(x_exp, p), label="Approximation")
    #    plt.legend()
    #    plt.show()

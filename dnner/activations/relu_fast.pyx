# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport log, exp, sqrt, M_PI
from scipy.special.cython_special cimport log1p, erfc, erfcx


cdef double _log_erfc(double x):
    """ Compute log(erfc(x)), using expansion for large x
    """
    cdef double corr
    cdef double res

    if x < 20:
        res = log(erfc(x))
    else:  # Taylor expansion around $x = \infty$
        corr = -.5 / x ** 2 + .75 / x ** 4 - 1.875 / x ** 6 + \
            6.5625 / x ** 8 - 29.53125 / x ** 10
        res = -x ** 2 - log(x) - .5 * log(M_PI) + log1p(corr)
    return res


cdef double _v_x(double a, double b, double w, double v_eff, double gamma):
    r""" Compute $\partial^2_B \log Z$ for leaky ReLU interface
    """
    cdef double mu_b_p, mu_b_n, sig_b_p, sig_b_n
    cdef double phi_p, phi_n, lnz_p, lnz_n, logit_p
    cdef double s_p, s_n, r_p, r_n, p_p, f_p, df_p, f_n, df_n

    # Pre-compute useful quantities
    phi_p = (b + w / v_eff) / sqrt(2 * (a + 1. / v_eff))
    phi_n = (gamma * b + w / v_eff) / sqrt(2 * (gamma ** 2 * a + 1. / v_eff))

    lnz_p = phi_p ** 2 + _log_erfc(-phi_p) - .5 * log1p(a * v_eff)
    lnz_n = phi_n ** 2 + _log_erfc(+phi_n) - .5 * log1p(gamma ** 2 * a * v_eff)
    logit_p = lnz_p - lnz_n

    s_p = sqrt(2 * M_PI * (a + 1 / v_eff))
    s_n = sqrt(2 * M_PI * (gamma ** 2 * a + 1 / v_eff))
    r_p = 2. / (erfcx(-phi_p) * s_p)
    r_n = -2. / (erfcx(phi_n) * s_n)

    # Quantities specific for _var_x
    mu_b_p = (b + w / v_eff) / (a + 1. / v_eff)
    mu_b_n = gamma * (gamma * b + w / v_eff) / (gamma ** 2 * a + 1. / v_eff)
    sig_b_p = 1. / (a + 1. / v_eff)
    sig_b_n = gamma ** 2 / (gamma ** 2 * a + 1. / v_eff)

    # Final quantities
    p_p = 1 / (1 + exp(-logit_p))  # careful with -logit_p too large

    f_p = mu_b_p + r_p
    df_p = sig_b_p - (2 * r_p * phi_p / s_p + r_p ** 2)
    f_n = mu_b_n + gamma * r_n
    df_n = sig_b_n - gamma ** 2 * (2 * r_n * phi_n / s_n + r_n ** 2)

    return p_p * df_p + (1 - p_p) * df_n + p_p * (1. - p_p) * (f_p - f_n) ** 2


cdef double _v_z(double a, double b, double w, double v_eff, double gamma):
    r"""Compute $\partial^2_w \log Z$ for leaky ReLU interface
    """
    cdef double mu_b_p, mu_b_n, sig_b_p, sig_b_n
    cdef double phi_p, phi_n, lnz_p, lnz_n, logit_p
    cdef double s_p, s_n, r_p, r_n, p_p, g_p, dg_p, g_n, dg_n

    # Pre-compute useful quantities
    phi_p = (b + w / v_eff) / sqrt(2 * (a + 1. / v_eff))
    phi_n = (gamma * b + w / v_eff) / sqrt(2 * (gamma ** 2 * a + 1. / v_eff))

    lnz_p = phi_p ** 2 + _log_erfc(-phi_p) - .5 * log1p(a * v_eff)
    lnz_n = phi_n ** 2 + _log_erfc(+phi_n) - .5 * log1p(gamma ** 2 * a * v_eff)
    logit_p = lnz_p - lnz_n

    s_p = sqrt(2 * M_PI * (a + 1 / v_eff))
    s_n = sqrt(2 * M_PI * (gamma ** 2 * a + 1 / v_eff))
    r_p = 2 / (erfcx(-phi_p) * s_p)
    r_n = -2 / (erfcx(phi_n) * s_n)

    # Quantities specific for _var_z
    mu_w_p = (b / a - w) / (1 / a + v_eff)
    mu_w_n = gamma * (b / a - gamma * w) / (1 / a + gamma ** 2 * v_eff)
    sig_w_p = 1 / (1 / a + v_eff)
    sig_w_n = gamma ** 2 / (1 / a + gamma ** 2 * v_eff)

    # Final quantities
    p_p = 1 / (1 + exp(-logit_p))  # careful with -logit_p too large

    g_p = mu_w_p + r_p / v_eff
    dg_p = sig_w_p + (2 * r_p * phi_p / s_p + r_p ** 2) / v_eff ** 2
    g_n = mu_w_n + r_n / v_eff
    dg_n = sig_w_n + (2 * r_n * phi_n / s_n + r_n ** 2) / v_eff ** 2

    return p_p * dg_p + (1 - p_p) * dg_n - p_p * (1. - p_p) * (g_p - g_n) ** 2


cdef double _lnz(double a, double b, double w, double v_eff, double gamma):
    r"""Compute $\log Z$ for leaky ReLU interface
    """
    cdef double arg_z1, arg_z2, z_r
    cdef double res

    arg_z1 = -(b + w / v_eff) / sqrt(2 * (a + 1 / v_eff))
    arg_z2 = (gamma * b + w / v_eff) / sqrt(2 * (gamma ** 2 * a + 1 / v_eff))

    if arg_z1 > -25 and arg_z2 > -25:
        z_r = erfcx(arg_z1) / sqrt(1 + a * v_eff) + \
            erfcx(arg_z2) / sqrt(1 + gamma ** 2 * a * v_eff)
        res = log(z_r / 2)
    else:  # log(erfcx(-x)) ~ x ** 2 + log(2)
        if arg_z1 < arg_z2:
            res = arg_z1 ** 2 - .5 * log(1 + a * v_eff)
        else:
            res = arg_z2 ** 2 - .5 * log(1 + gamma ** 2 * a * v_eff)
    res -= .5 * w ** 2 / v_eff

    return res


cdef double _z(double a, double b, double w, double v_eff, double gamma, int x):
    r"""Compute $Z$ for leaky ReLU interface
    """
    cdef double res

    res = gamma * b + w / v_eff
    res /= sqrt(2 * (gamma ** 2 * a + 1 / v_eff))
    res = .5 * erfc(x * res)
    return res


def _f(double b, double w, double a, double m, double v_eff, double gamma,
              int fun, int term):
    r"""Function $z f$ to be integrated over Gaussian measure"""
    cdef double s_w, s_b, w_, b_
    cdef double z, f
    cdef double res

    s_w = sqrt(m)

    # Pick between $z_+^{(1)}$ and $z_-^{(\gamma)}$
    if term == 0:
        s_b = sqrt(a * (1 + a * v_eff))

        w_ = s_w * w
        b_ = a * w_ + s_b * b

        z = _z(a, b_, w_, v_eff, 1.0, -1)
    else:
        s_b = sqrt(a * (1 + gamma ** 2 * a * v_eff))

        w_ = s_w * w
        b_ = gamma * a * w_ + s_b * b

        z = _z(a, b_, w_, v_eff, gamma, +1)

    # Pick $f$ according to value of fun
    if fun == 0:
        f = _v_z(a, b_, w_, v_eff, gamma)
    elif fun == 1:
        f = _v_x(a, b_, w_, v_eff, gamma)
    else:
        f = _lnz(a, b_, w_, v_eff, gamma)

    res = z * f
    res *= exp(-.5 * (b ** 2 + w ** 2)) / (2 * M_PI)
    return res

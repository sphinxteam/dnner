# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport log, exp, sqrt, M_PI
from scipy.special.cython_special cimport log1p, erf, erfc, erfcx


cdef double _v_x(double a, double b, double w, double v_eff, double thres):
    r""" Compute $\partial^2_B \log Z$ for hard-tanh interface
    """
    cdef double res, dz0, dz1, ddz0, ddz1
    cdef double b_r, a_r, w_r, v_r
    cdef double m_b, dm_b
    cdef double n_m, n_p, e_m, e_p
    cdef double zg, z0, z1, z

    # Pre-compute some quantities (same for _v_x and _v_z)
    b_r = b + w / v_eff
    a_r = a + 1 / v_eff
    w_r = w - b / a
    v_r = v_eff + 1 / a

    n_m = exp(-.5 * (b_r + a_r * thres) ** 2 / a_r) / sqrt(2 * M_PI * a_r)
    n_p = exp(-.5 * (b_r - a_r * thres) ** 2 / a_r) / sqrt(2 * M_PI * a_r)

    zg = exp(-.5 * w_r ** 2 / v_r) / sqrt(1 + a * v_eff)
    z0 = _z(a, b, w, v_eff, thres, 0, 1)
    z1 = _z(a, b, w, v_eff, thres, 1, 1) + _z(a, b, w, v_eff, thres, 2, 1)
    z = z0 + z1

    # This is different for _v_x and _v_z
    m_b = b_r / a_r
    dm_b = 1 / a_r

    e_m = exp(-.5 * a * (b / a - thres) ** 2)
    e_p = exp(-.5 * a * (b / a + thres) ** 2)

    # 1st and 2nd derivatives of ln z_0
    dz0 = m_b * z0 + (n_m - n_p) * zg
    ddz0 = (m_b ** 2 + dm_b) * z0
    ddz0 += ((m_b - thres) * n_m - (m_b + thres) * n_p) * zg

    # 1st and 2nd derivatives of ln z_\pm
    dz1 = e_m * erfc((thres - w) / sqrt(2 * v_eff))
    dz1 -= e_p * erfc((thres + w) / sqrt(2 * v_eff))
    dz1 = thres * dz1 / 2
    ddz1 = thres ** 2 * z1

    # Put things together
    if z > 0:
        res = (ddz0 + ddz1) / z
        res -= ((dz0 + dz1) / z) ** 2
    else:
        res = 0

    return res


cdef double _v_z(double a, double b, double w, double v_eff, double thres):
    r"""Compute $-\partial^2_w \log Z$ for hard-tanh interface
    """
    cdef double res, dz0, dz1, ddz0, ddz1
    cdef double b_r, a_r, w_r, v_r
    cdef double m_w, dm_w, m_wb
    cdef double n_m, n_p, e_m, e_p
    cdef double zg, z0, z1, z

    # Pre-compute some quantities
    b_r = b + w / v_eff
    a_r = a + 1 / v_eff
    w_r = w - b / a
    v_r = v_eff + 1 / a

    n_m = exp(-.5 * (b_r + a_r * thres) ** 2 / a_r) / sqrt(2 * M_PI * a_r)
    n_p = exp(-.5 * (b_r - a_r * thres) ** 2 / a_r) / sqrt(2 * M_PI * a_r)

    zg = exp(-.5 * w_r ** 2 / v_r) / sqrt(1 + a * v_eff)
    z0 = _z(a, b, w, v_eff, thres, 0, 1)
    z1 = _z(a, b, w, v_eff, thres, 1, 1) + _z(a, b, w, v_eff, thres, 2, 1)
    z = z0 + z1

    # This is different for _v_x and _v_z
    m_w = -w_r / v_r
    dm_w = -1 / v_r
    m_wb = 2 * m_w * v_eff - (b_r / a_r)

    e_m = exp(-.5 * a * (b / a - thres) ** 2 - .5 * (w - thres) ** 2 / v_eff) / \
            sqrt(2 * M_PI * v_eff)
    e_p = exp(-.5 * a * (b / a + thres) ** 2 - .5 * (w + thres) ** 2 / v_eff) / \
            sqrt(2 * M_PI * v_eff)

    # 1st and 2nd derivatives of ln z_0
    dz0 = m_w * z0 + (n_m - n_p) * zg / v_eff
    ddz0 = (m_w ** 2 + dm_w) * z0
    ddz0 += ((m_wb - thres) * n_m - (m_wb + thres) * n_p) * zg / v_eff ** 2

    # 1st and 2nd derivatives of ln z_\pm
    dz1 = (e_m - e_p)
    ddz1 = ((thres - w) * e_m + (thres + w) * e_p) / v_eff

    # Put things together
    if z > 0:
        res = (ddz0 + ddz1) / z
        res -= ((dz0 + dz1) / z) ** 2
    else:
        res = 0

    return -res


cdef double _lnz(double a, double b, double w, double v_eff, double thres):
    r"""Compute $\log Z$ for hard-tanh interface
    """
    cdef double res
    
    res = _z(a, b, w, v_eff, thres, 0, 1)
    res += _z(a, b, w, v_eff, thres, 1, 1)
    res += _z(a, b, w, v_eff, thres, 2, 1)
    
    res = log(res) if res > 0 else 0
    res += .5 * b ** 2 / a

    return res


cdef double _z(double a, double b, double w, double v_eff, double thres, int x, int r):
    r"""Compute $Z$ for hard-tanh interface
    """
    cdef double res
    cdef double b_r, a_r, w_r, v_r

    if x == 0:
        # Compute k
        b_r = b + w / v_eff
        a_r = a + 1 / v_eff

        res = erf(sqrt(a_r / 2) * thres + b_r / sqrt(2 * a_r))
        res += erf(sqrt(a_r / 2) * thres - b_r / sqrt(2 * a_r))
        res /= 2

        # Compute z_0 = k * z_G
        if r == 1:
            w_r = w - b / a
            v_r = v_eff + 1 / a

            res *= exp(-.5 * w_r ** 2 / v_r) / sqrt(1 + a * v_eff)
    if x == 1:
        # Compute z_-
        res = .5 * erfc((thres + w) / sqrt(2 * v_eff))

        if r == 1:
            res *= exp(-.5 * (b + a * thres) ** 2 / a)
    if x == 2:
        # Compute z_+
        res = .5 * erfc((thres - w) / sqrt(2 * v_eff))

        if r == 1:
            res *= exp(-.5 * (b - a * thres) ** 2 / a)

    return res


def test(double a, double b, double w, double v_eff, double h):
    r"""Compare outputs of _v_x and _v_z to numerical derivatives of _lnz"""
    cdef double f0, f1b, f2b, f1w, f2w
    cdef double v_x1, v_x2, v_z1, v_z2

    f0 = _lnz(a, b, w, v_eff, 1)
    f1b = _lnz(a, b - h, w, v_eff, 1)
    f2b = _lnz(a, b + h, w, v_eff, 1)
    f1w = _lnz(a, b, w - h, v_eff, 1)
    f2w = _lnz(a, b, w + h, v_eff, 1)

    v_x1 = _v_x(a, b, w, v_eff, 1)
    v_x2 = (f1b + f2b - 2 * f0) / (h ** 2)
    v_z1 = _v_z(a, b, w, v_eff, 1)
    v_z2 = -(f1w + f2w - 2 * f0) / (h ** 2)

    print("lnz = %.8f" % (f0))
    print("v_x (eval/diff): %.8f, %.8f" % (v_x1, v_x2))
    print("v_z (eval/diff): %.8f, %.8f" % (v_z1, v_z2))


def _f(double b, double w, double a, double m, double v_eff, double thres,
              int fun, int term):
    r"""Function $z f$ to be integrated over Gaussian measure"""
    cdef double w_, b_
    cdef double z, f
    cdef double res

    w_ = sqrt(m) * w

    # Pick between $z_0$ (0), $z_+$ (1) and $z_-$ (2)
    if term == 0:
        b_ = a * w_ + sqrt(a * (1 + a * v_eff)) * b
        z = _z(a, b_, w_, v_eff, thres, 0, 0)
    elif term == 1:
        b_ = -a * thres + sqrt(a) * b
        z = _z(a, b_, w_, v_eff, thres, 1, 0)
    else:
        b_ = a * thres + sqrt(a) * b
        z = _z(a, b_, w_, v_eff, thres, 2, 0)
    z *= exp(-.5 * (b ** 2 + w ** 2)) / (2 * M_PI)

    # Pick $f$ according to value of fun
    if fun == 0:
        f = _v_z(a, b_, w_, v_eff, thres)
    elif fun == 1:
        f = _v_x(a, b_, w_, v_eff, thres)
    else:
        f = _lnz(a, b_, w_, v_eff, thres)

    res = z * f
    return res

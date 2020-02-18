import numpy as np


def exp10(decimal):
    '''Function that gets the exponent of a double number.
    '''
    string = "{:.10e}".format(decimal)
    parts = string.split('e')
    return int(parts[-1])


def fortran_double(number, dble=True):
    '''Function that returns a floating point (as a string) in the FORTRAN
    notation
    '''
    string = "{:.10e}".format(number)
    parts = string.split('e')

    ss = parts[0].split('.')
    s = ss[1]
    while s.endswith('0'):
        s = s[:-1]
    if (len(s) > 0):
        ss = ss[0] + '.' + s
    else:
        ss = ss[0]
    ee = parts[-1]
    if ee.startswith('-'):
        e = ee.split('-')[-1]
        while e.startswith('0'):
            e = e[1:]
        ee = '-' + e
    else:
        while ee.startswith('+') or ee.startswith('0'):
            ee = ee[1:]
    if (len(ee) == 0):
        ee = '0'
    if dble:
        return "{0}d{1}".format(ss, ee)
    else:
        return "{0}e{1}".format(ss, ee)


# def sci_notation(n, prec=3, fortran=False):
#     """Represent n in scientific notation, with the specified precision.
#     >>> sci_notation(1234 * 10**1000)
#     '1.234e+1003'
#     >>> sci_notation(10**1000 // 2, prec=1)
#     '5.0e+999'
#     """
#     exponent = int(np.log10(n))
#     if (n < 1):
#         exponent -= 1
#     mantissa = n / np.power(10, exponent)
#     if fortran:
#         return "{0:.{1}f}d{2:+d}".format(mantissa, prec, exponent)
#     else:
#         return "{0:.{1}f}e{2:+d}".format(mantissa, prec, exponent)


def find_nearest(arr, val):
    if (arr != np.ndarray):
        arr = np.asarray(arr)
    i = (np.abs(arr - val)).argmin()
    return i, arr[i]


#
#  #     #
#  #  #  # # ##### #####   ##   #    # ###### #####
#  #  #  # #   #     #    #  #  #   #  #      #    #
#  #  #  # #   #     #   #    # ####   #####  #    #
#  #  #  # #   #     #   ###### #  #   #      #####
#  #  #  # #   #     #   #    # #   #  #      #   #
#   ## ##  #   #     #   #    # #    # ###### #    #
#
def whittM(k, m, z):
    """Evaluates the Whitaker function M(k, m, z) as defined in Abramowitz &
    Stegun, Section 13.1.
    """
    from scipy.special import hyp1f1
    if k is int or m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyp1f1(0.5 + float(m - k), 1 + 2 * m, z)
    elif k is int and m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyp1f1(0.5 + float(m - k), 1 + 2 * m, z)
    else:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + m) * hyp1f1(0.5 + m - k, 1.0 + 2.0 * m, z)


def whittW(k, m, z):
    """Evaluates the Whitaker function W(k, m, z) as defined in Abramowitz &
    Stegun, Section 13.1.
    """
    from scipy.special import hyperu
    if k is int or m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyperu(0.5 + float(m - k), 1 + 2 * m, z)
    elif k is int and m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyperu(0.5 + float(m - k), 1 + 2 * m, z)
    else:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + m) * hyperu(0.5 + m - k, 1.0 + 2.0 * m, z)


#
#   #####
#  #     # #    # ###### #####  #   #  ####  #    # ###### #    #
#  #       #    # #      #    #  # #  #    # #    # #      #    #
#  #       ###### #####  #####    #   #      ###### #####  #    #
#  #       #    # #      #    #   #   #      #    # #      #    #
#  #     # #    # #      #    #   #   #    # #    # #       #  #
#   #####  #    # ###### #####    #    ####  #    # ######   ##
def chebev(x, c, xmin, xmax):
    d = 0.0
    dd = 0.0
    y = (2.0 * x - xmin - xmax) / (xmax - xmin)
    y2 = 2.0 * y
    j = c.size - 1
    while j > 0:
        sv = dd
        dd = d
        d = y2 * dd - sv + c[j]
        j -= 1
    return y * d - dd + 0.5 * c[0]


#   ####   ####  #       ####  #####   ####
#  #    # #    # #      #    # #    # #
#  #      #    # #      #    # #    #  ####
#  #      #    # #      #    # #####       #
#  #    # #    # #      #    # #   #  #    #
#   ####   ####  ######  ####  #    #  ####
def col_tab(num=1):
    if num == 1:
        #  matplotlib default
        return ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    elif num == 2:
        # xkcd colors
        return ['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:red', 'xkcd:purple', 'xkcd:brown', 'xkcd:pink', 'xkcd:gray', 'xkcd:olive', 'xkcd:cyan']

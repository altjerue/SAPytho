import numpy as np
import SAPytho.constants as C
import scipy.integrate as integrate
import SAPytho.spectra as sp


class iCompton:

    def __init__(self, **kwargs):
        self.Zq = 1.0
        self.mq = C.me
        self.re = ((C.eCharge)**2) / (self.mq * C.cLight**2)
        self.__dict__.update(kwargs)

# total cross section valid in all regimes or 'klein nishina formula'
    def cross_section(self, nu):

        # eq. (7.5) from Rybicki & Lightman (1979)
        x = (C.hPlanck * nu) / (self.mq * C.cLight**2)
        sig_k = (1 + x) * ((2 * x * (1 + x) / (1 + (2 * x))) - np.log(1 + (2 * x))) / x**3
        sig_k += np.log(1 + 2 * x) / (2 * x)
        sig_k -= (1 + 3 * x) / (1 + (2 * x))**2
        return C.sigmaT * 3 * sig_k / 4

# Distibution function: see Dermer 6.75
    def Fc(self, E_s=None, g=None, E=None, q=None, Gam=None):
        # have to specify args
        # g is lorents factor, E_s is scattered photon energy and E is incoming photon energy.
        if q is None:
            Gam = 4 * g * E
            q = E_s / (g * Gam * (1 - (E_s / g)))
            fc = 2 * q * np.log(q) + (1 + (2 * q)) * (1 - q) + \
                (((Gam * q)**2) * (1 - q)) / (2 * (1 + (Gam * q)))
            return fc
        else:

            fc = 2 * q * np.log(q) + (1 + (2 * q)) * (1 - q) + \
                (((Gam * q)**2) * (1 - q)) / (2 * (1 + (Gam * q)))
            return fc

    def PowerLaw(self, E, Q0, Emin, Emax, q):
        if type(E) == float:
            if E <= Emin or E >= Emax:
                return 1e-200
            else:
                return Q0 * np.power(E, -q)
        else:

            pwl = np.piecewise(E,
                               [E < Emin, (E >= Emin) & (E <= Emax), E > Emax],
                               [lambda x: 1e-200,
                                lambda x: Q0 * np.power(x, -q),
                                lambda x: 1e-200])
            return pwl

    #
    #  ###### #    # #  ####   ####  # #    # # ##### #   #
    #  #      ##  ## # #      #      # #    # #   #    # #
    #  #####  # ## # #  ####   ####  # #    # #   #     #
    #  #      #    # #      #      # # #    # #   #     #
    #  #      #    # # #    # #    # #  #  #  #   #     #
    #  ###### #    # #  ####   ####  #   ##   #   #     #

    # mono energetic isotropic compton emissivity. see Dermer 6.77
    def j_ic(E_s, E, g):
        IC = iCompton()
        jic = np.zeros_like(E_s)

        for i in range(len(E_s) - 1):
            x = E_s[i] / 511000
            y = E
            if((4 * y) / (1 + (4 * y)) < x or y < x):
                Min = (x / 2) + (1 / 2) * np.sqrt(((x**2) * y + x) / y)
                # make general later for a function parameter.
                Max = 1e7

            elif(y / (1 + y) < x and x < y or y / (1 + y) < x and x <= 4 * y / (1 + (4 * y))):
                Max = (-x * y) / (x - y)
                #Min = 1
                # still need to make general for cut offs
                Min = 1e2
            elif(x > (3 / 4)):
                Min = (x / 2) + (1 / (2 * np.sqrt(3))) * np.sqrt(((3 / 4) * (x**2) + x))
                #Min = 1
                # still need to make general for cut offs
                Max = 1e7
            elif((4 * y) / (1 + (4 * y)) < x and x < y):
                Min = (x / 2) + (1 / 2) * np.sqrt(((x**2) * y + x) / y)
                Max = (-x * y) / (x - y)
            elif(x > (3 / 7) and x < (3 / 4)):
                Max = (-3 * x) / (-3 + (4 * x))
                #Min = 1
                # still need to make general for cut offs
                Min = 1e2
            else:
                continue

            def f(l):
                k = (IC.Fc(x, l, y) * IC.PowerLaw(l, 1, Min, Max, 2.2)) / (l**2)
                return k

            d = integrate.quad(f, Min, Max)
            jic[i] = (3 / 4) * C.sigmaT * ((x / y)**2) * d[0]
        # for i in range(len(g) - 1):
        # Max = E_s[i] / (1 - (E_s[i] / g[i]))
        # Min = E_s[i] / ((4 * ((g[i])**2)) * (1 - (E_s[i] / g[i])))
        # y = np.linspace(Min, Max, len(g))
        # for j in range(len(g) - 1):
        # def z(e):
        # d = IC.Fc(E_s[i], g[i], e) / (e**2)
        # return d
        # x = integrate.romberg(z, y[j], y[j + 1])

#
        # def z(e):
        # d = IC.PowerLaw(e, 1, 1e12, 1e-12, 2.2) / (e**2)
        # return d
        # b = x * integrate.romberg(z, g[i], g[i + 1])
        #    jic[i] = (3 / 4) * C.sigmaT * ((E_s[i])**2) *b

        return jic

import numpy as np
import SAPytho.constants as C
import scipy.integrate as integrate
import SAPytho.spectra as sp
from joblib import Parallel, delayed


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

    # mono energetic isotropic compton emissivity. see Dermer 6.74
    def j_ic(E_s, E, g, p=2.2, Qe=1, Qp=1, gmin=1e2, gmax=1e7, rtol=1.48e-25, tol=1.48e-25, divmax=22):
        IC = iCompton()

        def Fc(e_s=None, g=None, e=None, q=None, Gam=None):
            # have to specify args
            # g is lorents factor, E_s is scattered photon energy and E is incoming photon energy.
            if q is None:
                Gam = 4 * g * e
                q = (e_s / g) / (Gam * (1 - (e_s / g)))
                fc = 2 * q * np.log(q) + (1 + (2 * q)) * (1 - q) + \
                    (((Gam * q)**2) * (1 - q)) / (2 * (1 + (Gam * q)))
                return fc
            else:

                fc = 2 * q * np.log(q) + (1 + (2 * q)) * (1 - q) + \
                    (((Gam * q)**2) * (1 - q)) / (2 * (1 + (Gam * q)))
                return fc

        def longcode(h):

            x = E_s[h] / 510998.927602161
            y = E
            if((4 * y) / (1 + (4 * y)) < x or y < x):
                Min = (x / 2) + (1 / 2) * np.sqrt(((x**2) * y + x) / y)

                Max = gmax

            elif(y / (1 + y) < x and x < y or y / (1 + y) < x and x <= 4 * y / (1 + (4 * y))):
                Max = (-x * y) / (x - y)

                Min = gmin
            elif(x > (3 / 4)):
                Min = (x / 2) + (1 / (2 * np.sqrt(3))) * np.sqrt(((3 / 4) * (x**2) + x))
                if(Min < gmin):
                    Min = gmin
                Max = gmax
            elif((4 * y) / (1 + (4 * y)) < x and x < y):
                Min = (x / 2) + (1 / 2) * np.sqrt(((x**2) * y + x) / y)
                if(Min < gmin):
                    Min = gmin
                Max = (-x * y) / (x - y)
                if(Max > gmax):
                    Max = gmax
            elif(x > (3 / 7) and x < (3 / 4)):
                Max = (-3 * x) / (-3 + (4 * x))
                if(Max > gmax):
                    Max = gmax
                Min = gmin
            else:
                Max = gmax
                Min = gmin

            def f(l):
                # seems to run better if i call fc and power from inside this function
                k = (Fc(x, l, y) * np.power(l, -p)) / (l**2)
                return k

            d = integrate.romberg(f, Min, Max, rtol=rtol, tol=tol, divmax=divmax)
            jic = (3 / 4) * C.sigmaT * ((x / y)**2) * d
            print(jic)
            return jic
        results = Parallel(n_jobs=-2)(delayed(longcode)(h) for h in range(len(E_s)))

        return results

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

    def PowerLaw(E, Q0, Emin, Emax, q):
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

    def j_ic(nu, Q0, E, qe, qph, e_powerlaw=True, Emin=0., Emax=10**30):

        if(e_powerlaw):
            f = np.power
        else:
            f = C.cLight

        def cross_section(nu):

            # eq. (7.5) from Rybicki & Lightman (1979)
            x = (C.hPlanck * nu) / (C.me * C.cLight**2)
            sig_k = (1 + x) * ((2 * x * (1 + x) / (1 + (2 * x))) - np.log(1 + (2 * x))) / x**3
            sig_k += np.log(1 + 2 * x) / (2 * x)
            sig_k -= (1 + 3 * x) / (1 + (2 * x))**2
            return C.sigmaT * 3 * sig_k / 4

        def PowerLaw(E, Q0, Emin, Emax, q):

            return Q0 * np.power(E, -q)

        def f1(E1):
            f = (E1**2) * (PowerLaw(E1, Q0, Emin, Emax, qe)) * cross_section(sp.eV2Hz(E1))
            return f

        def f2(nuu, E1, E2):
            f = integrate.romberg(f1, E1, E2)
            # f = (sp.Hz2eV(nuu)**2) * cross_section(sp.eV2Hz(nuu))
            # f = (sp.Hz2eV(nuu)**2) * (PowerLaw(nuu, Q0, Emin, Emax, qph)) * cross_section(nuu)
            return f
        jnu = np.zeros_like(nu)

        for k in range(len(nu) - 1):

            for i in range(len(nu) - 1):

                # I1 = integrate.romberg(f1, E[i], E[i + 1])
                I4 = integrate.romberg(f2, sp.Hz2eV(
                    nu[k]), sp.Hz2eV(nu[k + 1]), args=(E[i], E[i+1]))
                jnu[k] = I4 * C.cLight * C.hPlanck * nu[k]
            # I2 = integrate.quad(lambda x: I1, 0, np.pi)
            # I3 = integrate.quad(lambda x: I2[0], 0, 2 * np.pi)
            # I4 = integrate.romberg(f2, sp.Hz2eV(nu[k]), sp.Hz2eV(nu[k + 1]))
            # I5 = integrate.quad(lambda x: I4, 0, np.pi)
            # I6 = integrate.quad(lambda x: I5[0], 0, 2 * np.pi)
            # jnu[k] = I1 * C.cLight * C.hPlanck * nu[k]
        return jnu

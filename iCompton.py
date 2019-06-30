import numpy as np
import SAPytho.constants as C


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
        sig_k = (1 + x) * ((2 * x * (1 + x) / (1 + 2 * x)) - np.log(1 + 2 * x)) / x**3
        sig_k += np.log(1 + 2 * x) / (2 * x)
        sig_k -= (1 + 3 * x) / (1 + 2 * x)**2
        return C.sigmaT * 3 * sig_k / 4

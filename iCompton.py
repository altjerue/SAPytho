import numpy as np
import SAPytho.constants as C
import math as math


class iCompton:

    def __init__(self, **kwargs):
        self.Zq = 1.0
        self.mq = C.me
        self.re = ((C.eCharge)**2) / (self.mq * C.cLight**2)
        self.__dict__.update(kwargs)

# total cross section valid in all regimes or 'klein nishina formula'
    def cross_section(self, nu):

        # eq from Rybicki
        sig_k = np.zeros_like(nu)
        for i in range(len(nu)):
            x = (C.hPlanck * nu[i]) / (self.mq * (C.cLight**2))
            sig_k[i] = C.sigmaT * (3 / 4) * ((((1 + x) / (x**3)) * (((2 * x * (1 + x)) / (1 + 2 * x)) - math.log(
                1 + 2 * x))) + ((1 / (2 * x)) * math.log(1 + 2 * x)) - ((1 + 3 * x) / ((1 + 2 * x)**2)))
        return sig_k

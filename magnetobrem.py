import scipy.special as scisp
import numpy as np
import scipy.integrate as integrate
from SAPyto import misc
import SAPyto.SRtoolkit as SR
import SAPyto.pwlFuncs as pwlf
import SAPyto.constants as C


class mbs:

    def __init__(self, **kwargs):
        self.Zq = 1.0
        self.mq = C.me
        self.__dict__.update(kwargs)

    def nu_g(self, B):
        '''Cyclotron frequency
              nu_g = Z e B / (2 pi m_q c)
        Dafault values: Z = 1.0, m_q = m_e
        '''
        return C.eCharge * self.Zq * B / (C.twopi * self.mq * C.cLight)

    def nu_B(self, B, g):
        '''Gyrofrequency'''
        return C.eCharge * self.Zq * B / (C.twopi * g * self.mq * C.cLight)

    def nu_c(self, B, g, alpha=C.halfpi):
        '''Synchrotron critical frequency'''
        return 1.5 * self.nu_g(B) * np.sin(alpha) * g**2

    def nu_c_iso(self, B, g):
        '''Synchrotron critical frequency'''
        return 1.5 * self.nu_g(B) * g**2

    def chi(self, nu, B):
        '''Harmonic frequency'''
        return nu / self.nu_g(B)

    def Psyn_iso(self, gamma, B):
        '''Total synchrotron radiated power for an isotropic distribution of
        velocities. Formula given in Rybicki & Lightman (1985), eq. (6.7b):

            P = (4 / 3) sigma_T c beta^2 gamma^2 (B^2 / 8 pi)
        '''
        return 4.0 * C.sigmaT * C.cLight * SR.speed2(gamma) * gamma**2 * B**2 / (24.0 * np.pi)

    def Fsync(self, Xc, asym_low=False, asym_high=False):
        '''Synchrotron function'''
        if asym_low:
            return 4.0 * np.pi * np.power(0.5 * Xc, 1.0 / 3.0) / (np.sqrt(3.0) * scisp.gamma(1.0 / 3.0))
        elif asym_high:
            return np.sqrt(np.pi * Xc * 0.5) * np.exp(-Xc)
        else:
            return np.asarray(
                [x * integrate.quad(lambda y: scisp.kv(5.0 / 3.0, y), x, np.inf)[0] for x in Xc])

    def Rsync(self, Xc, asym_low=False, asym_high=False):
        ''' R(x) in Crusius & Schlickeiser (1986)
        '''
        if asym_low:
            return 1.8084180211028021 * np.power(Xc, 1.0 / 3.0)
        if asym_high:
            return 0.5 * np.pi * (1.0 - 11.0 / (18.0 * Xc)) * np.exp(-Xc)
        return 0.5 * np.pi * Xc * (misc.whittW(0.0, 4.0 / 3.0, Xc)
                                   * misc.whittW(0.0, 1.0 / 3.0, Xc)
                                   - misc.whittW(0.5, 5.0 / 6.0, Xc)
                                   * misc.whittW(-0.5, 5.0 / 6.0, Xc))

    def SL07(self, Xc):
        '''Approximation in Schlickeiser & Lerche (2007)'''
        return 1.5 * np.power(Xc, -2.0/3.0) / (0.869 + np.power(Xc, 1.0/3.0) * np.exp(Xc))

    def FDB08fit(self, Xc):
        '''Fit by Finke, Dermer & Boettcher (2008)'''

        def A(x):
            return np.power(10.0,
                            - 0.3577524
                            - 0.8369539 * np.log10(x)
                            - 1.1449608 * np.power(np.log10(x), 2)
                            - 0.6813728 * np.power(np.log10(x), 3)
                            - 0.2275474 * np.power(np.log10(x), 4)
                            - 0.0319673 * np.power(np.log10(x), 5))

        def B(x):
            return np.power(10.0,
                            - 0.3584249
                            - 0.7965204 * np.log10(x)
                            - 1.6113032 * np.power(np.log10(x), 2)
                            + 0.2605521 * np.power(np.log10(x), 3)
                            - 1.6979017 * np.power(np.log10(x), 4)
                            + 0.0329550 * np.power(np.log10(x), 5))

        return np.piecewise(Xc,
                            [Xc < 0.01,
                             (Xc >= 0.01) & (Xc < 1.0),
                             (Xc >= 1.0) & (Xc < 10.0),
                             Xc >= 10.0],
                            [lambda x: self.Rsync(x, asym_low=True),
                             lambda x: A(x),
                             lambda x: B(x),
                             lambda x: self.Rsync(x, asym_high=True)])

    #
    #  ######  #     #    #
    #  #     # ##   ##   # #
    #  #     # # # # #  #   #
    #  ######  #  #  # #     #
    #  #   #   #     # #######
    #  #    #  #     # #     #
    #  #     # #     # #     #
    def RMAfit(self, Xc, g):
        '''Fit by Rueda-Becerril (2017)

        RMAfit(x) = Rsync(x) = 0.5 pi x CS(x)
        '''
        # xg3 = Xc * g**3

        def A(x):
            return np.exp(- 0.7871626401625178
                          - 0.7050933708504841 * np.log(x)
                          - 0.3553186929561062 * np.power(np.log(x), 2)
                          - 0.0650331246186839 * np.power(np.log(x), 3)
                          - 0.0060901233982264 * np.power(np.log(x), 4)
                          - 0.0002276461663805 * np.power(np.log(x), 5))

        def B(x):
            return np.exp(- 0.823645515457065
                          - 0.831668613094906 * np.log(x)
                          - 0.525630345887699 * np.power(np.log(x), 2)
                          - 0.220393146971054 * np.power(np.log(x), 3)
                          + 0.016691795295125 * np.power(np.log(x), 4)
                          - 0.028650695862678 * np.power(np.log(x), 5))

        def theFit(x):
            c1 = 3.218090050062573e-4
            c2 = 0.650532122717873
            c3 = 15.57990468980456
            return np.piecewise(x,
                                [x < c1,
                                 (x >= c1) & (x <= c2),
                                 (x > c2) & (x < c3),
                                 x >= c3],
                                [lambda x: self.Rsync(x, asym_low=True),
                                 lambda x: A(x),
                                 lambda x: B(x),
                                 lambda x: self.Rsync(x, asym_high=True)])
        return np.piecewise(Xc,
                            [Xc * g**3 < 0.5, Xc * g**3 >= 0.5],
                            [0.0, lambda x: theFit(x)])

    def RMA(self, Xc, g):
        return np.piecewise(Xc,
                            [Xc * g**3 <= 0.5, Xc * g**3 > 0.5],
                            [0.0, lambda x: x * self.SL07(x)])


#
#  ###### #    # #  ####   ####  # #    # # ##### #   #
#  #      ##  ## # #      #      # #    # #   #    # #
#  #####  # ## # #  ####   ####  # #    # #   #     #
#  #      #    # #      #      # # #    # #   #     #
#  #      #    # # #    # #    # #  #  #  #   #     #
#  ###### #    # #  ####   ####  #   ##   #   #     #
def j_mb(nu, g, N, B, Rsync=False):
    '''Description:
    This function reproduces the MBS emissivity from a power-law distribution.
    '''
    MBS = mbs()

    def f(lg, c, q):
        g = np.exp(lg)
        Xc = 2.0 * c / (3.0 * g**2)
        if Rsync is True:
            return g**(1.0 - q) * MBS.Rsync(Xc)
        else:
            return g**(1.0 - q) * MBS.RMAfit(Xc, g)

    nuB = C.nuConst * B
    chi = nu / nuB
    jnu = np.zeros_like(nu)

    for j in range(nu.size):
        for k in range(g.size - 1):
            if (N[k] > 1e-100 and N[k + 1] > 1e-100):
                q = -np.log(N[k + 1] / N[k]) / np.log(g[k + 1] / g[k])
                if (q > 8.):
                    q = 8.
                if (q < -8.):
                    q = -8.

                I2 = integrate.romberg(f, np.log(g[k]), np.log(g[k + 1]), args=(chi[j], q), divmax=15)
                jnu[j] = jnu[j] + N[k] * I2 * g[k]**q

            if (jnu[j] < 1e-200):
                jnu[j] = 0.

    return C.jmbConst * nuB * jnu


#
#    ##   #####   ####   ####  #####  #####  ##### #  ####  #    #
#   #  #  #    # #      #    # #    # #    #   #   # #    # ##   #
#  #    # #####   ####  #    # #    # #    #   #   # #    # # #  #
#  ###### #    #      # #    # #####  #####    #   # #    # #  # #
#  #    # #    # #    # #    # #   #  #        #   # #    # #   ##
#  #    # #####   ####   ####  #    # #        #   #  ####  #    #
def a_mb(nu, g, N, B, Rsync=False):
    '''Description:
    This function reproduces the MBS absorption from a power-law distribution.
    '''
    MBS = mbs()

    def f(lg, c, q):
        g = np.exp(lg)
        Xc = 2 * c / (3 * g**2)
        if Rsync is True:
            return g**(-q) * MBS.Rsync(Xc) * (q + 1 + g**2 / (g**2 - 1))
        else:
            return g**(-q) * MBS.RMAfit(Xc, g) * (q + 1 + g**2 / (g**2 - 1))

    nuB = C.nuConst * B
    chi = nu / nuB
    anu = np.zeros_like(nu)

    for j in range(nu.size):
        for k in range(g.size - 1):
            if (N[k] > 1e-100 and N[k + 1] > 1e-100):
                q = -np.log(N[k + 1] / N[k]) / np.log(g[k + 1] / g[k])
                if (q > 8.):
                    q = 8.
                if (q < -8.):
                    q = -8.

                A2 = integrate.romberg(f, np.log(g[k]), np.log(g[k + 1]), args=(chi[j], q), divmax=15)
                anu[j] = anu[j] + N[k] * A2 * g[k]**q
            if (anu[j] < 1e-200):
                anu[j] = 0.

    return C.ambConst * nuB * anu / nu**2

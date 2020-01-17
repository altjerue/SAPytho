import numpy as np
# import numpy.ma as ma
from scipy import integrate  # , interpolate
# from . import magnetobrem as mbs
from . import constants as C
from . import misc as misc
# from .spectra import spectrum as spec
from . import SRtoolkit as SR


def Band_function(E_eV, Ep_eV, alpha, beta, A=1e0):
    '''Band function as in Zhang et al., 2016, ApJ, 816, 72
    '''
    E0 = Ep_eV / (2.0 + alpha)
    f = []
    for e in E_eV:
        if (e <= (alpha - beta) * E0):
            f.append(np.power(e / 100e3, alpha) * np.exp(-e / E0))
        else:
            f.append(np.power(e / 100e3, beta) * np.exp(beta - alpha) * np.power((alpha - beta) * E0 / 100e3, alpha - beta))
    Flux = np.asarray(f)
    return A * Flux


class blobZS12(object):
    '''Emitting blob based on the model in Zacharias & Schlickeiser, 2013, ApJ, 777, 109.
    '''

    def __init__(self, Gbulk, theta, z, dL, D, R, t_obs, t_em, nus):
        self.Gbulk = Gbulk
        self.z = z
        self.dL = dL
        self.Dopp = D
        self.Radius = R
        self.mu = np.cos(np.deg2rad(theta))
        try:
            self.numt_obs = len(t_obs)
            self.t_obs = t_obs
        except TypeError:
            self.numt_obs = 1
            self.t_obs = [t_obs]
        try:
            self.numt_em = len(t_em)
            self.t_em = t_em
        except TypeError:
            self.numt_em = 1
            self.t_em = [t_em]
        try:
            self.numf = len(nus)
            self.nus = nus
        except TypeError:
            self.numf = 1
            self.nus = [nus]
        print("-->  Blob set up")

    def integrando_v(self, ll, l0, tt, It):
        res = np.zeros_like(ll)
        for i in range(res.size):
            te = self.Dopp * ((tt / (1.0 + self.z)) + (self.Gbulk * ll[i] * l0 * (self.mu - SR.speed(self.Gbulk))))
            # te = tt - ll[i] * l0
            if te < 0.0:
                res[i] = 0.0
            else:
                res[i] = self.pwl_interp(te, self.t_em, It) * (ll[i] - ll[i]**2)
        return res

    def integrando_s(self, ll, l0, tt, It):
        te = self.Dopp * ((tt / (1.0 + self.z)) + (self.Gbulk * ll * l0 * (self.mu - SR.speed(self.Gbulk))))
        # te = tt - ll * l0
        if te < 0.0:
            res = 0.0
        else:
            res = self.pwl_interp(te, self.t_em, It) * (ll - ll**2)
        return res

    def Inu_tot(self, Inut, wSimps=False):
        print("-->  Computing the received Intensity\n")
        print("---------------------------")
        print("| Iteration |  Frequency  |")
        print("---------------------------")
        lam0 = 2.0 * self.Radius * self.mu / C.cLight
        Itot = np.zeros((self.numt_obs, self.numf))
        lam = np.linspace(0.0, 1.0, num=100)

        for j in range(self.numf):
            I_lc = Inut[:, j]
            for i in range(self.numt_obs):
                if wSimps:  # ------->>>   SIMPSON
                    Itot[i, j] = 6.0 * integrate.simps(self.integrando_v(lam, lam0, self.t_obs[i], I_lc), x=lam)
                else:  # ------->>>   TRAPEZOIDAL
                    Itot[i, j] = 6.0 * integrate.trapz(self.integrando_v(lam, lam0, self.t_obs[i], I_lc), x=lam)
            if np.mod(j + 1, 32) == 0.0:
                print("| {0:>9d} | {1:>11.3E} |".format(j, self.nus[j]))
        print("---------------------------")
        return Itot

    def pwl_interp(self, t_in, times, Ilc):
        '''This function returns a power-law interpolation
        '''
        t_pos, t = misc.find_nearest(times, t_in)
        if t > t_in:
            t_pos += 1
        if t_pos >= len(times) - 1:
            t_pos = len(times) - 3
        if (Ilc[t_pos] > 1e-100) & (Ilc[t_pos + 1] > 1e-100):
            s = np.log(Ilc[t_pos + 1] / Ilc[t_pos]) / np.log(times[t_pos + 1] / times[t_pos])
            res = Ilc[t_pos] * (t_in / times[t_pos])**s
        else:
            res = 0.0
        return res


class SPN98(object):
    '''Following This is the setup for the model in Sari, Piran & Narayan, 1998, ApJ, 497, L17.
    '''

    def params(self):
        self.eps_e = 0.6
        self.eps_B = 0.5
        self.G0 = 1.0
        self.n1 = 1.0
        self.E0 = 1.0
        self.dL = 1.0
        self.z = 1.0
        self.pind = 2.5
        self.adiab = True

    def __init__(self, **kwargs):
        self.params()
        self.__dict__.update(kwargs)
        self.G2 = self.G0 / 100
        self.E52 = self.E0 / 1e52
        self.D28 = self.dL / 1e28

    def BlastWave(self, tobs, adiab=True):
        t = tobs / (1 + self.z)
        M = self.E0 / (self.G0 * C.cLight**2)
        L = (17 * M / (16 * np.pi * C.mp * self.n1))**(1 / 3)
        if adiab:
            Rshk = (17 * self.E0 * t / (4 * np.pi * C.mp * self.n1 * C.cLight))**0.25
            Gshk = (17 * self.E0 / (1024 * np.pi * C.mp * self.n1 * C.cLight**5 * t**3))**(0.125)
        else:
            Rshk = (4 * C.cLight * t / L)**(1 / 7) * L
            Gshk = (4 * C.cLight * t / L)**(-3 / 7)
        for i in range(Gshk.size):
            Gshk[i] = np.maximum(np.minimum(Gshk[i], self.G0), 1.001)
        return Rshk, Gshk

    #
    #  ###### #####  ######  ####   ####
    #  #      #    # #      #    # #
    #  #####  #    # #####  #    #  ####
    #  #      #####  #      #  # #      #
    #  #      #   #  #      #   #  #    #
    #  #      #    # ######  ### #  ####
    #
    def nuc(self, td):
        if self.adiab:
            # NOTE: Following Eq. (11)
            return 2.7e12 * self.eps_B**(-1.5) / (np.sqrt(self.E52 * td * (1 + self.z)) * self.n1)
        else:
            # NOTE: Following Eq. (12)
            return 1.3e13 * (self.G2 / self.E52)**(4.0 / 7.0) * self.eps_B**(-1.5) * td**(-2.0 / 7.0) * self.n1**(-13.0 / 14.0)

    def num(self, td):
        if self.adiab:
            # NOTE: Following Eq. (11)
            return 5.7e14 * self.eps_e**2 * np.sqrt((1 + self.z) * self.E52 * self.eps_B) * td**(-1.5)
        else:
            # NOTE: Following Eq. (12)
            return 1.2e14 * np.sqrt(self.eps_B) * self.eps_e**2 * (self.E52 / self.G2)**(4.0 / 7.0) * self.n1**(-1.0 / 14.0) * td**(-12.0 / 7.0)

    def nua_fast(self, td):
        return 0.15e9 * (self.eps_B / 0.01)**1.2 * self.E52**0.7 * self.n1**1.1 / np.sqrt(td * (1 + self.z))

    def nua_slow(self):
        return 3.6e9 * (0.5 / self.eps_e) * (self.E52 * self.eps_B / 0.01)**0.2 * self.n1**0.6 / (1 + self.z)

    def nu0(self):
        if self.adiab:
            nu0 = 1.8e11 * self.eps_B**(-2.5) * self.n1**(-1.5) / (self.eps_e * self.E52)
        else:
            nu0 = 8.5e12 * self.eps_B**(-1.9) * self.eps_e**(-0.4) * self.n1**(-1.1) * (self.G2 / self.E52)**0.8
        return nu0

    #
    #  ##### # #    # ######  ####
    #    #   # ##  ## #      #
    #    #   # # ## # #####   ####
    #    #   # #    # #           #
    #    #   # #    # #      #    #
    #    #   # #    # ######  ####
    #
    def tc(self, nu15):
        if self.adiab:
            return 7.3e-6 * np.ones(nu15.size) / (self.eps_B**3 * self.E52 * self.n1**2 * nu15**2)
        else:
            return 2.7e-7 * self.eps_B**(-0.25 * 21.0) * (self.G2 / self.E52)**2 * self.n1**(-0.25 * 13.0) * nu15**(-0.5 * 7.0)

    def tm(self, nu15):
        if self.adiab:
            return 0.69 * (self.eps_B * self.E52)**(1.0 / 3.0) * self.eps_e**(4.0 / 3.0)**(1.0 / 3.0) * nu15**(-2.0 / 3.0) * np.ones(nu15.size)
        else:
            return 0.29 * self.eps_B**(7.0 / 24.0) * self.eps_e**(7.0 / 6.0) * (self.E52 / self.G2)**(1.0 / 3.0) * self.n1**(-1.0 / 24.0) * nu15**(-7.0 / 12.0)

    def t0(self):
        if self.adiab:
            t0 = 210.0 * (self.eps_B * self.eps_e)**2 * self.E52 * self.n1
        else:
            t0 = 4.6 * (self.eps_B * self.eps_e)**1.4 * (self.E52 / self.G2)**0.8 * self.n1**0.6
        return t0

    #
    #  ###### #      #    # #    # ######  ####
    #  #      #      #    #  #  #  #      #
    #  #####  #      #    #   ##   #####   ####
    #  #      #      #    #   ##   #           #
    #  #      #      #    #  #  #  #      #    #
    #  #      ######  ####  #    # ######  ####
    #
    def Fmax(self, td, Ne, B, pwise=True):
        '''Flux in micro janskys (uJy)'''
        if pwise:
            if self.adiab:
                # NOTE: Following Eq. (11)
                return 1.1e5 * np.sqrt(self.eps_B * self.n1) * self.E52 / self.D28**2 * np.ones(td.size)
            else:
                # NOTE: Following Eq. (12)
                return 4.5e3 * np.sqrt(self.eps_B) * (self.E52 / self.G2)**(8. / 7.) * self.n1**(5. / 14.) / (td**(3. / 7.) * self.D28**2)
        else:
            return Ne * C.me * C.cLight**2 * C.sigmaT * B / (12. * np.pi * C.eCharge * self.dL**2)

    def fast_cooling(self, nu, nua, nuc, num, Fnu_max):
        return np.piecewise(nu,
                            [num > nu,  # nu < nua,
                             # (num > nu) & (nu >= nua),
                             (num >= nu) & (nu >= nuc),
                             nu > num],
                            [  # (nua / nuc)**(1/3) * (nu / nua)**2 * Fnu_max,
                             (nu / nuc)**(1.0 / 3.0) * Fnu_max,
                             Fnu_max / np.sqrt(nu / nuc),
                             (nu / num)**(-0.5 * self.pind) * Fnu_max / np.sqrt(num / nuc)])

    def slow_cooling(self, nu, nua, nuc, num, Fnu_max):
        return np.piecewise(nu,
                            [num > nu,  # nu < nua,
                             # (num > nu) & (nu >= nua),
                             (nuc >= nu) & (nu >= num),
                             nu > nuc],
                            [  # (nua / nuc)**(1/3) * (nu / nua)**2 * Fnu_max,
                             (nu / num)**(1/3) * Fnu_max,
                             (nu / num)**(-0.5 * (self.pind - 1.0)) * Fnu_max,
                             (nuc / num)**(-0.5 * (self.pind - 1.0)) * (nu / nuc)**(-0.5 * self.pind) * Fnu_max])

    # def fluxSPN98(self, nu, t):
    def fluxSPN98(self, nu, t, G, R, Ne):
        nu15 = nu * (1 + self.z) / 1e15
        td = t / (8.64e4 * (1 + self.z))
        flux = np.ndarray((t.size, nu.size))
        nua = np.ones_like(t)
        nuc = self.nuc(td)
        num = self.num(td)
        nu0 = self.nu0()
        tc = self.tc(nu15)
        tm = self.tm(nu15)
        t0 = self.t0()
        B = np.sqrt(32. * np.pi * C.mp * self.eps_B * self.n1) * G * C.cLight
        Fnu_max = self.Fmax(td, Ne, B, pwise=False) * (1 + self.z)
        # Fnu_max = Ne * Pmax * (1 + self.z) / (4 * np.pi * self.dL**2)
        for i in range(t.size):
            for j in range(nu.size):
                if td[i] < t0:
                    nua[i] = self.nua_fast(td[i])
                    flux[i, j] = self.fast_cooling(nu[j], nua[i], nuc[i], num[i], Fnu_max[i])
                else:
                    nua[i] = self.nua_slow()
                    flux[i, j] = self.slow_cooling(nu[j], nua[i], nuc[i], num[i], Fnu_max[i])
        t0 *= (1 + self.z)
        nu0 /= (1 + self.z)
        return nuc, num, nua, nu0, tc, tm, t0, flux

import numpy as np
import numpy.ma as ma
import scipy.integrate as sci_integ
from . import misc
from . import pwlFuncs as pwlf
from . import SRtoolkit as srtool
from . import constants as C
import scipy.optimize as op


def conv2Jy(flux):
    '''Convert flux density (in egs cm^{-2} s^{-1} Hz^{-1}) to janskys.
    '''
    return flux * 1e23


def Hz2eV(nu):
    '''Convert frequency in hertz to energy in electronvolt
    '''
    return nu * 4.135667662e-15


def eV2Hz(nu):
    '''Convert energy in electronvolt to frequency in hertz
    '''
    return nu * 2.4179937422321953e14


def Hz2m(nu):
    '''Convert frequency in hertz to wavelength in meters
    '''
    return C.cLight * 1e-2 / nu


def m2Hz(wavelength):
    '''Convert frequency in hertz to wavelength in meters
    '''
    return C.cLight * 1e-2 / wavelength


def sec2dy(time):
    '''Convert time in seconds to days
    '''
    return time / 8.64e4


def dy2sec(time):
    '''Convert time in days to seconds
    '''
    return time * 86400.0


def sec2hr(time):
    '''Convert time in seconds to hours
    '''
    return time / 3600.0


def hr2sec(time):
    '''Convert time in hours to seconds
    '''
    return time * 3600.0


def pc2cm(distance):
    '''Convert distance from parsecs to centimeters
    '''
    return distance * 3.08567758149137e18


def cm2pc(distance):
    '''Convert distance from centimeters to parsecs
    '''
    return distance / 3.08567758149137e18


def specEnergyFlux(Inu, dL, z, Doppler, radius):
    '''Calculates the spectral energy flux of a sphere.
    '''
    return 4 * np.pi * radius**2 * Inu * Doppler**3 * (1 + z) / (3 * dL**2)


def EnergyFlux(nuInu, dL, Doppler, radius):
    '''Calculates the energy flux of a sphere.
    '''
    return 4 * np.pi * radius**2 * nuInu * Doppler**4 / (3 * dL**2)


#
#  ###   ##                                    ##
#   #   #   #####          ####  #####   ####    #
#   #  #      #           #    # #    # #         #
#   #  #      #           #    # #####   ####     #
#   #  #      #           #    # #    #      #    #
#   #   #     #           #    # #    # #    #   #
#  ###   ##   #            ####  #####   ####  ##
#                 #######
def Itobs(t, nu, jnut, sen_lum, R, muc, Gbulk, muo, z, D):
    pwl = pwlf.PwlInteg()
    Itobs = np.zeros_like(jnut)
    i_edge = np.argmin(np.abs(2.0 * R * muc - sen_lum))
    if (sen_lum[i_edge] > 2.0 * R * muc):
        i_edge = i_edge - 1

    for j in range(nu.size):
        for i in range(t.size):

            if (i <= i_edge):
                i_start = 0
            else:
                i_start = i - i_edge

            for ii in range(i_start, i):
                tob_min = srtool.t_com(t[i], z, Gbulk, muo, x=t[ii - 1] * C.cLight * muc)
                tob_max = srtool.t_com(t[i], z, Gbulk, muo, x=t[ii] * C.cLight * muc)

                if ii == 0:
                    Itobs[i, j] = np.abs(tob_max - tob_min) * jnut[0, j]
                else:
                    if (jnut[ii, j] > 1e-100) & (jnut[ii - 1, j] > 1e-100):
                        sind = -np.log(jnut[ii, j] / jnut[ii - 1, j]) / np.log(tob_max / tob_min)
                        if (sind < -8.0):
                            sind = -8.0
                        if (sind > 8.0):
                            sind = 8.0
                        Itobs[i, j] = Itobs[i, j] + jnut[ii - 1, j] * tob_min * \
                            pwl.P(tob_max / tob_min, sind, 1e-6) / \
                            (Gbulk * muc * (muo - srtool.speed(Gbulk)) * D)
    return Itobs


#
#  #      #  ####  #    # #####  ####  #    # #####  #    # ######  ####
#  #      # #    # #    #   #   #    # #    # #    # #    # #      #
#  #      # #      ######   #   #      #    # #    # #    # #####   ####
#  #      # #  ### #    #   #   #      #    # #####  #    # #           #
#  #      # #    # #    #   #   #    # #    # #   #   #  #  #      #    #
#  ###### #  ####  #    #   #    ####   ####  #    #   ##   ######  ####
class LightCurves:
    def __init__(self):
        pass

    def nearest(self, nu_in, nus, flux):
        '''This function returns the light curve of the frequency nearest to
        the frequency given: nu_in.
        '''
        nu_pos, nu = misc.find_nearest(nus, nu_in)
        print("Nearest frequency: {0} Hz".format(misc.fortran_double(nu, dble=False)))
        return flux[:, nu_pos]

    def pwl_interp(self, nu_in, t, nus, flux):
        '''This function returns a power-law interpolated light curve
        '''
        nu_pos, nu = misc.find_nearest(nus, nu_in)
        lc = np.zeros_like(t)
        flux /= nus
        if len(nus) > 1:
            if nu > nu_in:
                nu_pos += 1
            if nus[nu_pos] >= nus[-1]:
                nu_pos -= 1
            for i in range(t.size):
                if (flux[i, nu_pos] > 1e-100) & (flux[i, nu_pos + 1] > 1e-100):
                    s = -np.log(flux[i, nu_pos + 1] / flux[i, nu_pos]) / \
                        np.log(nus[nu_pos + 1] / nus[nu_pos])
                    if s > 8.0:
                        s = 8.0
                    if s < -8.0:
                        s = -8.0
                    lc[i] = flux[i, nu_pos] * (nu_in / nus[nu_pos])**s
        else:
            lc = self.nearest(nu, nus, flux)
        return lc

    def integ(self, nu_min, nu_max, t, freqs, flux):
        '''This function returns the integrated light curve in the given frequency band[nu_min, nu_max]
        '''
        licur = np.zeros_like(t)
        if nu_min < freqs[0]:
            print("nu_min =", nu_min, "\nminimum frequency in array =", freqs[0])
            return licur
        if nu_max > freqs[-1]:
            print("nu_max =", nu_max, "\nmaximum frequency in array=", freqs[-1])
            return licur

        if nu_max == nu_min:
            for i in range(t.size):
                licur[i] = np.exp(np.interp(np.log(nu_max), np.log(freqs),
                                            np.log(flux[i, :], where=(flux[i, :] != 0.0))))
        else:
            nu_mskd = ma.masked_outside(freqs, nu_min, nu_max)
            nus = nu_mskd.compressed()
            Fnu = flux[:, ~nu_mskd.mask] / nus
            for i in range(t.size):
                # NOTE: The integral is logarithmic, therfore the nus multiplying Fnu
                licur[i] = sci_integ.simps(nus * Fnu[i, :], x=np.log(nus))

        return licur

    def luminosity(self, t, nus, jnu, vol):
        Lnu = vol * jnu
        Lum = np.zeros_like(t)
        for i in range(t.size):
            Lum[i] = sci_integ.simps(nus * Lnu[i, :], x=np.log(nus))
        return Lum

        #
        #   ####  #####  ######  ####  ##### #####    ##
        #  #      #    # #      #    #   #   #    #  #  #
        #   ####  #    # #####  #        #   #    # #    #
        #       # #####  #      #        #   #####  ######
        #  #    # #      #      #    #   #   #   #  #    #
        #   ####  #      ######  ####    #   #    # #    #


class spectrum:
    def __init__(self):
        pass

    def nearest(self, t_in, times, flux):
        '''This function returns the spectrum at th nearest time to
        the given one: t_in.
        '''
        t_pos, t = misc.find_nearest(times, t_in)
        print("Nearest time: {0} s".format(misc.sci_notation(t)))
        return flux[t_pos, :]

    def pwl_interp(self, t_in, nu, times, flux):
        '''This function returns a power-law interpolated spectrum at the given time: t_in
        '''
        t_pos, t = misc.find_nearest(times, t_in)
        if t > t_in:
            t_pos += 1
        if times[t_pos] >= times[-1]:
            t_pos -= 1
        spec = np.zeros_like(nu)
        for j in range(nu.size):
            if (flux[t_pos, j] > 1e-100) & (flux[t_pos + 1, j] > 1e-100):
                s = -np.log(flux[t_pos + 1, j] / flux[t_pos, j]) / \
                    np.log(times[t_pos + 1] / times[t_pos])
                if s > 8.0:
                    s = 8.0
                if s < -8.0:
                    s = -8.0
                spec[j] = flux[t_pos, j] * (t_in / times[t_pos])**s
        return spec

    def integ(self, t_min, t_max, nu, times, flux, ret_tmasked=False):
        '''This function returns the integrated spectrum during the period[t_min, t_max]
        '''

        if t_max < times[0]:
            print("Input t_max: ", t_max, "\nMinimum time in array:", times[0])
            if ret_tmasked:
                return np.zeros_like(nu), times
            else:
                return np.zeros_like(nu)
        if t_min > times[-1]:
            print("Input t_min: ", t_min, "\nMaximum time in array:", times[-1])
            if ret_tmasked:
                return np.zeros_like(nu), times
            else:
                return np.zeros_like(nu)

        if t_min < times[0]:
            t_min = times[0]
            print("Input t_min: ", t_min, "\nMinimum time in array:", times[0])
        if t_max > times[-1]:
            t_max = times[-1]
            print("Input t_max: ", t_max, "\nMaximum time in array:", times[-1])

        if t_max == t_min:
            return self.pwl_interp(t_min, nu, times, flux)

        spec = np.zeros_like(nu)

        if (t_min == times[0]) & (t_max == times[-1]):
            tt = times
            Fnu = flux
        else:
            t_mskd = ma.masked_outside(times, t_min, t_max)
            tt = t_mskd.compressed()
            Fnu = flux[~t_mskd.mask, :]

        for j in range(nu.size):
            spec[j] = sci_integ.simps(tt * Fnu[:, j], x=np.log(tt))
            # spec[j] = sci_integ.simps(Fnu[:, j], x=tt)

        if ret_tmasked:
            return spec, tt
        else:
            return spec

    def averaged(self, t_min, t_max, nu, times, flux):
        '''This function returns the averaged spectrum over the period[t_min, t_max]
        '''
        spec, tt = self.integ(t_min, t_max, nu, times, flux, ret_tmasked=True)
        tott = np.sum(tt[1:] - tt[:-1]) + tt[0]
        return spec / tott


#   #####                                           ######
#  #     #  ####  #    # #####  #####  ####  #    # #     #  ####  #    #
#  #       #    # ##  ## #    #   #   #    # ##   # #     # #    # ##  ##
#  #       #    # # ## # #    #   #   #    # # #  # #     # #    # # ## #
#  #       #    # #    # #####    #   #    # #  # # #     # #    # #    #
#  #     # #    # #    # #        #   #    # #   ## #     # #    # #    #
#   #####   ####  #    # #        #    ####  #    # ######   ####  #    #
def ComptonDom(nus, Fsyn, Fic, t_min, t_max, times):
    spec = spectrum()
    # pwli = pwlf.PwlInteg()
    # Nf = nus.size

    # NOTE  synchrotron spectrum and peak
    synint = spec.integ(t_min, t_max, nus, times, Fsyn)
    syn_pos = synint.argmax()
    syn_peak = synint[syn_pos]
    nu_syn = nus[syn_pos]

    # syntot = 0.0
    # for j in range(Nf - 1):
    #     if (synint[j] > 1e-100) & (synint[j + 1] > 1e-100):
    #         s = -np.log(synint[j + 1] / synint[j]) / np.log(nus[j + 1] / nus[j])
    #         syntot += synint[j] * nus[j] * pwli.P(nus[j + 1] / nus[j], s)

    # NOTE  IC spectrum and peak
    ICint = spec.integ(t_min, t_max, nus, times, Fic)
    IC_pos = ICint.argmax()
    IC_peak = ICint[IC_pos]
    nu_IC = nus[IC_pos]

    # ICtot = 0.0
    # for j in range(Nf - 1):
    #     if (ICint[j] > 1e-100) & (ICint[j + 1] > 1e-100):
    #         s = -np.log(ICint[j + 1] / ICint[j]) / np.log(nus[j + 1] / nus[j])
    #         ICtot += ICint[j] * nus[j] * pwli.P(nus[j + 1] / nus[j], s)

    A_C = IC_peak / syn_peak
    return nu_syn, nu_IC, A_C

    #
    #  ######                                       #######
    #  #     # #    #  ####  #####  ####  #    #    #       #      #    # #    #
    #  #     # #    # #    #   #   #    # ##   #    #       #      #    #  #  #
    #  ######  ###### #    #   #   #    # # #  #    #####   #      #    #   ##
    #  #       #    # #    #   #   #    # #  # #    #       #      #    #   ##
    #  #       #    # #    #   #   #    # #   ##    #       #      #    #  #  #
    #  #       #    #  ####    #    ####  #    #    #       ######  ####  #    #
    def Fph(nu_min, nu_max, freqs, Fnu):
        '''Calculate the photon flux for the frequency band [nu_min, nu_max] from
        a given flux density.
        Input:
            nu_min, nu_max: scalars
            freqs: array
            Fnu: array
        Output:
            Photon flux: scalar
            photon flux spectral indices: array
        '''
        if nu_min < freqs[0] or nu_max > freqs[-1]:
            return print('Error: nu_min and nu_max outside frequencies array')

        nu_mskd = ma.masked_outside(freqs, nu_min, nu_max)
        nus = nu_mskd.compressed()
        flux = Fnu[~nu_mskd.mask] / nus
        num_nus = len(nus)

        integral = 0.0
        pwli = pwlf.PwlInteg()
        for i in range(num_nus - 1):
            if (flux[i] > 1e-100) & (flux[i + 1] > 1e-100):
                s = -np.log(flux[i + 1] / flux[i]) / np.log(nus[i + 1] / nus[i])
                integral += flux[i] * nus[i] * pwli.P(nus[i + 1] / nus[i], s)
        return nus[:-1], flux[:-1], integral / C.hPlanck

    def photon_index(freqs, fluxes):
        def f(x, a, b):
            return a * x + b
        popt, pcov = op.curve_fit(f, np.log10(freqs), np.log10(fluxes))
        return np.power(10.0, f(np.log10(freqs), *popt)), popt, pcov

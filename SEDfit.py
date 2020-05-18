import numpy as np
import sys
import scipy.optimize as sciop


def FindingMaximum(poly, o=1):
    dpdx = np.polyder(poly)
    extrema = np.roots(dpdx)
    index = np.isreal(extrema)
    Rextrema = extrema[index]
    RRextrema = []
    for extra in Rextrema:
        if extra.real > 5. and extra.real < 25.:
            RRextrema.append(extra)

    if len(RRextrema) == 0:
        sys.exit('No maxima/minima in the range.')

    ext_eval = np.polyval(poly, RRextrema).real
    imax = np.argmax(ext_eval)
    if o == 1:
        maxresult = np.power(10, RRextrema[imax].real)
    else:
        maxresult = [np.power(10, RRextrema[imax].real), np.power(10, np.amax(ext_eval))]
    return maxresult


def expo_cutoff(E, Ec, K, G):
    '''
    Power-law function with exponential cut-off (PL+EC model) at the energy enc
    $$
    F(E) = K E^{-\Gamma} \exp(-E / E_{c}) ph / (cm^{2} s keV)
    $$
    '''
    return K * E**(-G) * np.exp(-E / Ec)


def logparabol(E, E1, K, a, b):
    '''
    Log-parabolic model
    $$
    F(E) = K (E / E1)^{(-(a + b * \log(E / E1)))} ph / (cm^{2} s keV)
    $$
    '''
    return K * (E / E1)**(-(a + b * np.log(E / E1)))


def two_powlaws(E, EB, K, Gminf, Gpinf, f):
    '''
    Two power laws
    $$
    F(E) = K E^{-\Gamma_{-\infty}} {\left[1 + {\left(\frac{E}{E_{B}}\right)}^{f} \right]}^{(\Gamma_{-\infty} - \Gamma_{\infty}) / f}
    $$
    '''
    return K * E**(-Gminf) * (1. + (E / EB)**f)**((Gminf - Gpinf) / f)


def logSED(nu, b, nup, nupFnup):
    return np.log10(nupFnup) - b * (np.log10(nu / nup))**2

def polyfitSyn(nus, nuFnu, pol_order):
    # >>>  Doing the polynomial fitting
    p, res = np.polyfit(np.log10(nus), np.log10(nuFnu), pol_order, full=True)
    return p

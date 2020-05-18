import sys
import numpy as np
import scipy.optimize as sciop
import h5py
import scipy.integrate as scint


def Fitting(filename, model, syn_pol_order, ic_pol_order):
    # print '---> In Fitting'
    gf = h5py.File(filename + '.glob_spec.h5', 'r')
    nuturn_ds = gf['nuturn']
    nuturn = nuturn_ds[0][0]
    pf = h5py.File(filename + '.h5', 'r')
    nu_ds = pf['freqs']
    provnu = nu_ds[:]
    index_nuturn = np.nonzero(provnu == nuturn)[0][0] + 1
    gf.close()
    pf.close()

    specf = filename + '.tot_spec.dat'
    temp_nu, temp_SED = np.loadtxt(specf, usecols=(0, 2), unpack=True)
    nu = temp_nu[np.where(temp_nu < 10.**24)]
    nuF_nu = temp_SED[0:len(nu)]
    nusync = nu[:index_nuturn]
    nuic = nu[index_nuturn+1:]
    sed_sync = nuF_nu[:index_nuturn]
    sed_ic = nuF_nu[index_nuturn+1:]

    # >>>  Doing the polynomial fitting
    psyn, syn_res = np.polyfit(np.log10(nusync), np.log10(sed_sync), syn_pol_order, full=True)
    pic, ic_res = np.polyfit(np.log10(nuic), np.log10(sed_ic), ic_pol_order, full=True)

    return psyn, pic


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


def IntegFitting(syn_c, ic_c):
    p_syn = np.poly1d(syn_c)
    p_ic = np.poly1d(ic_c)
    p_diff = p_syn - p_ic

    def SynIntegrand(x): return np.log(10) * np.power(10, p_syn(x))

    def ICIntegrand(x): return np.log(10) * np.power(10, p_ic(x))

    intersection = sciop.newton(p_diff, 17., fprime=np.polyder(p_diff), fprime2=np.polyder(p_diff, m=2))
    Fsyn = scint.romberg(SynIntegrand, 12., float(intersection))
    Fic = scint.romberg(ICIntegrand, float(intersection), 24.)
    fluence_rat = Fic/Fsyn

    return fluence_rat

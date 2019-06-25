import numpy as np


def Inu_Guold79(jnu, anu, R):
    tau = anu * R
    fnu = np.zeros_like(jnu)
    for i in range(anu.size):
        if tau[i] > 100.0:
            u = 0.5 - 1.0 / tau[i]**2
        elif (tau[i] >= 0.01) & (tau[i] <= 100.0):
            u = 0.5 * (1.0 - 2.0 * (1.0 - (1.0 + tau[i]) * np.exp(-tau[i])) / tau[i]**2)
        else:
            u = (tau[i] / 3.0) - 0.125 * tau[i]**2
        if u > 0.0:
            fnu[i] = 0.125 * u * jnu[i] / (np.pi * anu[i])
        else:
            fnu[i] = 0.25 * jnu[i] * R / np.pi
    return fnu


def Inu_Guold79_v(jnu, anu, R):
    tau = anu * R
    fnu = np.zeros_like(jnu)
    Nt = np.size(jnu, axis=0)
    Nf = np.size(jnu, axis=1)
    for i in range(Nt):
        for j in range(Nf):
            if tau[i, j] > 100.0:
                u = 0.5 - 1.0 / tau[i, j]**2
            elif (tau[i, j] >= 0.01) & (tau[i, j] <= 100.0):
                u = 0.5 * (1.0 - 2.0 * (1.0 - (1.0 + tau[i, j]) * np.exp(-tau[i, j])) / tau[i, j]**2)
            else:
                u = (tau[i, j] / 3.0) - 0.125 * tau[i, j]**2
            if u > 0.0:
                fnu[i, j] = 0.125 * u * jnu[i, j] / (np.pi * anu[i, j])
            else:
                fnu[i, j] = 0.25 * jnu[i, j] * R / np.pi
    return fnu


def opt_depth_blob(absor, R):
    tau = 2. * R * absor
    if (tau <= 1e-50):
        u = 1.
    else:
        if (tau > 100.):
            u = 0.5 - 1. / tau**2
        elif (tau >= 0.01 and tau <= 100.):
            u = 0.5 * (1. - 2. * (1. - (1. + tau) * np.exp(-tau)) / tau**2)
        else:
            u = (tau / 3.) - 0.125 * tau**2
        u = 3. * u / tau
    return u


def intensity_blob(jnu, anu, R):
    Inu = np.zeros_like(jnu)
    for i in range(jnu.size):
        Inu[i] = 2. * R * jnu[i] * opt_depth_blob(anu[i], R)
    return Inu


def opt_depth_slab(absor, r):
    tau = r * absor
    if (tau <= 1e-10):
        u = 1.
    else:
        u = (1. - np.exp(-tau)) / tau
    return u


def intensity_slab(jnu, anu, s):
    tau = anu * s
    Inu = np.zeros_like(jnu)
    for j in range(jnu.size):
        if (tau[j] > 1e-10):
            Inu[j] = s * jnu[j] * (1. - np.exp(-tau[j])) / tau[j]
        else:
            Inu[j] = s * jnu[j]
    return Inu

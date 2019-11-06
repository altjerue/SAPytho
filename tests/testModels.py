# In[]:
import os
import os.path
import SAPytho.SSCCdriver as driver
import SAPytho.spectra as sp
import SAPytho.models as mod
import matplotlib.pyplot as plt

# If one wants to run this script using, e.g., Atoms' Hydrogen package or
# jupyter Notebook, make sure you are in the desired directory
os.chdir('/Users/jesus/lab/cSPEV/tests/SSCC_discret')  # <-- Change accordingly

# In[]: Making a test with SPN98
nuF90, SEDF90 = driver.build_totSpec(only_load=True, inJanskys=False, R=0.0,
                                     dLum=1.1e28, z=0.6, theta_obs=0.0,
                                     dtacc=1e10, tstep=1e-10, tmax=1e7, n0=1e0,
                                     wMBSabs=False, wSSCem=False,
                                     file_label='SPN98test',
                                     ISdir='/Users/jesus/lab/cSPEV/InternalShocks/',
                                     HYB=False, MBS=False, DBG=1)
tF90, LCF90 = driver.build_LCs(1e12, 1e15, only_load=True, inJanskys=False,
                               R=0.0, dLum=1.1e28, z=0.6, theta_obs=0.0,
                               dtacc=1e10, tstep=1e-10, tmax=1e7, n0=1e0,
                               wMBSabs=False, wSSCem=False,
                               file_label='SPN98test',
                               ISdir='/Users/jesus/lab/cSPEV/InternalShocks/',
                               HYB=False, MBS=False, DBG=1)

SPN98py = mod.SPN98(D28=1.1e28, adiab=False)
nuc, num, nu0, tc, tm, t0, FnuPY = SPN98py.fluxSPN98(nuF90, tF90)

LC = sp.LightCurves()
SP = sp.spectrum()
Fnu_lc = LC.integ(1e12, 1e15, len(tF90), nuF90, FnuPY)
Fnu_spec_av = SP.averaged(tF90[0], tF90[-1], len(nuF90), tF90, FnuPY)

plt.yscale('log')
plt.xscale('log')
plt.plot(nuF90, SEDF90)
plt.plot(nuF90, Fnu_spec_av)
plt.show()

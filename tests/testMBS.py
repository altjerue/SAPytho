#
# In[]: Preamble
import numpy as np
# import scipy.optimize as op
import SAPytho.magnetobrem as mbs
import matplotlib.pyplot as pl
# pl.ion()

#
# In[]: Setup global variables
MBS = mbs.mbs()

nu = np.logspace(5.0, 10.0, num=150)
Bmag = 1.0

#
# In[]: Fig. 3.15 in Rueda-Becerril (2017)
#
# This bubble reproduces Fig. 3.15 in my PhD thesis (Rueda-Becerril 2017)
LorFact = 10.0
Xc = nu / MBS.nu_c_iso(Bmag, LorFact)

pl.xscale('log')
pl.yscale('log')
pl.xlim(1e-4, 1e2)
pl.ylim(1e-3, 2.0)
pl.plot(Xc, MBS.Rsync(Xc), c='k', lw=4.0)
pl.plot(Xc, MBS.RMA(Xc, LorFact), c='b')
pl.plot(Xc, MBS.RMAfit(Xc, LorFact), c='r')
# pl.plot(Xc, MBS.FDB08fit(Xc), c='g')
pl.plot(Xc, MBS.Fsync(Xc))


#
# In[]: Testing spTable
#
# This bubble produces the rdiated power spectrum from a moving electron
spTab = mbs.spTable('/Users/jesus/lab/tools/SAPyto/tests/spTable.h5')
X = MBS.chi(nu, Bmag)
gamma = np.logspace(np.log10(1.01), np.log10(100), num=5)
PowSpec = np.zeros_like(X)

pl.xscale('log')
pl.yscale('log')
# pl.xlim(1e-2, 1e4)
pl.ylim(1e-8, 2.0)
for g in gamma:
    for i in range(X.size):
        PowSpec[i] = np.exp(spTab.I1_interp(X[i], g))
    pl.plot(X, PowSpec)

#
# In[]: Testing disTable - j_mb
#
# This bubble produces the emissivity from a distribution of particles
disTab = mbs.disTable('/Users/jesus/lab/tools/SAPyto/tests/disTable.h5')
gamma = np.logspace(np.log10(1.01), 1.0, num=5)
jnu = np.zeros_like(nu)

pl.xscale('log')
pl.yscale('log')
pl.xlim(1e5, 1e10)
pl.ylim(1e-27, 1e-22)
for i in range(gamma.size - 1):
    for j in range(nu.size):
        jnu[j] = disTab.j_mb(nu[j], Bmag, 1.0, gamma[i], gamma[i + 1], 2.5)
    pl.plot(nu, jnu)

#
# In[]: Testing disTable - a_mb
#
# This bubble produces the emissivity from a distribution of particles
disTab = mbs.disTable('/Users/jesus/lab/tools/SAPyto/tests/disTable.h5', RMA=True)
gamma = np.logspace(np.log10(1.01), 1.0, num=5)
anu = np.zeros_like(nu)

pl.xscale('log')
pl.yscale('log')
pl.xlim(1e5, 1e10)
pl.ylim(1e-27, 1e-6)
for i in range(gamma.size - 1):
    for j in range(nu.size):
        anu[j] = disTab.a_mb(nu[j], Bmag, 1.0, gamma[i], gamma[i + 1], 2.5, ambtab=disTab.aRMATable)
    pl.plot(nu, anu)

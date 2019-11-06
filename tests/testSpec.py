# In[]: Loading modules
# import numpy as np
import SAPytho.spectra as spec
import SAPytho.SRtoolkit as SR
import extractor.fromHDF5 as extr
import matplotlib.pyplot as plt


# In[]: Loading data
f = 'HSTOThick-wSSC_FinDif-Ng256Nt300Nf384.h5'
wdir = '/Users/jesus/lab/cSPEV/tests/SSCC_discret/Eq1p30RL79/'

Inu = extr.hdf5Extract2D(wdir + f, 'Inut')
t = extr.hdf5Extract1D(wdir + f, 'time')
nu = extr.hdf5Extract1D(wdir + f, 'frequency')
numt = extr.hdf5ExtractScalar(wdir + f, 'numdt')
numf = extr.hdf5ExtractScalar(wdir + f, 'numdf')

D = SR.Doppler(10.0, 5.0)
Fnu = spec.EnergyFlux(Inu, 4.0793e26, 0.03, D, 1e18)
nu_obs = SR.nu_obs(nu, 0.03, 10.0, 5.0)
t_obs = SR.t_obs(t, 0.03, 10.0, view_angle=5.0)


# In[]: Building light curves
LC = spec.LightCurves()
Fnu_lc = LC.integ(1e12, 1e15, numt, nu_obs, nu_obs * Fnu)
Fnu_lc_mono = LC.integ(1e14, 1e14, numt, nu_obs, nu_obs * Fnu)
Fnu_lc_near = LC.nearest(1e14, nu_obs, nu_obs * Fnu)
Fnu_lc_pwl = LC.pwl_interp(1e14, numt, nu_obs, nu_obs * Fnu)
# nuFnu_lc_Jy = LC.pwl_interp(1e12, numt, nu_obs, nu_obs * Fnu)

# In[]: Plotting light curves
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$t\; \mathrm{[s]}$', fontsize='x-large')
ax.set_ylabel(r'$\nu F_{\nu}\: \mathrm{[erg\; cm^{-2}\; s^{-1}]}$', fontsize='x-large')
ax.plot(t_obs, Fnu_lc)
ax.plot(t_obs, Fnu_lc_mono)
ax.plot(t_obs, Fnu_lc_near)
ax.plot(t_obs, Fnu_lc_pwl, ls='--')
# ax.set_ylim(1e5, 1e17)
# ax.set_xlim(numin, numax)
fig.tight_layout()
fig.show()

# In[]: Building spectra
SP = spec.spectrum()
nuFnu_sp = SP.integ(50.0, 1e3, numf, t_obs, Fnu)
nuFnu_sp_aver = SP.averaged(t_obs[0], t_obs[-1], numf, t_obs, nu_obs * Fnu)
nuFnu_sp_aver = SP.averaged(50.0, 1e3, numf, t_obs, nu_obs * Fnu)
nuFnu_sp_mono = SP.integ(100.0, 100.0, numf, t_obs, nu_obs * Fnu)
nuFnu_sp_near = SP.nearest(100.0, t_obs, nu_obs * Fnu)
nuFnu_sp_pwlw = SP.pwl_interp(100.0, numf, t_obs, nu_obs * Fnu)
# nuFnu_sp_Jy = SP.pwl_interp(1e12, numt, nu_obs, nu_obs * Fnu)

# In[]: Plotting spectra
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\nu\; \mathrm{[Hz]}$', fontsize='x-large')
ax.set_ylabel(r'$\nu F_{\nu}\: \mathrm{[erg\; cm^{-2}\; s^{-1}]}$', fontsize='x-large')
ax.plot(nu_obs, nuFnu_sp)
ax.plot(nu_obs, nuFnu_sp_aver)
ax.plot(nu_obs, nuFnu_sp_mono)
ax.plot(nu_obs, nuFnu_sp_near)
ax.plot(nu_obs, nuFnu_sp_pwlw, ls='--')
ax.set_ylim(1e-30, 1e-15)
# ax.set_xlim(numin, numax)
fig.tight_layout()
fig.show()

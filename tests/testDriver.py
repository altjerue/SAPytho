# In[]: preparing stuff
import os
import os.path
import SAPytho.SSCCdriver as driver
import SAPytho.spectra as sp
import SAPytho.models as mod
import matplotlib.pyplot as plt

# If one wants to run this script using, e.g., Atoms' Hydrogen package or
# jupyter Notebook, make sure you are in the desired directory
os.chdir('/dir/to/runs/folder')  # <-- Change accordingly


# These are simple examples for the simplest simulation
params = {'HYB': False,
          'MBS': False,
          'wCool': False,
          'wMBSabs': False,
          'wSSCem': False,
          'ISdir': '/dir/to/InternalShocks/',  # <-- Change this line accordingly
          'params_file': 'input.par'}

# In[]: Compute an SED but compile and run first
x, y = driver.build_SEDs(only_load=False, **params)
plt.yscale('log')
plt.xscale('log')
plt.plot(x, y)
plt.show()

# In[]: Compute a light curve but without compiling nor running
x, y = driver.build_LCs(1e12, 1e15, **params)
plt.yscale('log')
plt.xscale('log')
plt.plot(x, y)
plt.show()

# In[]: Compute averaged spectrum in the perios [t_min, t_max] in janskys
x, y = driver.build_avSpec(5e1, 2e3, inJanskys=True, **params)
plt.yscale('log')
plt.xscale('log')
plt.plot(x, y)
plt.show()

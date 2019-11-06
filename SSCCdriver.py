import os
import SAPytho.spectra as spec
from SAPytho.misc import fortran_double
import SAPytho.SRtoolkit as SR
import extractor.fromHDF5 as extr


#
#  #####    ##   #####    ##   #    #  ####
#  #    #  #  #  #    #  #  #  ##  ## #
#  #    # #    # #    # #    # # ## #  ####
#  #####  ###### #####  ###### #    #      #
#  #      #    # #   #  #    # #    # #    #
#  #      #    # #    # #    # #    #  ####
class SSCC_params(object):

    def params(self):
        # -----  PARAMETERS  -----
        self.R = 1e18                   # radius of emitting region (assuming spherical)
        self.R0 = 1e15                  # radius of emitting region (assuming spherical)
        self.Rinit = 1e15               # radius of emitting region (assuming spherical)
        self.dLum = 4.0793e26           # luminosity distance (default Mrk 421)
        self.z = 0.03                   # redshift (default Mrk 421)
        self.gamma_bulk = 1e2           # emitting region bulk Lorentz factor
        self.theta_obs = 5.0            # observer viewing angle
        self.B = 1.0                    # magnetic field magnitude
        self.b_index = 0.0              # magnetic field decay index
        self.theta_e = 10.0             # electrons temperature
        self.dtacc = 1e2                # injection period
        self.tstep = 1e-2               # time step factor
        self.tmax = 1e5                 # maximum time
        self.Q0 = 1.0                   # num. dens. of particles injected per second
        self.g1 = 1e2                   # power-law min Lorentz factor
        self.g2 = 1e4                   # power-law max Lorentz factor
        self.gmin = 1.01                # EED minimum Lorentz factor
        self.gmax = 2e4                 # EED maximum Lorentz factor
        self.qind = 2.5                 # EED power-law index
        self.numin = 1e7                # minimum frequency
        self.numax = 1e15               # maximum frequency
        self.numbins = 128              # number of EED bins
        self.numdt = 300                # number of time steps
        self.numdf = 256                # number of frequencies
        self.cool_kind = 1              # kind of cooling
        self.time_grid = 1              # kind of cooling
        # -----  ARGS OF THE EXECUTABLE  -----
        self.wCool = True               # variable cooling
        self.wMBSabs = True             # compute MBS self-absorption
        self.wSSCem = True              # compute SSC emissivity
        # -----  INPUT AND OUTPUT  -----
        self.file_label = 'DriverTest'  # a label to identify each output
        self.exec_dir = './'               # address to InternalShocks, must end with '/'
        self.params_file = 'input.par'  # name of the parameters file
        # -----  COMPILER PARAMS  -----
        self.HYB = True                 # compile with HYB=1 flag
        self.MBS = True                 # compile with MBS=1 flag
        self.arc = 'i7'                 # compile with specific arch flag
        self.OMP = False                # compile with OpenMP
        self.DBG = False                # compile for debugging

    def __init__(self, **kwargs):
        self.params()
        self.__dict__.update(kwargs)

    def write_params(self):
        with open(self.params_file, 'w') as f:
            print(fortran_double(self.R), '! Radius', file=f)
            print(fortran_double(self.R0), '! Normalization radius', file=f)
            print(fortran_double(self.Rinit), '! Initial radius', file=f)
            print(fortran_double(self.dLum), '! luminosity distance', file=f)
            print(fortran_double(self.z), '! redshift', file=f)
            print(fortran_double(self.gamma_bulk), '! bulk Lorentz factor', file=f)
            print(fortran_double(self.theta_obs), '! viewing angle', file=f)
            print(fortran_double(self.B), '! magnetic field', file=f)
            print(fortran_double(self.b_index), '! magnetic field decay index', file=f)
            print(fortran_double(self.theta_e), '! electrons temperature', file=f)
            print(fortran_double(self.dtacc), '! injection period',  file=f)
            print(fortran_double(self.tstep), '! time step factor', file=f)
            print(fortran_double(self.tmax), '! maximum time', file=f)
            print(fortran_double(self.Q0), '! num. dens. of particles injected', file=f)
            print(fortran_double(self.g1), '! power-law min Lorentz factor', file=f)
            print(fortran_double(self.g2), '! power-law max Lorentz factor', file=f)
            print(fortran_double(self.gmin), '! EED min Lorentz factor', file=f)
            print(fortran_double(self.gmax), '! EED max Lorentz factor', file=f)
            print(fortran_double(self.qind), '! EED power-law index', file=f)
            print(fortran_double(self.numin), '! min frequency', file=f)
            print(fortran_double(self.numax), '! max frequency', file=f)
            print(self.numbins, '! number of EED bins', file=f)
            print(self.numdt, '! number of time steps', file=f)
            print(self.numdf, '! number of frequencies', file=f)
            print(self.cool_kind, '! kind of cooling', file=f)
            print(self.time_grid, '! kind of time grid', file=f)
            print(self.file_label, '! label to identify each output', file=f)

    def output_file(self):
        outf = ''
        argv = ''
        if self.HYB:
            outf += 'H'
        else:
            outf += 'P'

        if self.MBS:
            outf += 'M'
        else:
            outf += 'S'

        if self.wCool:
            outf += 'V'
            argv += ' T'
        else:
            outf += 'C'
            argv += ' F'

        if self.wMBSabs:
            outf += 'O'
            argv += ' T'
        else:
            outf += 'T'
            argv += ' F'

        if self.wSSCem:
            outf += 'wSSC'
            argv += ' T'
        else:
            outf += 'oSSC'
            argv += ' F'

        return outf + '-' + self.file_label + '.h5', argv


#
#  #####  #    # #    #
#  #    # #    # ##   #
#  #    # #    # # #  #
#  #####  #    # #  # #
#  #   #  #    # #   ##
#  #    #  ####  #    #
class runSSCC(object):
    def __init__(self, **kwargs):
        self.par = SSCC_params(**kwargs)
        self.par.write_params()
        self.outfile, self.argv = self.par.output_file()
        self.cwd = os.getcwd()

    def compile(self):
        make = 'make NewSSCC -j4'
        if self.par.arc in ['i7', 'corei7', 'I7', 'COREI7']:
            make += ' COREI7=1'
        else:
            make += ' NATIVE=1'

        if self.par.OMP:
            make += ' OPENMP=1'

        if self.par.DBG:
            make += ' DBG=1'

        if self.par.HYB:
            make += ' HYB=1'

        if self.par.MBS:
            make += ' MBS=1'

        os.chdir(self.par.exec_dir)
        print("--> Running Makefile:\n   ", make, "\n")
        os.system(make)
        os.chdir(self.cwd)
        print("\n--> Compilation successful\n")

    def runNewSSCC(self):
        run_cmd = '{0}/NewSSCC {1}{2}'.format(self.par.exec_dir, self.par.params_file, self.argv)
        print("\n--> Running:\n  ", run_cmd, "\n")
        os.system(run_cmd)
        print("\n--> NewSSCC ran successfully")

    def cleanup(self):
        os.chdir(self.par.exec_dir)
        os.system("make clean")
        os.chdir(self.cwd)


#
#   ####  #    # ##### #####  #    # #####
#  #    # #    #   #   #    # #    #   #
#  #    # #    #   #   #    # #    #   #
#  #    # #    #   #   #####  #    #   #
#  #    # #    #   #   #      #    #   #
#   ####   ####    #   #       ####    #
#

def build_LCs(nu_min, nu_max, dset='Inut', only_load=True, inJanskys=False, **kwargs):
    run = runSSCC(**kwargs)
    if not only_load:
        run.cleanup()
        run.compile()
        run.runNewSSCC()
    D = SR.Doppler(run.par.gamma_bulk, run.par.theta_obs)
    nu = extr.hdf5Extract1D(run.outfile, 'frequency')
    t = extr.hdf5Extract1D(run.outfile, 'time')
    nu_obs = SR.nu_obs(nu, run.par.z, run.par.gamma_bulk, run.par.theta_obs)
    t_obs = SR.t_obs(t, run.par.z, run.par.gamma_bulk, view_angle=run.par.theta_obs)
    Inu = extr.hdf5Extract2D(run.outfile, dset)
    Fnu = spec.flux_dens(Inu, run.par.dLum, run.par.z, D, run.par.R)
    LC = spec.LightCurves()

    if inJanskys:
        return t_obs, spec.conv2Jy(LC.integ(nu_min, nu_max, run.par.numdt, nu_obs, Fnu))
    else:
        return t_obs, LC.integ(nu_min, nu_max, run.par.numdt, nu_obs, Fnu)


def build_avSpec(t_min, t_max, dset='Inut', only_load=True, inJanskys=False, **kwargs):
    run = runSSCC(**kwargs)
    if not only_load:
        run.cleanup()
        run.compile()
        run.runNewSSCC()
    D = SR.Doppler(run.par.gamma_bulk, run.par.theta_obs)
    nu = extr.hdf5Extract1D(run.outfile, 'frequency')
    t = extr.hdf5Extract1D(run.outfile, 'time')
    nu_obs = SR.nu_obs(nu, run.par.z, run.par.gamma_bulk, run.par.theta_obs)
    t_obs = SR.t_obs(t, run.par.z, run.par.gamma_bulk, view_angle=run.par.theta_obs)
    Inu = extr.hdf5Extract2D(run.outfile, dset)
    Fnu = spec.flux_dens(Inu, run.par.dLum, run.par.z, D, run.par.R)
    sp = spec.spectrum()

    if inJanskys:
        return nu_obs, spec.conv2Jy(sp.averaged(t_min, t_max, run.par.numdf, t_obs, Fnu))
    else:
        return nu_obs, sp.averaged(t_min, t_max, run.par.numdf, t_obs, Fnu)


def build_totSpec(dset='Inut', only_load=True, inJanskys=False, **kwargs):
    run = runSSCC(**kwargs)
    if not only_load:
        run.cleanup()
        run.compile()
        run.runNewSSCC()
    D = SR.Doppler(run.par.gamma_bulk, run.par.theta_obs)
    nu = extr.hdf5Extract1D(run.outfile, 'frequency')
    t = extr.hdf5Extract1D(run.outfile, 'time')
    sLum = extr.hdf5Extract1D(run.outfile, 'sen_lum')
    nu_obs = SR.nu_obs(nu, run.par.z, run.par.gamma_bulk, run.par.theta_obs)
    t_obs = SR.t_obs(t, run.par.z, run.par.gamma_bulk, view_angle=run.par.theta_obs)
    Inu = extr.hdf5Extract2D(run.outfile, dset)
    Fnu = spec.flux_dens(Inu, run.par.dLum, run.par.z, D, run.par.dLum - sLum)
    # print(Inu)
    # print(Fnu)
    sp = spec.spectrum()
    Spec = sp.averaged(t_obs[0], t_obs[-1], run.par.numdf, t_obs, Fnu)

    if inJanskys:
        return nu_obs, spec.conv2Jy(Spec)
    else:
        return nu_obs, Spec

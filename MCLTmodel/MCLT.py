import numpy as np
import pandas as pd
from photon import Photon, PhotonILT
import time
from multiprocessing import Pool

'''
Input parameters to a Monte Carlo simulation of photon propagation through a single medium layer
values based on VDHulst table verification data

Tissue parameters
(relate to optical properties)
    mua - absorption coefficient [1/cm]
    mus - scattering coefficient [1/cm]
    g - anisotropy factor
    n - refractive index

Spatial boundary parameters
(relate to sample surface)
    d - thickness of slab [cm]
    nf - refractive index external medium frontside of sample
    nr - refractive index external medium rearside of sample

Grid parameters
    Nz - number of z bins (depth)
    Nr - number of r bins (depth)
    na - number of alpha bins (depth)
    dz - z grid separation [cm]
    dr - r grid separation [cm]
    da - alpha grid separation [radian]

Monte Carlo parameters
    Nphotons - total number of photons simulated
'''

# Retrieve look-up table for determining scattered angle from arbitrary phase function
# Invert and interpolate using cubic spline to retrieve relation between RV (CDF) and cos(theta)
# tables = np.load('input/inverseMieTable.pkl', allow_pickle=True)
HGstarData = np.load('input/phasefunction/inverseHGstarTable.pkl', allow_pickle=True)

#MieData contains list of 5 size distributions that each contain data for 29 wavelengths: an inverse lookup table and the associated angles
# mieData = np.load('input/phasefunction/inverseLognormalMieTable.pkl', allow_pickle=True)
mieData = np.load('input/phasefunction/ILTMIEsizedistributions.pkl', allow_pickle=True)
mieWavelengthDependentData =  np.load('input/phasefunction/WavelengthdependentLognormalILT.pkl',  allow_pickle=True)
mieComparisonData = np.load('input/phasefunction/inverseLognormalMieComparisonTable.pkl', allow_pickle=True)
mieDependentData = np.load('input/phasefunction/ILTMIEdependentscattering.pkl', allow_pickle=True)
mieWithoutDependentData = np.load('input/phasefunction/ILTMIEwithoutdependentscattering.pkl', allow_pickle=True)
# Retrieve absorption and scattering coefficients from dataframe generated by notebook
# notebooks/Smalt data to coefficient.ipynb

df = pd.read_pickle('input/dataframe/mua_mus_dataframe.pkl')
# df = pd.read_pickle('input/dataframe/mua_mus_dataframe_backup_400-840.pkl')
# df = pd.read_pickle('input/dataframe/mua_mus_dataframe_adjustedbackup.pkl')


# multiprocessor function that computes the reflection from nphotons photon trajectories, can be called through a starmap in Pool
def worker(x, nphotons, mua, mus, g, n, d, nf, nr, Nz, Nr, Na, dr):
    np.random.seed(x)
    R=0
    for j in range(nphotons):
        phtn = Photon(mua, mus, g, n, d, nf, nr, Nz, Nr, Na, dr)
        phtn.Propagate()
        R += phtn.output[1]
    return R/nphotons

# multiprocessor function that computes the reflection from nphotons photon trajectories with ILT method, can be called through a starmap in Pool
def workerILT(x, nphotons, mua, mus, g, n, d, nf, nr, Nz, Nr, Na, dr, invLT, cos_theta_star):
    np.random.seed(x)
    R=0
    for j in range(nphotons):
        phtn = PhotonILT(mua, mus, g, n, d, nf, nr, Nz, Nr, Na, dr, invLT, cos_theta_star)
        phtn.Propagate()
        R += phtn.output[1]
    return R/nphotons


class MonteCarlo:
    def __init__(self,  mua, mus, g, n, d, nf, nr, Nz, Nr, Na, dr):
        self.mua = mua
        self.mus = mus
        self.g = g
        self.n = n

        self.d = d
        self.nf = nf
        self.nr = nr

        self.Nz = Nz
        self.Nr = Nr
        self.Na = Na
        self.dz = d / Nz  # z grid separation [cm]
        self.dr = dr
        self.da = 2 * np.pi / Na  # alpha grid separation [radian]


    ####################################
    # Parameter analysis functions
    ####################################


    def convergence_Nphot(self):

        # for mua in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        #
        #     self.mua = mua
        #     for mus in [10, 50, 100]:
        #         self.mus = mus

        exponents = 7
        batches = 100

        avgdata = np.zeros((exponents-1, batches))
        print(self.mua, self.mus)
        for n in range(1, exponents):
            Nphotons = 10**n
            print(Nphotons)
            for batch in range(batches):
                R_r = np.zeros(self.Nr)
                for j in range(Nphotons):

                    phtn = Photon(self.mua, self.mus, self.g, self.n, self.d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr)
                    phtn.Propagate()
                    R_r[phtn.output[0]] += phtn.output[1]

                avgdata[n-1, batch] = np.sum(R_r) / Nphotons

            title = 'output/Thesis/convergence/Thesis7exponents'
            np.save(title + '.npy', avgdata)

    # #simulate spectrum of smalt layer for different layer thicknesses
    def HG_thickness_smalt_simulation(self, wavelengtharray, id=0, repititions = 1, Nphotons = int(1e5), pigment = 'smalt', param=1, withVarnish = False, progress = False, saveOutput=False):

        # gs = [0.905, 0.96, 0.995] #coming across with the phase function from the largerst three lognormal size distributions
        # ds = [.0059, .0046, .0085] #correct for layer thickness difference
        # ns = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]  #refractive index of paint layer
        # gs = [0.053447619816065425, 0.21098339770011817, 0.4395459709608761, 0.6390083213533262, 0.74861645359337]  #gvalues for comparison with Mie phase functions

        pool = Pool(nprocessors)
        Multiphotons = int(Nphotons / nprocessors)

        randomseeds = list(range(0, len(wavelengtharray) * nprocessors, nprocessors))[::-1] # list with randomseeds intervals

        ds = [.001, .002, .003, .004, .005, .006, .007, .008, .009, .01]   # thickness of paint layer range from 10 to 100 microns with interval of 10 microns

        for intervalid, d in enumerate(ds):

            # self.d = param

            R = np.zeros(len(wavelengtharray))
            # Rerror = np.zeros(len(wavelengtharray))

            tStart = time.time()
            for i, wv in enumerate(wavelengtharray):
                parameters = Multiphotons, self.mua[i], self.mus[i], self.g, self.n, d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr
                data = [(randomseeds[i]+j, *parameters) for j in range(nprocessors)]

                R[i] = np.mean(pool.starmap(worker, data))

            if saveOutput:
                # title = 'output/phasefunction-results/HG/{}Reflection_Nphot{}_g{}'.format(pigment, Nphotons, np.round(g,4))
                title = 'output/Thesis/layerthickness/{}Reflection_Nphot{}_HG_d{}'.format(pigment, Nphotons, d)
                np.save(title +'.npy', R)
            if progress:
                tEnd = time.time() 
                print('spend time:' + str(tEnd - tStart))
        pool.close()
        pool.join()




    # #simulate spectrum of smalt layer for refractive indices
    def HG_RI_smalt_simulation(self, wavelengtharray, id=0, repititions=1, Nphotons=int(1e5),
                                      pigment='smalt', param=1, withVarnish=False, progress=False,
                                      saveOutput=False):
        pool = Pool(nprocessors)
        Multiphotons = int(Nphotons / nprocessors)

        ns = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]  #refractive index of paint layer
        for intervalid, n in enumerate(ns):

            # self.d = param

            R = np.zeros(len(wavelengtharray))
            # Rerror = np.zeros(len(wavelengtharray))

            tStart = time.time()
            for i, wv in enumerate(wavelengtharray):
                parameters = Multiphotons, self.mua[i], self.mus[i], self.g, n, self.d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr
                data = [(i, *parameters) for i in range(nprocessors)]
                R[i] = np.mean(pool.starmap(worker, data))

            if saveOutput:
                # title = 'output/phasefunction-results/HG/{}Reflection_Nphot{}_g{}'.format(pigment, Nphotons, np.round(g,4))
                title = 'output/Thesis/refractiveindex/{}Reflection_Nphot{}_HG_n{}'.format(pigment, Nphotons, n)
                np.save(title +'.npy', R)
            if progress:
                tEnd = time.time()
                print('spend time:' + str(tEnd - tStart))
        pool.close()
        pool.join()


    # #simulate spectrum of smalt layer for refractive indices
    def HG_anisotropy_smalt_simulation(self, wavelengtharray, id=0, repititions=1, Nphotons=int(1e5),
                                      pigment='smalt', param=1, withVarnish=False, progress=False,
                                      saveOutput=False):
        pool = Pool(nprocessors)
        Multiphotons = int(Nphotons / nprocessors)

        glist = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0. ,  0.1, 0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8, .9] #anisotropy factor
        for intervalid, g in enumerate(glist):

            # self.d = param

            R = np.zeros(len(wavelengtharray))
            # Rerror = np.zeros(len(wavelengtharray))

            tStart = time.time()
            for i, wv in enumerate(wavelengtharray):
                parameters = Multiphotons, self.mua[i], self.mus[i], g, self.n, self.d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr
                data = [(i, *parameters) for i in range(nprocessors)]
                R[i] = np.mean(pool.starmap(worker, data))
            if saveOutput:
                title = 'output/Thesis/anisotropyfactor/{}Reflection_Nphot{}_HG_g{}'.format(pigment, Nphotons, g)
                np.save(title +'.npy', R)
            if progress:
                tEnd = time.time()  
                print('spend time:' + str(tEnd - tStart))
        pool.close()
        pool.join()

        ####################################
        # Comparison HG and Mie
        ####################################

    # #simulate spectrum of smalt layer using Mie theory phase function data
    def simulate_smalt_layer_ComparisonHGMie(self, wavelengtharray, id=0, repititions = 1, Nphotons = int(5e5), pigment = 'smalt', param=1, withVarnish = False, progress = False, saveOutput=False):

        # Run HG phase function simulations

        glist = [0.053458193791593306, #anisotropy factors coming across with the anisotropy in the Mie phase functions
              0.21102274063526796,
              0.4395878930354646,
              0.6390355394767377,
              0.7486360530262498]

        pool = Pool(nprocessors)
        Multiphotons = int(Nphotons / nprocessors)

        for intervalid, g in enumerate(glist):

            # self.d = param

            R = np.zeros(len(wavelengtharray))
            # Rerror = np.zeros(len(wavelengtharray))

            tStart = time.time()
            for i, wv in enumerate(wavelengtharray):
                parameters = Multiphotons, self.mua[i], self.mus[i], g, self.n, self.d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr
                data = [(i, *parameters) for i in range(nprocessors)]
                R[i] = np.mean(pool.starmap(worker, data))
            if saveOutput:
                title = 'output/Thesis/comparisonMieHG/{}Reflection_Nphot{}_HG{}'.format(pigment, Nphotons, intervalid)
                np.save(title +'.npy', R)
            if progress:
                tEnd = time.time() 
                print('spend time:' + str(tEnd - tStart))
        pool.close()
        pool.join()

        # Run Mie phase function simulations

        pool = Pool(nprocessors)

        for intervalid, sizedistribution in enumerate(mieComparisonData):
            invLT, cos_theta_star = sizedistribution

            # self.d = param
            # gs = [0.053447619816065425, 0.21098339770011817, 0.4395459709608761, 0.6390083213533262, 0.74861645359337,
            #       0.9120370242111717]  # gvalues coming across with Mie phase functions

            R = np.zeros(len(wavelengtharray))
            # Rerror = np.zeros(len(wavelengtharray))

            tStart = time.time()
            for i, wv in enumerate(wavelengtharray):
                parameters = Multiphotons, self.mua[i], self.mus[i], self.g, self.n, self.d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr, invLT, cos_theta_star
                data = [(i, *parameters) for i in range(nprocessors)]
                R[i] = np.mean(pool.starmap(workerILT, data))

            if saveOutput:
                # title = 'output/phasefunction-results/Mie-Lognormal-distribution/{}Reflection_Nphot{}_Mie{}'.format(pigment, Nphotons, intervalid)
                title = 'output/Thesis/comparisonMieHG/{}Reflection_Nphot{}_Mie{}'.format(pigment, Nphotons, intervalid)
                np.save(title +'.npy', R)
            if progress:
                tEnd = time.time()
                print('spend time:' + str(tEnd - tStart))
        pool.close()
        pool.join()


        ####################################
        # Simulate reflection spectra of Mie lognormal size distributions associated with paint layer samples
        ####################################

    def simulate_smalt_layer_MieSizeDistributions(self, wavelengtharray, id=0, repititions=1, Nphotons=int(5e5),
                                             pigment='smalt', param=1, withVarnish=False, progress=False,
                                             saveOutput=False):

        pool = Pool(nprocessors)
        Multiphotons = int(Nphotons / nprocessors)

        for intervalid, sizedistribution in enumerate(mieData):
            invLT, cos_theta_star = sizedistribution

            # self.d = param
            # gs = [0.053447619816065425, 0.21098339770011817, 0.4395459709608761, 0.6390083213533262, 0.74861645359337,
            #       0.9120370242111717]  # gvalues coming across with Mie phase functions

            R = np.zeros(len(wavelengtharray))
            # Rerror = np.zeros(len(wavelengtharray))

            tStart = time.time()
            for i, wv in enumerate(wavelengtharray):
                parameters = Multiphotons, self.mua[i], self.mus[i], self.g, self.n, self.d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr, invLT, cos_theta_star
                data = [(i, *parameters) for i in range(nprocessors)]
                R[i] = np.mean(pool.starmap(workerILT, data))

            if saveOutput:
                title = 'output/Thesis/sizedistributions/{}Reflection_Nphot{}_MieSizeDistribution{}'.format(
                    pigment, Nphotons, intervalid)
                np.save(title + '.npy', R)
            if progress:
                tEnd = time.time() 
                print('spend time:' + str(tEnd - tStart))

        pool.close()
        pool.join()

        glist = [0.905, 0.96, 0.995] #coming across with the phase function from the largerst three lognormal size distributions

        pool = Pool(nprocessors)

        for intervalid, g in enumerate(glist):

            # self.d = param

            R = np.zeros(len(wavelengtharray))
            # Rerror = np.zeros(len(wavelengtharray))

            tStart = time.time()
            for i, wv in enumerate(wavelengtharray):
                parameters = Multiphotons, self.mua[i], self.mus[i], g, self.n, self.d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr
                data = [(i, *parameters) for i in range(nprocessors)]
                R[i] = np.mean(pool.starmap(worker, data))
            if saveOutput:
                title = 'output/Thesis/sizedistributions/{}Reflection_Nphot{}_HG{}'.format(pigment, Nphotons, intervalid+3)
                np.save(title +'.npy', R)
            if progress:
                tEnd = time.time() 
                print('spend time:' + str(tEnd - tStart))
        pool.close()
        pool.join()


        # Run wavelength dependent Mie phase function simulations

        pool = Pool(nprocessors)

        for intervalid, sizedistribution in enumerate(mieWavelengthDependentData):

            # self.d = param
            # gs = [0.053447619816065425, 0.21098339770011817, 0.4395459709608761, 0.6390083213533262, 0.74861645359337,
            #       0.9120370242111717]  # gvalues coming across with Mie phase functions

            R = np.zeros(len(wavelengtharray))
            # Rerror = np.zeros(len(wavelengtharray))

            tStart = time.time()
            for i, wv in enumerate(wavelengtharray):
                invLT, cos_theta_star = sizedistribution[i]  #wavelength dependent look-up tables
                parameters = Multiphotons, self.mua[i], self.mus[i], self.g, self.n, self.d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr, invLT, cos_theta_star
                data = [(i, *parameters) for i in range(nprocessors)]
                R[i] = np.mean(pool.starmap(workerILT, data))

            if saveOutput:
                # title = 'output/phasefunction-results/Mie-Lognormal-distribution/{}Reflection_Nphot{}_Mie{}'.format(pigment, Nphotons, intervalid)
                title = 'output/Thesis/sizedistributions/{}Reflection_Nphot{}_MieSizeDistributionWavelengthDependent{}'.format(
                    pigment, Nphotons, intervalid)
                np.save(title + '.npy', R)
            if progress:
                tEnd = time.time()
                print('spend time:' + str(tEnd - tStart))
        pool.close()
        pool.join()

    def simulate_smalt_layer_Mie_DependentScattering(self, wavelengtharray, id=0, repititions=1, Nphotons=int(5e5),
                                             pigment='smalt', param=1, withVarnish=False, progress=False,
                                             saveOutput=False):

        pool = Pool(nprocessors)
        Multiphotons = int(Nphotons / nprocessors)

        for intervalid, sizedistribution in enumerate(mieWithoutDependentData):
            invLT, cos_theta_star = sizedistribution

            # self.d = param
            # gs = [0.053447619816065425, 0.21098339770011817, 0.4395459709608761, 0.6390083213533262, 0.74861645359337,
            #       0.9120370242111717]  # gvalues coming across with Mie phase functions

            R = np.zeros(len(wavelengtharray))
            # Rerror = np.zeros(len(wavelengtharray))

            tStart = time.time()
            for i, wv in enumerate(wavelengtharray):
                parameters = Multiphotons, self.mua[i], self.mus[i], self.g, self.n, self.d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr, invLT, cos_theta_star
                data = [(i, *parameters) for i in range(nprocessors)]
                R[i] = np.mean(pool.starmap(workerILT, data))

            if saveOutput:
                title = 'output/Thesis/sizedistributions/{}Reflection_Nphot{}_MieWithoutDependent{}'.format(
                    pigment, Nphotons, intervalid)
                np.save(title + '.npy', R)
            if progress:
                tEnd = time.time()  
                print('spend time:' + str(tEnd - tStart))

        pool.close()
        pool.join()

        pool = Pool(nprocessors)

        for intervalid, sizedistribution in enumerate(mieDependentData):
            invLT, cos_theta_star = sizedistribution

            # self.d = param
            # gs = [0.053447619816065425, 0.21098339770011817, 0.4395459709608761, 0.6390083213533262, 0.74861645359337,
            #       0.9120370242111717]  # gvalues coming across with Mie phase functions

            R = np.zeros(len(wavelengtharray))
            # Rerror = np.zeros(len(wavelengtharray))

            tStart = time.time()
            for i, wv in enumerate(wavelengtharray):
                parameters = Multiphotons, self.mua[i], self.mus[i], self.g, self.n, self.d, self.nf, self.nr, self.Nz, self.Nr, self.Na, self.dr, invLT, cos_theta_star
                data = [(i, *parameters) for i in range(nprocessors)]
                R[i] = np.mean(pool.starmap(workerILT, data))

            if saveOutput:
                title = 'output/Thesis/sizedistributions/{}Reflection_Nphot{}_MieDependent{}'.format(
                    pigment, Nphotons, intervalid)
                np.save(title + '.npy', R)
            if progress:
                tEnd = time.time()
                print('spend time:' + str(tEnd - tStart))

        pool.close()
        pool.join()


if __name__ == '__main__':

    mua = 10
    mus = 90
    mut = mua + mus
    g = .8
    n = 1.5

    d = .008  # 80 microns (d=cm)
    nf = 1.
    nr = 1.

    Nz = 30
    Nr = 50
    Na = 10
    dz = d / Nz
    dr = .01
    da = 2 * np.pi / Na

    model = MonteCarlo(mua, mus, g, n, d, nf, nr, Nz, Nr, Na, dr)
    model.convergence_Nphot()

    # model.simulate_titanium_layer(id = 'withoutVarnish', progress=True)
    # model.simulate_titanium_layer(id = 'withVarnish', withVarnish=True, progress=True)

    wavelengths = df.wavelength
    mua = list(df.absorption)
    mus = list(df.scattering)

    model = MonteCarlo(mua, mus, g, n, d, nf, nr, Nz, Nr, Na, dr)
    Nphot=int(5e5)
    nprocessors = 6

    # model.simulate_smalt_layer_Mie(wavelengths , Nphotons=Nphot, id = 'Mie', progress=True, saveOutput=True)
    # model.HG_thickness_smalt_simulation(wavelengths, Nphotons=Nphot, id = 'HG', progress=True, saveOutput=True)
    # model.HG_RI_smalt_simulation(wavelengths, Nphotons=Nphot, id = 'HG', progress=True, saveOutput=True)
    # model.HG_anisotropy_smalt_simulation(wavelengths, Nphotons=Nphot, id = 'HG', progress=True, saveOutput=True)
    # model.simulate_smalt_layer_ComparisonHGMie(wavelengths, Nphotons=Nphot, progress=True, saveOutput=True)
    # model.simulate_smalt_layer_MieSizeDistributions(wavelengths, Nphotons=Nphot, id='Mie', progress=True, saveOutput=True)
    # model.simulate_smalt_layer_Mie_DependentScattering(wavelengths, Nphotons=Nphot, id='Mie', progress=True, saveOutput=True)

    print('Finished!')
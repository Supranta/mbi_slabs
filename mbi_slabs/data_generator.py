import jax.numpy as np
import numpy as onp
import h5py as h5

class DataGenerator:
    def __init__(self, fourier_transform, sigma_noise, nbar):
        self.F = fourier_transform
        self.sigma_noise = sigma_noise
        self.nbar = nbar
        
    def generate_shape_data(self, kappa_list, kappa_ia_list):
        """Generate shape data from kappa maps"""
        shape_data = []
        
        for kappa, kappa_ia in zip(kappa_list, kappa_ia_list):
            gamma_1, gamma_2 = self.F.kappa2gamma(kappa)
            gamma_ia_1, gamma_ia_2 = self.F.kappa2gamma(kappa_ia)
            
            e1_obs = gamma_1 + gamma_ia_1 + np.array(self.sigma_noise * onp.random.normal(size=kappa.shape))
            e2_obs = gamma_2 + gamma_ia_2 + np.array(self.sigma_noise * onp.random.normal(size=kappa.shape))
            
            shape_data.append(np.array([e1_obs, e2_obs]))
            
        return shape_data
    
    def generate_galaxy_counts(self, proj_density_list):
        """Generate galaxy counts from projected density"""
        N_gals_data = []
        
        for proj_density in proj_density_list:
            mu = self.nbar * (1. + proj_density)
            N_gals = onp.random.poisson(np.clip(mu, 1e-3))
            N_gals_data.append(np.array(N_gals))
            
        return N_gals_data

def write_data(datafile, dens_slabs_true, shape_data, counts_data):
    with h5.File(datafile, 'w') as f:
        f['delta_slabs'] = dens_slabs_true
        f['shape_data']  = np.array(shape_data)
        f['N_gals_data'] = np.array(counts_data)

def read_data(datafile):
    with h5.File(datafile, 'r') as f:
        shape_data  = f['shape_data'][:]
        counts_data = f['N_gals_data'][:]
    return shape_data, counts_data

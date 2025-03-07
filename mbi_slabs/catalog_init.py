# catalog_init.py
import jax.numpy as np
import numpy as onp
import h5py as h5
from dataclasses import dataclass
from typing import List, Tuple
import jax_cosmo as jc
from .utils import get_slabs_z_boundaries
from .observables import HistogramDist, get_cosmo 
from tqdm import trange
import os

@dataclass
class Catalogs:
    nz_src_list: List
    nz_lens_list: List
    N_SRC_BINS: int
    N_LENS_BINS: int

class PolyDist:
    def __init__(self, z_arr, polyfit=None, Om_fid=0.27, Om_minmax=[0.15,0.45], deg=4):
        self.z_arr   = z_arr
        self.deg     = deg
        self.Om_fid    = Om_fid
        self.Om_minmax = Om_minmax
        self.chi_fid = self.get_distances(self.Om_fid)
        self.poly_deg = np.arange(self.deg + 1)[::-1]
        if polyfit is not None:
            self.polyfit = polyfit
        else:
            self.chi_arr = self.create_chi_arr(self.Om_minmax)
            self.x = self.Om_arr / self.Om_fid - 1.
            self.y = self.chi_arr / self.chi_fid - 1.
            self.fit_polynomial()
            self.test_polyfit()
        
    def fit_polynomial(self):
        self.polyfit = np.polyfit(self.x, self.y, self.deg)
        
    def get_distances(self, Omega_m):
        cosmo = get_cosmo([Omega_m, 0.82])
        chi = jc.background.radial_comoving_distance(cosmo, 1. / (1. + self.z_arr)) 
        return chi + 1e-15

    def get_poly_pred(self, x_pred):
        return np.sum(self.polyfit * np.power(x_pred, self.poly_deg)[:,np.newaxis], axis=0)

    def pred_distance(self, Om_pred):
        x_pred = Om_pred / self.Om_fid - 1.
        y_pred = self.get_poly_pred(x_pred)
        return self.chi_fid * (y_pred + 1.)        
        
    def create_chi_arr(self, Om_minmax, N_Om=100):
        Omega_m_min, Omega_m_max = Om_minmax
        self.Om_arr = np.linspace(Omega_m_min, Omega_m_max, N_Om)
        chi_slab_boundaries_list = []
        for i in trange(N_Om):
            chi_slab_boundaries = self.get_distances(self.Om_arr[i])
            chi_slab_boundaries_list.append(chi_slab_boundaries)
        return np.array(chi_slab_boundaries_list)

    def test_polyfit(self, N_test=100):
        Omega_m_min, Omega_m_max = self.Om_minmax
        
        Om_pred = onp.random.uniform(Omega_m_min, Omega_m_max, N_test)

        max_err_list = []

        for i in trange(N_test):
            chi_pred_poly = self.pred_distance(Om_pred[i])
            chi_pred_jc   = self.get_distances(Om_pred[i])
            max_err = np.abs(chi_pred_poly / chi_pred_jc - 1.).max()
            max_err_list.append(max_err)

        max_err_arr = np.array(max_err_list)
        max_index = np.argmax(max_err_arr)
        print("Maximum error: %2.5f percent at Omega_m=%2.3f"%(100 * max_err_arr[max_index], Om_pred[max_index]))

class CatalogInitializer:
    def __init__(self, z_max: float = 3.0, z_step: float = 0.01, polydist_file = None):
        self.z_bins = np.array(onp.arange(0., z_max + z_step, z_step))
        self.delta_z = (self.z_bins[1:] - self.z_bins[:-1])
        self.z_grid = 0.5 * (self.z_bins[1:] + self.z_bins[:-1])
        self.polydist_file = polydist_file
        
    def setup_boundaries(self, cosmo, slab_params):
        """Set up redshift boundaries based on slab parameters"""
        slab_definition = [slab_params.chi_min, slab_params.chi_max, slab_params.slab_width]
        self.z_boundaries = get_slabs_z_boundaries(cosmo, self.z_grid, slab_definition)
        self.z_slabs = 0.5 * (self.z_boundaries[1:] + self.z_boundaries[:-1])
        return self.z_slabs
    
    def load_polydist_file(self):
        with h5.File(self.polydist_file, 'r') as f:
            deg = f['deg'][()]
            Om_fid       = f['Om_fid'][()]
            polyfit_grid = f['polyfit_grid'][:]
            polyfit_slab = f['polyfit_slab'][:]
        return deg, Om_fid, polyfit_grid, polyfit_slab

    def save_polydist_file(self, deg, Om_fid, polyfit_grid, polyfit_slab):
        with h5.File(self.polydist_file, 'w') as f:
            f['deg'] = deg
            f['Om_fid']       = Om_fid
            f['polyfit_grid'] = polyfit_grid
            f['polyfit_slab'] = polyfit_slab

    def create_catalogs(self, observables) -> Catalogs:
        """Create source and lens catalogs from observable configurations"""
        nz_src_list = []
        nz_lens_list = []

        if self.polydist_file is not None and os.path.exists(self.polydist_file):
            print("Loading polyfit file...")
            deg, Om_fid, polyfit_grid, polyfit_slab = self.load_polydist_file()
        else:
            deg = 4
            Om_fid       = 0.27
            polyfit_grid = None
            polyfit_slab = None

        polydist_boundaries = PolyDist(self.z_boundaries, polyfit=polyfit_slab, Om_fid=Om_fid, deg=deg)
        polydist_grid       = PolyDist(self.z_grid, polyfit=polyfit_grid, Om_fid=Om_fid, deg=deg)
        
        if self.polydist_file is not None:
            if not os.path.exists(self.polydist_file):
                self.save_polydist_file(deg, Om_fid, polydist_grid.polyfit, polydist_boundaries.polyfit)

        polydists = [polydist_boundaries, polydist_grid]

        
        for tomobin in observables:
            obs_type = observables[tomobin]['type']
            nz_file = observables[tomobin]['nz_file']
            nz_data = np.load(nz_file)        
            nz = HistogramDist(self.z_boundaries, [self.z_bins, nz_data[1]], polydists)

            if obs_type == 'dens':
                nz_lens_list.append(nz)
            elif obs_type == 'kappa':
                nz_src_list.append(nz)
                                
        return Catalogs(
            nz_src_list=nz_src_list,
            nz_lens_list=nz_lens_list,
            N_SRC_BINS=len(nz_src_list),
            N_LENS_BINS=len(nz_lens_list)
        )

# catalog_init.py
import jax.numpy as np
import numpy as onp
from dataclasses import dataclass
from typing import List, Tuple
from .utils import get_slabs_z_boundaries
from .observables import HistogramDist 

@dataclass
class Catalogs:
    nz_src_list: List
    nz_lens_list: List
    N_SRC_BINS: int
    N_LENS_BINS: int

class CatalogInitializer:
    def __init__(self, z_max: float = 3.0, z_step: float = 0.01):
        self.z_bins = np.array(onp.arange(0., z_max + z_step, z_step))
        self.delta_z = (self.z_bins[1:] - self.z_bins[:-1])
        self.z_grid = 0.5 * (self.z_bins[1:] + self.z_bins[:-1])
        
    def setup_boundaries(self, cosmo, slab_params):
        """Set up redshift boundaries based on slab parameters"""
        slab_definition = [slab_params.chi_min, slab_params.chi_max, slab_params.slab_width]
        self.z_boundaries = get_slabs_z_boundaries(cosmo, self.z_grid, slab_definition)
        self.z_slabs = 0.5 * (self.z_boundaries[1:] + self.z_boundaries[:-1])
        return self.z_slabs
        
    def create_catalogs(self, observables) -> Catalogs:
        """Create source and lens catalogs from observable configurations"""
        nz_src_list = []
        nz_lens_list = []
        
        for tomobin in observables:
            obs_type = observables[tomobin]['type']
            nz_file = observables[tomobin]['nz_file']
            nz_data = np.load(nz_file)
            nz = HistogramDist(self.z_boundaries, [self.z_bins, nz_data[1]])
            
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

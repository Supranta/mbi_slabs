import jax.numpy as np
import jax_cosmo as jc
from ..utils import get_cosmo
from jax import vmap, jit
from .map_tools import MapTools

def powerlaw(k, A, alpha):
    return A * k**alpha

class Transform:
    def __init__(self, N_slabs, N_grid, L, Pk_slabs):
        self.map_tools = MapTools(N_grid, L)
        self.N_grid    = N_grid
        self.N_slabs   = N_slabs
        self.L         = L
        
        self.log_k_slabs  = np.log(Pk_slabs['k'])        
        self.log_Pk_slabs = np.log(Pk_slabs['Pk'])        
        self.mu           = Pk_slabs['mu']

        self.k_mask = np.ones_like(self.map_tools.ell, dtype=bool)
        self.k_mask = self.k_mask.at[0, 0].set(False)

        self.log_k_eval = np.log(self.map_tools.ell[self.k_mask])
        
        self.fourier2map_slabs = jit(vmap(self.map_tools.fourier2map))

        self.set_Pk_arr()

    def set_Pk_arr(self):
        Pk_arr = np.zeros((self.N_slabs, 2, self.N_grid, self.N_grid//2 + 1))

        for i in range(self.N_slabs):
            pk_eval = np.exp(jc.scipy.interpolate.interp(self.log_k_eval, self.log_k_slabs, self.log_Pk_slabs[i]))
            Pk_arr = Pk_arr.at[i,:,self.k_mask].add(pk_eval[:,np.newaxis])
        
        Pk_arr = Pk_arr.at[:,:,0,0].set(1e-20)

        Pk_arr.at[:,0,0,0].multiply(2.) 
        Pk_arr.at[:,0,0,-1].multiply(2.) 
        Pk_arr.at[:,0,self.N_grid//2,0].multiply(2.) 
        Pk_arr.at[:,0,self.N_grid//2,-1].multiply(2.) 

        Pk_arr.at[:,1,0,0].set(1e-20) 
        Pk_arr.at[:,1,0,-1].set(1e-20) 
        Pk_arr.at[:,1,self.N_grid//2,0].set(1e-20) 
        Pk_arr.at[:,1,self.N_grid//2,-1].set(1e-20) 

        self.Pk_arr = Pk_arr * self.map_tools.Omega_s

class GaussianTransform(Transform):
    def __init__(self, N_slabs, N_grid, L, Pk_slabs):
        super().__init__(N_slabs, N_grid, L, Pk_slabs)

        
    def x2delta(self, x_l, A_cosmo):
        delta_l = x_l * np.sqrt(A_cosmo * self.Pk_arr)
        delta_map = self.fourier2map_slabs(delta_l)
        return delta_map

class LogNormalTransform(Transform):
    def __init__(self, N_slabs, N_grid, L, Pk_slabs):
        super().__init__(N_slabs, N_grid, L, Pk_slabs)
        self.mu = self.mu[:,np.newaxis,np.newaxis]

    def x2delta(self, x_l, A_cosmo):
        y_l = x_l * np.sqrt(A_cosmo * self.Pk_arr)
        y_map = self.fourier2map_slabs(y_l)
        delta_map = np.exp(y_map + A_cosmo * self.mu) - 1.
        return delta_map


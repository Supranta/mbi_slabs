import jax.numpy as np
import jax_cosmo as jc
from ..utils import get_cosmo
from jax import vmap, jit
from .map_tools import MapTools

def powerlaw(k, A, alpha):
    return A * k**alpha

class Transform:
    def __init__(self, N_slabs, N_grid, L, z_slabs):
        self.map_tools = MapTools(N_grid, L)
        self.N_grid    = N_grid
        self.N_slabs   = N_slabs
        self.L         = L
        
        cosmo = get_cosmo(0.3)        
        self.Dz_slabs = jc.background.growth_factor(cosmo, 1. / (1. + z_slabs))

        self.fourier2map_slabs = jit(vmap(self.map_tools.fourier2map))

    def set_Pk_arr(self, A, alpha):
        Pk = 0.5 * powerlaw(self.map_tools.ell, A, alpha)

        Pk_arr = np.zeros((self.N_slabs, 2, self.N_grid, self.N_grid//2 + 1))
        Pk_arr = np.tile(Pk, (self.N_slabs,2,1,1)) * self.Pk_scaling
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
    def __init__(self, N_slabs, N_grid, L, z_slabs):
        super().__init__(N_slabs, N_grid, L, z_slabs)
        self.Pk_scaling = (self.Dz_slabs**2)[:,np.newaxis,np.newaxis,np.newaxis]

        A_fid, alpha_fid = 4., -1.
        self.set_Pk_arr(A_fid, alpha_fid)

    def x2delta(self, x_l, A_cosmo):
        delta_l = x_l * np.sqrt(A_cosmo * self.Pk_arr)
        delta_map = self.fourier2map_slabs(delta_l)
        return delta_map

class LogNormalTransform(Transform):
    def __init__(self, N_slabs, N_grid, L, z_slabs):
        super().__init__(N_slabs, N_grid, L, z_slabs)
        A_fid = 2.7
        mu0 = -(0.5 * 1.07)**2 * (A_fid / 2.)
        self.mu = -0.5 * np.log(1. + self.Dz_slabs**2 * (np.exp(-2. * mu0) - 1.))[:,np.newaxis,np.newaxis]
        self.Pk_scaling = (self.mu / mu0)[:,:,:,np.newaxis]
        self.set_Pk_arr(A_fid, -1.)

    def x2delta(self, x_l, A_cosmo):
        y_l = x_l * np.sqrt(A_cosmo * self.Pk_arr)
        y_map = self.fourier2map_slabs(y_l)
        delta_map = np.exp(y_map + A_cosmo * self.mu) - 1.
        return delta_map


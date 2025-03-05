import jax.numpy as np
import jax_cosmo as jc
from jax import vmap, jit
from .map_tools import MapTools
from .emulator import PkEmulator

def powerlaw(k, A, alpha):
    return A * k**alpha

class Transform:
    def __init__(self, N_slabs, N_grid, L, pk_emu_file):
        self.map_tools = MapTools(N_grid, L)
        self.N_grid    = N_grid
        self.N_slabs   = N_slabs
        self.L         = L       

        self.k_mask = np.ones_like(self.map_tools.ell, dtype=bool)
        self.k_mask = self.k_mask.at[0, 0].set(False)

        self.log_k_eval = np.log(self.map_tools.ell[self.k_mask])
        
        self.fourier2map_slabs = jit(vmap(self.map_tools.fourier2map))

    def get_Pk_arr(self, theta_cosmo):
        Pk_arr = np.zeros((self.N_slabs, 2, self.N_grid, self.N_grid//2 + 1))

        log_pk = np.log(self.pk_emu.predict_pk(theta_cosmo)[0])
        for i in range(self.N_slabs):
            pk_eval = np.exp(jc.scipy.interpolate.interp(self.log_k_eval, self.log_k_slabs, log_pk[i]))
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

        return Pk_arr * self.map_tools.Omega_s

class GaussianTransform(Transform):
    def __init__(self, N_slabs, N_grid, L, pk_emu_file):
        super().__init__(N_slabs, N_grid, L, pk_emu_file)
        self.pk_emu = PkEmulator(pk_emu_file, False)
        self.log_k_slabs  = np.log(self.pk_emu.k) 
    
    def x2delta(self, x_l, theta_cosmo):
        Pk_arr    = self.get_Pk_arr(theta_cosmo)
        delta_l   = x_l * np.sqrt(Pk_arr)
        delta_map = self.fourier2map_slabs(delta_l)
        return delta_map

class LogNormalTransform(Transform):
    def __init__(self, N_slabs, N_grid, L, pk_emu_file):
        super().__init__(N_slabs, N_grid, L, pk_emu_file)
        self.pk_emu = PkEmulator(pk_emu_file, True)
        self.log_k_slabs  = np.log(self.pk_emu.k) 

    def x2delta(self, x_l, theta_cosmo):
        Pk_arr = self.get_Pk_arr(theta_cosmo)
        y_l    = x_l * np.sqrt(Pk_arr)
        y_map  = self.fourier2map_slabs(y_l)
        y_mean = self.pk_emu.get_y_mean(theta_cosmo)[:,np.newaxis,np.newaxis]
        delta_map = np.exp(y_map + y_mean) - 1.
        return delta_map


import jax.numpy as np
import jax_cosmo as jc
from jax import vmap, jit
from .map_tools import MapTools
from .emulator import ClEmulator

def powerlaw(k, A, alpha):
    return A * k**alpha

class Transform:
    def __init__(self, N_slabs, N_grid, theta_max, pk_emu_file):
        self.map_tools = MapTools(N_grid, theta_max)
        self.N_grid    = N_grid
        self.N_slabs   = N_slabs    

        self.ell_mask = np.ones_like(self.map_tools.ell, dtype=bool)
        self.ell_mask = self.ell_mask.at[0, 0].set(False)

        self.log_ell_eval = np.log(self.map_tools.ell[self.ell_mask])
        
        self.fourier2map_slabs = jit(vmap(self.map_tools.fourier2map))

    def get_cl_arr(self, theta_cosmo):
        cl_arr = np.zeros((self.N_slabs, 2, self.N_grid, self.N_grid//2 + 1))

        log_cl = np.log(self.cl_emu.predict_cl(theta_cosmo)[0])
        for i in range(self.N_slabs):
            cl_eval = np.exp(jc.scipy.interpolate.interp(self.log_ell_eval, self.log_ell_slabs, log_cl[i]))
            cl_arr = cl_arr.at[i,:,self.ell_mask].add(cl_eval[:,np.newaxis])
        
        cl_arr = cl_arr.at[:,:,0,0].set(1e-20)

        cl_arr = cl_arr.at[:,0,0,0].multiply(2.) 
        cl_arr = cl_arr.at[:,0,0,-1].multiply(2.) 
        cl_arr = cl_arr.at[:,0,self.N_grid//2,0].multiply(2.) 
        cl_arr = cl_arr.at[:,0,self.N_grid//2,-1].multiply(2.) 

        cl_arr = cl_arr.at[:,1,0,0].set(1e-20) 
        cl_arr = cl_arr.at[:,1,0,-1].set(1e-20) 
        cl_arr = cl_arr.at[:,1,self.N_grid//2,0].set(1e-20) 
        cl_arr = cl_arr.at[:,1,self.N_grid//2,-1].set(1e-20) 

        return cl_arr * self.map_tools.Omega_s

class GaussianTransform(Transform):
    def __init__(self, N_slabs, N_grid, theta_max, cl_emu_file):
        super().__init__(N_slabs, N_grid, theta_max, cl_emu_file)
        self.cl_emu = ClEmulator(cl_emu_file, False)
        self.log_ell_slabs  = np.log(self.cl_emu.ell) 
    
    def x2delta(self, x_l, theta_cosmo):
        cl_arr    = self.get_cl_arr(theta_cosmo)
        delta_l   = x_l * np.sqrt(cl_arr)
        delta_map = self.fourier2map_slabs(delta_l)
        return delta_map

class LogNormalTransform(Transform):
    def __init__(self, N_slabs, N_grid, theta_max, cl_emu_file):
        super().__init__(N_slabs, N_grid, theta_max, cl_emu_file)
        self.cl_emu = ClEmulator(cl_emu_file, True)
        self.log_ell_slabs  = np.log(self.cl_emu.ell) 

    def x2delta(self, x_l, theta_cosmo):
        cl_arr = self.get_cl_arr(theta_cosmo)
        y_l    = x_l * np.sqrt(cl_arr)
        y_map  = self.fourier2map_slabs(y_l)
        y_mean = self.cl_emu.get_y_mean(theta_cosmo)[:,np.newaxis,np.newaxis]
        delta_map = np.exp(y_map + y_mean) - 1.
        return delta_map


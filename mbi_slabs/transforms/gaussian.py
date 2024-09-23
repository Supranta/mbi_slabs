import jax.numpy as np
from .map_tools import MapTools

def powerlaw(k, A, alpha):
    return A * k**alpha

class GaussianTransform:
    def __init__(self, N_slabs, N_grid, L):
        self.map_tools = MapTools(N_grid, L)
        self.N_grid    = N_grid
        self.N_slabs   = N_slabs
        self.L         = L
        A_fid, alpha_fid = 4., -1.
        self.set_Pk_arr(A_fid, alpha_fid)
        
    def x2G(self, x_l):
        y_l = x_l * np.sqrt(self.Pk_arr)
        y_map = np.zeros((self.N_slabs, self.N_grid, self.N_grid))
        for i in range(self.N_slabs):
            y_map_i = self.map_tools.fourier2map(y_l[i])
            y_map   = y_map.at[i].set(y_map_i)
        return y_map

    def set_Pk_arr(self, A, alpha):
        Pk = 0.5 * powerlaw(self.map_tools.ell, A, alpha)

        Pk_arr = np.zeros((self.N_slabs, 2, self.N_grid, self.N_grid//2 + 1))
        Pk_arr = np.tile(Pk, (self.N_slabs,2,1,1)) 
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
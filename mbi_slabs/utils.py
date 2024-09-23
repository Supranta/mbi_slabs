import jax_cosmo as jc
import jax.numpy as np
import numpy as onp

def get_cosmo(Omega_m):
    Omega_b, h, ns, sigma8, w0, wa = 0.05, 1., 0.95, 0.80, -1., 0.
    Omega_c = Omega_m - Omega_b
    return jc.Cosmology(Omega_c, Omega_b, h, ns, sigma8, 0., w0, wa)


def get_slabs_z_boundaries(cosmo_fid, z_grid, slab_definition):
    """
    Get the redshift boundaries of the density slabs from comoving distance definitions
    """
    CHI_MIN, CHI_MAX, DELTA_CHI = slab_definition
    chi_grid       = jc.background.radial_comoving_distance(cosmo_fid, 1. / (1. + z_grid))
    chi_boundaries = np.arange(CHI_MIN, CHI_MAX + 0.00001, DELTA_CHI)
    z_boundaries   = np.interp(chi_boundaries, chi_grid, z_grid)
    return z_boundaries

def get_all_slabs(N_slabs=24, randomize=True, Gaussian_sim=False):
    if Gaussian_sim:
        filename = '/spiff/ssarnabo/N_body_sims/MDR1/dens_512_gaussian.npy'
    else:
        filename = '/spiff/ssarnabo/N_body_sims/MDR1/dens_512_mdr1.npy'
    dens_3d = onp.load(filename)
    dens_2d_slabs = []
    for i in range(8):
        ind_start = 64 * i
        ind_end   = 64 * (i + 1)
        dens_2d_x = dens_3d[ind_start:ind_end,:,:].mean(0)        
        dens_2d_y = dens_3d[:,ind_start:ind_end,:].mean(1)
        dens_2d_z = dens_3d[:,:,ind_start:ind_end].mean(2)
        dens_2d_slabs.append(dens_2d_x)
        dens_2d_slabs.append(dens_2d_y)
        dens_2d_slabs.append(dens_2d_z)
    if not randomize:
        return np.array(dens_2d_slabs)
    randomized_idx  = onp.random.choice(np.arange(24), size=24, replace=False).tolist()
    if(N_slabs>24):
        randomized_idx1 = onp.random.choice(np.arange(24), size=24, replace=False).tolist()
        randomized_idx = randomized_idx + randomized_idx1
    return np.array(onp.array(dens_2d_slabs)[randomized_idx][:N_slabs])

class FourierTransforms:
    def __init__(self, N_grid):
        self.N_grid = N_grid
        
        self.lx = 2*np.pi*np.fft.fftfreq(self.N_grid)
        self.ly = 2*np.pi*np.fft.fftfreq(self.N_grid)

        self.ell_x = np.tile(self.lx[:, None], (1, self.N_grid))       
        self.ell_y = np.tile(self.ly[None, :], (self.N_grid, 1))

        self.ell_sq = self.ell_x**2 + self.ell_y**2
        self.ell_sq = self.ell_sq.at[0,0].set(1.)
        
        self.KS_operator1 = ((self.ell_x**2 - self.ell_y**2)/ self.ell_sq)
        self.KS_operator2 = (2. * self.ell_x * self.ell_y   / self.ell_sq)
        
    def kappa2gamma(self, kappa):
        kappa_l = np.fft.fftn(kappa)
    
        gamma_1_l = self.KS_operator1 * kappa_l 
        gamma_2_l = self.KS_operator2 * kappa_l 
        
        gamma_1 =  np.fft.ifftn(gamma_1_l).real
        gamma_2 =  np.fft.ifftn(gamma_2_l).real
    
        return gamma_1, gamma_2
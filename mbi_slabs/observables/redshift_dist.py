import jax.numpy as np
import jax_cosmo as jc
import numpy as onp

def get_cosmo(theta_cosmo):
    Omega_m, sigma8 = theta_cosmo
    Omega_b, h, ns, w0, wa = 0.05, 1., 0.97, -1., 0.
    Omega_c = Omega_m - Omega_b
    return jc.Cosmology(Omega_c, Omega_b, h, ns, sigma8, 0., w0, wa, gamma=0.55)

class RedshiftDist:
    def __init__(self, z_boundaries, nz_data):
        z_bins, nz_bin = nz_data
        
        self.nz_bin  = nz_bin / np.sum(nz_bin)
        
        self.z_boundaries = z_boundaries
        self.z_bins       = z_bins

    def get_mixing_matrix(self, Delta_z):
        pass
    
    def get_slab_weights_proj_density(self, Delta_z):
        M = self.get_mixing_matrix_proj_density(Delta_z)
        weights = np.sum((self.nz_bin * M), axis=1)
        return np.expand_dims(np.expand_dims(weights, -1), -1)
    
    def get_slab_weights_kappa(self, Omega_m, Delta_z):
        M = self.get_kappa_mixing_matrix(Omega_m, Delta_z)
        weights = np.sum((self.nz_bin * M), axis=1)
        return np.expand_dims(np.expand_dims(weights, -1), -1)
    
class HistogramDist(RedshiftDist):
    def __init__(self, z_boundaries, nz_data, polydist):
        super().__init__(z_boundaries, nz_data)
        self.polydist = polydist
        self.z_plus  = np.expand_dims(self.z_bins[1:], 0)
        self.z_minus = np.expand_dims(self.z_bins[:-1], 0)        
        self.z_diff = (self.z_bins[1:] - self.z_bins[:-1])
        self.z_grid = 0.5 * (self.z_bins[1:] + self.z_bins[:-1])
        
        self.polydist_boundaries = polydist[0]
        self.polydist_grid       = polydist[1]
 
        self.z_boundaries_max = np.expand_dims(self.z_boundaries[1:], 1)
        self.z_boundaries_min = np.expand_dims(self.z_boundaries[:-1], 1)
       
    def get_mixing_matrix_proj_density(self, Delta_z):
        term1 = np.minimum(self.z_plus + Delta_z,  self.z_boundaries_max)
        term2 = np.maximum(self.z_minus + Delta_z, self.z_boundaries_min)
        x     = (term1 - term2) / self.z_diff
        return np.clip(x, 0.)

    def get_linear_z_coefficients(self, chi_min, chi_max):
        Delta_z   = (self.z_boundaries_max - self.z_boundaries_min)
        Delta_chi = (chi_max - chi_min)
        linear_z_slope = Delta_z / Delta_chi
        linear_z_intercept = 1. + self.z_boundaries_min - Delta_z / Delta_chi * chi_min
        return linear_z_intercept, linear_z_slope

    def _kappa_mixing_matrix(self, linear_z_intercept, linear_z_slope, chi_lim_lo, chi_lim_hi, chi_grid):
        term1 = 0.5 * linear_z_intercept * (chi_lim_hi**2 - chi_lim_lo**2)
        term_21 = linear_z_slope * (chi_lim_hi**3 - chi_lim_lo**3) / 3.
        term_22 = linear_z_intercept * (chi_lim_hi**3 - chi_lim_lo**3) / 3. * np.expand_dims(1. / chi_grid, 0)
        term2 = term_21 - term_22
        term3 = -0.25 * linear_z_slope * (chi_lim_hi**4 - chi_lim_lo**4) * np.expand_dims(1. / chi_grid, 0)
        return np.clip(term1 + term2 + term3, 0.)

    def get_kappa_mixing_matrix(self, Omega_m, Delta_z=0.):
        prefactor   = 1.5 * Omega_m / jc.constants.rh / jc.constants.rh
   
        chi_boundaries = np.expand_dims(self.polydist_boundaries.pred_distance(Omega_m), 1)
        chi_max = chi_boundaries[1:]
        chi_min = chi_boundaries[:-1]
        chi_grid = self.polydist_grid.pred_distance(Omega_m) + self.get_taylor_correction_z(Omega_m, Delta_z) 

        linear_z_intercept, linear_z_slope = self.get_linear_z_coefficients(chi_min, chi_max)
    
        chi_lim_hi  = np.minimum(chi_max, np.expand_dims(chi_grid, 0))
        chi_lim_lo1 = np.maximum(chi_min, np.expand_dims(chi_grid, 0))
        chi_lim_lo2 = chi_min
    
        term1 = self._kappa_mixing_matrix(linear_z_intercept, linear_z_slope, chi_lim_lo1, chi_lim_hi, chi_grid)
        term2 = self._kappa_mixing_matrix(linear_z_intercept, linear_z_slope, chi_lim_lo2, chi_lim_hi, chi_grid)
    
        return prefactor * (term2 - term1)

    def get_taylor_correction_z(self, Omega_m, Delta_z):
        E2 = Esqr(Omega_m, self.z_grid)
        Delta_chi_1 = jc.constants.rh / np.sqrt(Esqr(Omega_m, self.z_grid)) * Delta_z
        second_order_correction = 3. * Omega_m  * Delta_z / 4. / E2 / (1. + self.z_grid)**4
        return Delta_chi_1 * (1. + second_order_correction)

def Esqr(Omega_m, z):
    return 1. - Omega_m * (1. - 1./(1. + z)**3)

class ObservableCalculator:
    def __init__(self, z_slabs, C_cr=0.013877, z0=0.62):
        self.z_slabs = z_slabs
        self.C_cr = C_cr
        self.z0 = z0
        cosmo_fid = get_cosmo([0.27, 0.82])
        self.growth_factor_slabs = jc.background.growth_factor(cosmo_fid, 1. / (1. + self.z_slabs))

    def A2C(self, Omega_m, A1, eta=0.):
        """
        Calculate C1 parameter according to Eqn 15 of 1811.06989
        """
        C1 = -A1 * self.C_cr * Omega_m * ((1. + self.z_slabs)/(1. + self.z0))**eta
        return C1 / self.growth_factor_slabs
    
    def get_kappa(self, nz, cosmo, Delta_z, dens_slabs):
        """Calculate kappa (lensing convergence) for given density slabs"""
        weights = nz.get_slab_weights_kappa(cosmo, Delta_z)
        return np.sum((weights * dens_slabs), axis=0)
    
    def get_proj_density(self, nz, Delta_z, dens_slabs):
        """Calculate projected density for given density slabs"""
        weights = nz.get_slab_weights_proj_density(Delta_z)
        return np.sum((weights * dens_slabs), axis=0)
    
    def get_kappa_ia(self, nz, Omega_m, Delta_z, A1, eta, dens_slabs):
        """Calculate intrinsic alignment contribution to kappa"""
        C_ia                = self.A2C(Omega_m, A1, eta)
        weights             = C_ia[:,np.newaxis,np.newaxis] * nz.get_slab_weights_proj_density(Delta_z)
        return np.sum((weights * dens_slabs), axis=0)

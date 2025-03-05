import jax.numpy as np
import jax_cosmo as jc

def get_cosmo(theta_cosmo):
    Omega_m, sigma8 = theta_cosmo
    Omega_b, h, ns, w0, wa = 0.05, 0.7, 0.97, -1., 0.
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
    
    def get_slab_weights_kappa(self, cosmo, Delta_z):
        M = self.get_kappa_mixing_matrix(cosmo, Delta_z)
        weights = np.sum((self.nz_bin * M), axis=1)
        return np.expand_dims(np.expand_dims(weights, -1), -1)
    
class HistogramDist(RedshiftDist):
    def __init__(self, z_boundaries, nz_data):
        super().__init__(z_boundaries, nz_data)
        
        self.z_boundaries_max = np.expand_dims(self.z_boundaries[1:], 1)
        self.z_boundaries_min = np.expand_dims(self.z_boundaries[:-1], 1)
        self.z_plus  = np.expand_dims(self.z_bins[1:], 0)
        self.z_minus = np.expand_dims(self.z_bins[:-1], 0)        
        self.z_diff = (self.z_bins[1:] - self.z_bins[:-1])
        self.z_grid = 0.5 * (self.z_bins[1:] + self.z_bins[:-1])

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

    def get_kappa_mixing_matrix(self, cosmo, Delta_z=0.):
        prefactor   = 1.5 * cosmo.Omega_m / jc.constants.rh / jc.constants.rh
    
        chi_max = jc.background.radial_comoving_distance(cosmo, 1. / (1. + self.z_boundaries_max))
        chi_min = jc.background.radial_comoving_distance(cosmo, 1. / (1. + self.z_boundaries_min))
        chi_grid = jc.background.radial_comoving_distance(cosmo, 1. / (1. + self.z_grid + Delta_z)) + 1e-15
    
        linear_z_intercept, linear_z_slope = self.get_linear_z_coefficients(chi_min, chi_max)
    
        chi_lim_hi  = np.minimum(chi_max, np.expand_dims(chi_grid, 0))
        chi_lim_lo1 = np.maximum(chi_min, np.expand_dims(chi_grid, 0))
        chi_lim_lo2 = chi_min
    
        term1 = self._kappa_mixing_matrix(linear_z_intercept, linear_z_slope, chi_lim_lo1, chi_lim_hi, chi_grid)
        term2 = self._kappa_mixing_matrix(linear_z_intercept, linear_z_slope, chi_lim_lo2, chi_lim_hi, chi_grid)
    
        return prefactor * (term2 - term1)
    
class ObservableCalculator:
    def __init__(self, z_slabs, C_cr=0.013877, z0=0.62):
        self.z_slabs = z_slabs
        self.C_cr = C_cr
        self.z0 = z0
    
    def A2C(self, A1, eta=0., OmegaM=0.3):
        """
        Calculate C1 parameter according to Eqn 15 of 1811.06989
        """
        C1 = -A1 * self.C_cr * OmegaM * ((1. + self.z_slabs)/(1. + self.z0))**eta
        return C1
    
    def get_kappa(self, nz, cosmo, Delta_z, dens_slabs):
        """Calculate kappa (lensing convergence) for given density slabs"""
        weights = nz.get_slab_weights_kappa(cosmo, Delta_z)
        return np.sum((weights * dens_slabs), axis=0)
    
    def get_proj_density(self, nz, Delta_z, dens_slabs):
        """Calculate projected density for given density slabs"""
        weights = nz.get_slab_weights_proj_density(Delta_z)
        return np.sum((weights * dens_slabs), axis=0)
    
    def get_kappa_ia(self, nz, cosmo, Delta_z, A1, eta, dens_slabs):
        """Calculate intrinsic alignment contribution to kappa"""
        Dz      = jc.background.growth_factor(cosmo, 1. / (1. + self.z_slabs))
        C_ia    = self.A2C(A1, eta, cosmo.Omega_m) / Dz
        weights = C_ia[:,np.newaxis,np.newaxis] * nz.get_slab_weights_proj_density(Delta_z)
        return np.sum((weights * dens_slabs), axis=0)

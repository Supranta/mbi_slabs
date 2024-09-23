import jax
import jax.numpy as np
import numpy as onp
import h5py as h5
import jax_cosmo as jc
import matplotlib.pyplot as plt
from tqdm import trange
import yaml
import sys

configfile = sys.argv[1]

from mbi_slabs.utils import *

jax.config.update("jax_enable_x64", True)

cosmo = get_cosmo(0.3)

from mbi_slabs.observables import HistogramDist
from mbi_slabs.transforms import GaussianTransform

with open(configfile, "r") as stream:
    config = yaml.safe_load(stream)
    
z_bins  = np.arange(0., 3.000000001, 0.01)
delta_z = (z_bins[1:] - z_bins[:-1])
z_grid = 0.5 * (z_bins[1:] + z_bins[:-1])

##############################################################
########## To-do: Define the slab boundaries in the config
##############################################################


CHI_MIN    = config['slabs']['chi_min']
CHI_MAX    = config['slabs']['chi_max']
SLAB_WIDTH = config['slabs']['slab_width']
N_grid     = config['slabs']['N_grid']
L          = config['slabs']['L']

slab_definition = [CHI_MIN, CHI_MAX, SLAB_WIDTH]

z_boundaries    = get_slabs_z_boundaries(cosmo, z_grid, slab_definition)
z_slabs         = 0.5 * (z_boundaries[1:] + z_boundaries[:-1])

dens_slabs = get_all_slabs(randomize=False, Gaussian_sim=True)[:,:N_grid,:N_grid]

from mbi_slabs.transforms import MapTools

map_tools = MapTools(N_grid, L)

N_ELL = 16

ELL_MIN = np.sort(map_tools.ell.flatten())[1]
ELL_MAX = np.max(map_tools.ell.flatten())

ell_bins = np.logspace(np.log10(ELL_MIN), np.log10(ELL_MAX), N_ELL)

N_slabs = z_slabs.shape[0]

transform = GaussianTransform(N_slabs, N_grid, L)

x_l             = np.array(onp.random.normal(size=(N_slabs, 2, N_grid, N_grid//2 + 1))) 
dens_slabs_true = transform.x2G(x_l)
    
nz_src_list = []
nz_lens_list = []

for tomobin in config['observables']:
    obs_type = config['observables'][tomobin]['type']
    nz_file  = config['observables'][tomobin]['nz_file']
    nz_data  = np.load(nz_file)
    nz       = HistogramDist(z_boundaries, [z_bins, nz_data[1]])
    if(obs_type=='dens'):
        nz_lens_list.append(nz)
    elif(obs_type=='kappa'):
        nz_src_list.append(nz)
        
N_LENS_BINS = len(nz_lens_list)
N_SRC_BINS  = len(nz_src_list) 

n_warmup  = config['sampling']['mcmc']['n_warmup']
n_samples = config['sampling']['mcmc']['n_samples']

output_dir = config['io']['output_dir']

with h5.File(output_dir + '/truth.h5', 'w') as f:
    f['delta_slabs'] = dens_slabs_true
        
C_cr = 0.013877
z0   = 0.62

def A2C(A1, eta=0., OmegaM=0.3):
    """
    See Eqn 15 of 1811.06989
    """
    C1   = -A1 * C_cr * OmegaM * ((1. + z_slabs)/(1. + z0))**eta
    return C1

def get_kappa(nz, Omega_m, Delta_z, dens_slabs):
    weights = nz.get_slab_weights_kappa(Omega_m, Delta_z)
    return np.sum((weights * dens_slabs), axis=0)

def get_proj_density(nz, Delta_z, dens_slabs):
    weights = nz.get_slab_weights_proj_density(Delta_z)
    return np.sum((weights * dens_slabs), axis=0)

def get_kappa_ia(nz, Omega_m, Delta_z, A1, eta, dens_slabs):
    C_ia = A2C(A1, eta, Omega_m)
    weights = C_ia[:,np.newaxis,np.newaxis] * nz.get_slab_weights_proj_density(Delta_z)
    return np.sum((weights * dens_slabs), axis=0)

weights_src  = [nz.get_slab_weights_kappa(0.3, 0.) for nz in nz_src_list]
weights_lens = [nz.get_slab_weights_proj_density(0.) for nz in nz_lens_list]

def get_proj_field(weights, dens_slabs):
    return np.sum((weights * dens_slabs), axis=0)

A_ia_fid = 0.5

kappa_list        = [get_kappa(nz_src_list[i], cosmo.Omega_m, 0., dens_slabs_true) for i in range(N_SRC_BINS)]
kappa_ia_list     = [get_kappa_ia(nz_src_list[i], cosmo.Omega_m, 0., A_ia_fid, 0., dens_slabs_true) for i in range(N_SRC_BINS)]
proj_density_list = [get_proj_density(nz_lens_list[i], 0., dens_slabs_true) for i in range(N_LENS_BINS)]

F = FourierTransforms(N_grid)

sigma_noise = 0.05

shape_data = []

for kappa, kappa_ia in zip(kappa_list, kappa_ia_list):
    gamma_1,    gamma_2    = F.kappa2gamma(kappa)
    gamma_ia_1, gamma_ia_2 = F.kappa2gamma(kappa_ia)

    e1_obs = gamma_1 + gamma_ia_1 + np.array(sigma_noise * onp.random.normal(size=kappa.shape))
    e2_obs = gamma_2 + gamma_ia_2 + np.array(sigma_noise * onp.random.normal(size=kappa.shape))
    
    shape_data.append(np.array([e1_obs, e2_obs]))
    
    
l = (L / N_grid)
nbar        = 10e-4 * l**2 * SLAB_WIDTH
N_gals_data = []

for proj_density in proj_density_list:
    mu = nbar * (1. + proj_density)
    N_gals = onp.random.poisson(np.clip(mu,1e-3))
    N_gals_data.append(np.array(N_gals))
    
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

key = jax.random.PRNGKey(onp.random.randint(1000000))
rng_key, rng_key_ = jax.random.split(key)

def density_slab_model(nz_src_list, nz_lens_list, dens_slabs):
    Omega_m = 0.3
    x_l     = numpyro.sample("x_l", dist.Normal(np.zeros((N_slabs,2,N_grid,N_grid//2 + 1)), np.ones((N_slabs,2,N_grid,N_grid//2 + 1))), rng_key=key)
    dens_slabs = transform.x2G(x_l)
    Dz_src = numpyro.sample("Dz_src", dist.Normal(np.zeros(N_SRC_BINS), 0.01 * np.ones(N_SRC_BINS)), rng_key=key)        
    m      = numpyro.sample("m", dist.Normal(np.zeros(N_SRC_BINS), 0.01 * np.ones(N_SRC_BINS)), rng_key=key)
    A_ia   = numpyro.sample("A_ia",   dist.Uniform(-5., 5.), rng_key=key)
    eta_ia = numpyro.sample("eta_ia", dist.Uniform(-5., 5.), rng_key=key)
        
    for i in range(N_SRC_BINS):
        kappa = get_kappa(nz_src_list[i], Omega_m, Dz_src[i], dens_slabs)
        kappa_IA = get_kappa_ia(nz_src_list[i], Omega_m, Dz_src[i], A_ia, eta_ia, dens_slabs)
        gamma_1,    gamma_2 = F.kappa2gamma(kappa)
        gamma_IA_1, gamma_IA_2 = F.kappa2gamma(kappa_IA)
        numpyro.sample('e1_obs_%d'%(i+1), dist.Normal((1. + m[i]) * (gamma_1 + gamma_IA_1), sigma_noise), obs=shape_data[i][0])
        numpyro.sample('e2_obs_%d'%(i+1), dist.Normal((1. + m[i]) * (gamma_2 + gamma_IA_2), sigma_noise), obs=shape_data[i][1])

    Dz_lens = numpyro.sample("Dz_lens", dist.Normal(np.zeros(N_LENS_BINS), 0.01 * np.ones(N_LENS_BINS)), rng_key=key)
    bg      = numpyro.sample("bg", dist.Normal(np.ones(N_LENS_BINS), 0.1 * np.ones(N_LENS_BINS)), rng_key=key)

    for i in range(N_LENS_BINS):
        proj_density = get_proj_density(nz_lens_list[i], Dz_lens[i], dens_slabs)
        mu = np.clip(nbar * (1. + bg[i] * proj_density), 1e-3) 
        numpyro.sample('Ng_%d'%(i+1), dist.Poisson(mu), obs=N_gals_data[i])
        
kernel = NUTS(density_slab_model, target_accept_prob=0.65)
mcmc   = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples)

mcmc.run(rng_key_, nz_src_list, nz_lens_list, dens_slabs)

    
samples = mcmc.get_samples()

x_l             = samples["x_l"]
bg_samples      = samples["bg"]
m_samples       = samples["m"]
Dz_src_samples  = samples["Dz_src"]
Dz_lens_samples = samples["Dz_lens"]
A_ia_samples    = samples["A_ia"]
eta_ia_samples  = samples["eta_ia"]

for i in trange(n_samples):
    dens_slabs_sample = transform.x2G(x_l[i])    
    with h5.File(output_dir + '/mcmc_%d.h5'%(i), 'w') as f:
        f['slab_dens'] = dens_slabs_sample    
        f['bg']        = bg_samples[i]
        f['m']         = m_samples[i]
        f['Dz_src']    = Dz_src_samples[i]
        f['Dz_lens']   = Dz_lens_samples[i]
        f['A_ia']      = A_ia_samples[i]
        f['eta_ia']    = eta_ia_samples[i]
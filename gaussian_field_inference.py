import jax
import jax.numpy as np
from jax import jit, vmap
import numpy as onp
import h5py as h5
import jax_cosmo as jc
from tqdm import trange
import yaml
import sys

configfile = sys.argv[1]

from mbi_slabs.utils import *
from mbi_slabs import EnvironmentSetup, ConfigLoader, DataGenerator 

EnvironmentSetup.setup_jax_env()

cosmo = get_cosmo(0.3)

from mbi_slabs.observables import *
from mbi_slabs.transforms import GaussianTransform

config = ConfigLoader(configfile)

z_bins  = np.array(onp.arange(0., 3.000000001, 0.01))
delta_z = (z_bins[1:] - z_bins[:-1])
z_grid = 0.5 * (z_bins[1:] + z_bins[:-1])

slab_params     = config.get_slab_config()
observables     = config.get_observables()
sampling_params = config.get_sampling_config()
output_dir      = config.get_output_dir()

slab_definition = [slab_params.chi_min, slab_params.chi_max, slab_params.slab_width]

z_boundaries    = get_slabs_z_boundaries(cosmo, z_grid, slab_definition)
z_slabs         = 0.5 * (z_boundaries[1:] + z_boundaries[:-1])

dens_slabs = get_all_slabs(randomize=False, Gaussian_sim=True)[:,:slab_params.N_grid,:slab_params.N_grid]

from mbi_slabs.transforms import MapTools

map_tools = MapTools(slab_params.N_grid, slab_params.L)

N_slabs = z_slabs.shape[0]

transform = GaussianTransform(N_slabs, slab_params.N_grid, slab_params.L)

x_l             = np.array(onp.random.normal(size=(N_slabs, 2, slab_params.N_grid, slab_params.N_grid//2 + 1))) 
dens_slabs_true = transform.x2G(x_l)
    
obs_calc = ObservableCalculator(z_slabs)

nz_src_list = []
nz_lens_list = []

for tomobin in observables:
    obs_type = observables[tomobin]['type']
    nz_file  = observables[tomobin]['nz_file']
    nz_data  = np.load(nz_file)
    nz       = HistogramDist(z_boundaries, [z_bins, nz_data[1]])
    if(obs_type=='dens'):
        nz_lens_list.append(nz)
    elif(obs_type=='kappa'):
        nz_src_list.append(nz)
        
N_LENS_BINS = len(nz_lens_list)
N_SRC_BINS  = len(nz_src_list) 

with h5.File(output_dir + '/truth.h5', 'w') as f:
    f['delta_slabs'] = dens_slabs_true
        
weights_src  = [nz.get_slab_weights_kappa(0.3, 0.) for nz in nz_src_list]
weights_lens = [nz.get_slab_weights_proj_density(0.) for nz in nz_lens_list]

A_ia_fid = 0.5

kappa_list        = [obs_calc.get_kappa(nz_src_list[i], cosmo.Omega_m, 0., dens_slabs_true) for i in range(N_SRC_BINS)]
kappa_ia_list     = [obs_calc.get_kappa_ia(nz_src_list[i], cosmo.Omega_m, 0., A_ia_fid, 0., dens_slabs_true) for i in range(N_SRC_BINS)]
proj_density_list = [obs_calc.get_proj_density(nz_lens_list[i], 0., dens_slabs_true) for i in range(N_LENS_BINS)]

F = FourierTransforms(slab_params.N_grid)

sigma_noise = 0.05 
l = (slab_params.L / slab_params.N_grid)
nbar        = 10e-4 * l**2 * slab_params.slab_width 

data_gen    = DataGenerator(F, sigma_noise=0.05, nbar=nbar)
shape_data  = data_gen.generate_shape_data(kappa_list, kappa_ia_list)
N_gals_data = data_gen.generate_galaxy_counts(proj_density_list)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

key = jax.random.PRNGKey(onp.random.randint(1000000))
rng_key, rng_key_ = jax.random.split(key)

def get_kappa_from_slabs(nz_src_list, Dz_src, dens_slabs):
    kappa_list        = [obs_calc.get_kappa(nz_src_list[i], cosmo.Omega_m, Dz_src[i], dens_slabs) for i in range(N_SRC_BINS)]
    return np.stack(kappa_list)

def get_kappa_ia_from_slabs(nz_src_list, Dz_src, A_ia, eta_ia, dens_slabs):
    kappa_list        = [obs_calc.get_kappa_ia(nz_src_list[i], cosmo.Omega_m, Dz_src[i], A_ia, eta_ia, dens_slabs) for i in range(N_SRC_BINS)]
    return np.stack(kappa_list)

get_gamma = jit(vmap(F.kappa2gamma))

def density_slab_model(nz_src_list, nz_lens_list):
    Omega_m = 0.3
    x_l     = numpyro.sample("x_l", dist.Normal(np.zeros((N_slabs,2,slab_params.N_grid,slab_params.N_grid//2 + 1)), np.ones((N_slabs,2,slab_params.N_grid,slab_params.N_grid//2 + 1))), rng_key=key)
    dens_slabs = transform.x2G(x_l)
    Dz_src = numpyro.sample("Dz_src", dist.Normal(np.zeros(N_SRC_BINS), 0.01 * np.ones(N_SRC_BINS)), rng_key=key)        
    m      = numpyro.sample("m", dist.Normal(np.zeros(N_SRC_BINS), 0.01 * np.ones(N_SRC_BINS)), rng_key=key)
    A_ia   = numpyro.sample("A_ia",   dist.Uniform(-5., 5.), rng_key=key)
    eta_ia = numpyro.sample("eta_ia", dist.Uniform(-5., 5.), rng_key=key)
    
    kappa    = get_kappa_from_slabs(nz_src_list, Dz_src, dens_slabs) 
    kappa_ia = get_kappa_ia_from_slabs(nz_src_list, Dz_src, A_ia, eta_ia, dens_slabs)

    gamma    = get_gamma(kappa + kappa_ia)

    for i in range(N_SRC_BINS):
        numpyro.sample('e_obs_%d'%(i+1), dist.Normal((1. + m[i]) * gamma[i], sigma_noise), obs=shape_data[i])

    Dz_lens = numpyro.sample("Dz_lens", dist.Normal(np.zeros(N_LENS_BINS), 0.01 * np.ones(N_LENS_BINS)), rng_key=key)
    bg      = numpyro.sample("bg", dist.Normal(np.ones(N_LENS_BINS), 0.1 * np.ones(N_LENS_BINS)), rng_key=key)

    for i in range(N_LENS_BINS):
        proj_density = obs_calc.get_proj_density(nz_lens_list[i], Dz_lens[i], dens_slabs)
        mu = np.clip(nbar * (1. + bg[i] * proj_density), 1e-3) 
        numpyro.sample('Ng_%d'%(i+1), dist.Poisson(mu), obs=N_gals_data[i])
 
kernel = NUTS(density_slab_model, target_accept_prob=0.65, max_tree_depth=sampling_params.nuts_tree_depth)
mcmc   = MCMC(kernel, num_warmup=sampling_params.n_warmup, num_samples=sampling_params.n_samples)

mcmc.run(rng_key_, nz_src_list, nz_lens_list)
    
samples = mcmc.get_samples()

x_l             = samples["x_l"]
bg_samples      = samples["bg"]
m_samples       = samples["m"]
Dz_src_samples  = samples["Dz_src"]
Dz_lens_samples = samples["Dz_lens"]
A_ia_samples    = samples["A_ia"]
eta_ia_samples  = samples["eta_ia"]

for i in trange(sampling_params.n_samples):
    dens_slabs_sample = transform.x2G(x_l[i])    
    with h5.File(output_dir + '/mcmc_%d.h5'%(i), 'w') as f:
        f['slab_dens'] = dens_slabs_sample   
        f['bg']        = bg_samples[i]
        f['m']         = m_samples[i]
        f['Dz_src']    = Dz_src_samples[i]
        f['Dz_lens']   = Dz_lens_samples[i]
        f['A_ia']      = A_ia_samples[i]
        f['eta_ia']    = eta_ia_samples[i]

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
from mbi_slabs import *

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
F         = FourierTransforms(slab_params.N_grid)

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
        
weights_src  = [nz.get_slab_weights_kappa(0.3, 0.) for nz in nz_src_list]
weights_lens = [nz.get_slab_weights_proj_density(0.) for nz in nz_lens_list]

A_ia_fid = 0.5

kappa_list        = [obs_calc.get_kappa(nz_src_list[i], cosmo.Omega_m, 0., dens_slabs_true) for i in range(N_SRC_BINS)]
kappa_ia_list     = [obs_calc.get_kappa_ia(nz_src_list[i], cosmo.Omega_m, 0., A_ia_fid, 0., dens_slabs_true) for i in range(N_SRC_BINS)]
proj_density_list = [obs_calc.get_proj_density(nz_lens_list[i], 0., dens_slabs_true) for i in range(N_LENS_BINS)]

sigma_noise = 0.05 
l = (slab_params.L / slab_params.N_grid)
nbar        = 10e-4 * l**2 * slab_params.slab_width 

data_gen    = DataGenerator(F, sigma_noise, nbar)
shape_data  = data_gen.generate_shape_data(kappa_list, kappa_ia_list)
counts_data = data_gen.generate_galaxy_counts(proj_density_list)

write_data(output_dir, dens_slabs_true, shape_data, counts_data)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

key = jax.random.PRNGKey(onp.random.randint(1000000))
rng_key, rng_key_ = jax.random.split(key)

sampler = MCMCSampler(transform, F, obs_calc, N_slabs, slab_params.N_grid, 
                      sigma_noise, nbar)

# Setup and run MCMC
model = sampler.setup_model(N_SRC_BINS, N_LENS_BINS, shape_data, counts_data, key)
samples = sampler.run_mcmc(model, 
                            sampling_params,
                            nz_src_list, 
                            nz_lens_list,
                            rng_key_)

# Save samples
sampler.save_samples(samples, output_dir, sampling_params.n_samples)


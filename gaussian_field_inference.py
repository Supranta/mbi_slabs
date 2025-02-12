import sys

from mbi_slabs.utils import *
from mbi_slabs import *
from mbi_slabs.observables import *
from mbi_slabs.transforms import GaussianTransform
from mbi_slabs.transforms import MapTools

configfile = sys.argv[1]

EnvironmentSetup.setup_jax_env()

cosmo = get_cosmo(0.3)

config = ConfigLoader(configfile)

slab_params     = config.get_slab_config()
observables     = config.get_observables()
sampling_params = config.get_sampling_config()
output_dir      = config.get_output_dir()
data_config     = config.get_data_config()
prior_config    = config.get_prior_config()

slab_definition = [slab_params.chi_min, slab_params.chi_max, slab_params.slab_width]

map_tools = MapTools(slab_params.N_grid, slab_params.L)
F         = FourierTransforms(slab_params.N_grid)

# Initialize catalogs
catalog_init = CatalogInitializer()
z_slabs = catalog_init.setup_boundaries(cosmo, slab_params)
catalogs = catalog_init.create_catalogs(observables)

N_slabs = z_slabs.shape[0]

transform = GaussianTransform(N_slabs, slab_params.N_grid, slab_params.L)

obs_calc = ObservableCalculator(z_slabs)

N_LENS_BINS = len(catalogs.nz_lens_list)
N_SRC_BINS  = len(catalogs.nz_src_list) 

l           = (slab_params.L / slab_params.N_grid)
nbar        = data_config.nbar_Mpc3 * l**2 * slab_params.slab_width 

shape_data, counts_data = read_data(data_config.datafile)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

key = jax.random.PRNGKey(onp.random.randint(1000000))
rng_key, rng_key_ = jax.random.split(key)

sampler = MCMCSampler(transform, F, obs_calc, N_slabs, slab_params.N_grid, 
                      data_config.sigma_e, nbar)

# Setup and run MCMC
model = sampler.setup_model(N_SRC_BINS, N_LENS_BINS, shape_data, counts_data, key)
samples = sampler.run_mcmc(model, 
                            sampling_params,
                            catalogs.nz_src_list, 
                            catalogs.nz_lens_list,
                            prior_config.prior,
                            rng_key_)

# Save samples
sampler.save_samples(samples, output_dir, sampling_params.n_samples)

import sys

from mbi_slabs.utils import *
from mbi_slabs import *
from mbi_slabs.observables import *
from mbi_slabs.transforms import GaussianTransform, LogNormalTransform

configfile = sys.argv[1]

EnvironmentSetup.setup_jax_env()

config = ConfigLoader(configfile)

slab_params     = config.get_slab_config()
observables     = config.get_observables()
sampling_params = config.get_sampling_config()
data_config     = config.get_data_config()
prior_config    = config.get_prior_config()
io_config       = config.get_io_config()

cosmo     = get_cosmo(data_config.cosmo_fid)

slab_definition = [slab_params.chi_min, slab_params.chi_max, slab_params.slab_width]

F         = FourierTransforms(slab_params.N_grid)

# Initialize catalogs
catalog_init = CatalogInitializer(polydist_file=slab_params.polydist_file)
z_slabs = catalog_init.setup_boundaries(cosmo, slab_params)
catalogs = catalog_init.create_catalogs(observables)

N_slabs = z_slabs.shape[0]

if(slab_params.transform=='gaussian'):
    transform = GaussianTransform(N_slabs, slab_params.N_grid, slab_params.theta_max, config.cl_emu_file)
elif(slab_params.transform=='lognormal'):
    transform = LogNormalTransform(N_slabs, slab_params.N_grid, slab_params.theta_max, config.cl_emu_file)

obs_calc = ObservableCalculator(z_slabs)

N_LENS_BINS = len(catalogs.nz_lens_list)
N_SRC_BINS  = len(catalogs.nz_src_list) 

nbar_pix        = data_config.nbar_lens * slab_params.pixel_area_arcmin2
shape_noise_pix = data_config.sigma_e / np.sqrt(data_config.nbar_src * slab_params.pixel_area_arcmin2)

shape_data, counts_data = read_data(data_config.datafile)

##========================================================
##================= Run numpyro sampler ==================
##========================================================

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

key = jax.random.PRNGKey(onp.random.randint(1000000))
rng_key, rng_key_ = jax.random.split(key)

sampler = MCMCSampler(transform, F, obs_calc, 
                                        catalogs.nz_src_list, catalogs.nz_lens_list, 
                                        shape_noise_pix, nbar_pix)
# run burnin MCMC
print("Running a burnin chain...")
model = sampler.setup_model(shape_data, counts_data, key)

burnin_sample = sampler.run_mcmc(model, 
                                    sampling_params, 
                                    prior_config.prior, 
                                    rng_key_)

init_values = sampler.get_init_sample(burnin_sample)
del burnin_sample
# Setup and run MCMC
sampler.set_burn_in(False)

samples = sampler.run_mcmc(model, 
                            sampling_params, 
                            prior_config.prior, 
                            rng_key_,
                            init_values=init_values)

# Save samples
sampler.save_samples(samples, io_config, sampling_params.n_samples)
n_start = sampling_params.n_samples
for n in range(sampling_params.num_sampling_iterations):
    print("Running additional sampling iteration # %d"%(n+1))
    del samples
    samples = sampler.run_mcmc(model, 
                                sampling_params, 
                                prior_config.prior, 
                                rng_key_,
                                last_state=sampler.last_state)
    sampler.save_samples(samples, io_config, sampling_params.n_samples, n_start)
    n_start = n_start + sampling_params.n_samples


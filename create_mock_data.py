import sys

from mbi_slabs.utils import *
from mbi_slabs import *
from mbi_slabs.observables import *
from mbi_slabs.transforms import GaussianTransform
from mbi_slabs.transforms import MapTools

EnvironmentSetup.setup_jax_env()

configfile = sys.argv[1]
cosmo = get_cosmo(0.3)

config = ConfigLoader(configfile)

slab_params     = config.get_slab_config()
observables     = config.get_observables()
sampling_params = config.get_sampling_config()
output_dir      = config.get_output_dir()

slab_definition = [slab_params.chi_min, slab_params.chi_max, slab_params.slab_width]

map_tools = MapTools(slab_params.N_grid, slab_params.L)
F         = FourierTransforms(slab_params.N_grid)

# Initialize catalogs
catalog_init = CatalogInitializer()
z_slabs      = catalog_init.setup_boundaries(cosmo, slab_params)
catalogs     = catalog_init.create_catalogs(observables)

N_slabs = z_slabs.shape[0]

transform = GaussianTransform(N_slabs, slab_params.N_grid, slab_params.L)

x_l             = np.array(onp.random.normal(size=(N_slabs, 2, slab_params.N_grid, slab_params.N_grid//2 + 1))) 
dens_slabs_true = transform.x2G(x_l)
    
obs_calc = ObservableCalculator(z_slabs)

N_LENS_BINS = len(catalogs.nz_lens_list)
N_SRC_BINS  = len(catalogs.nz_src_list) 

A_ia_fid = 0.5

kappa_list        = [obs_calc.get_kappa(catalogs.nz_src_list[i], cosmo.Omega_m, 0., dens_slabs_true) for i in range(N_SRC_BINS)]
kappa_ia_list     = [obs_calc.get_kappa_ia(catalogs.nz_src_list[i], cosmo.Omega_m, 0., A_ia_fid, 0., dens_slabs_true) for i in range(N_SRC_BINS)]
proj_density_list = [obs_calc.get_proj_density(catalogs.nz_lens_list[i], 0., dens_slabs_true) for i in range(N_LENS_BINS)]

sigma_noise = 0.05 
l = (slab_params.L / slab_params.N_grid)
nbar        = 10e-4 * l**2 * slab_params.slab_width 

data_gen    = DataGenerator(F, sigma_noise, nbar)
shape_data  = data_gen.generate_shape_data(kappa_list, kappa_ia_list)
counts_data = data_gen.generate_galaxy_counts(proj_density_list)

write_data(output_dir, dens_slabs_true, shape_data, counts_data)

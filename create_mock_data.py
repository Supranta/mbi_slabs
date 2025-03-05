import sys

from mbi_slabs.utils import *
from mbi_slabs import *
from mbi_slabs.observables import *
from mbi_slabs.transforms import GaussianTransform, LogNormalTransform
from mbi_slabs.transforms import MapTools

EnvironmentSetup.setup_jax_env()

configfile = sys.argv[1]

theta_fid = np.array([0.27, 0.82])[np.newaxis]
cosmo     = get_cosmo(theta_fid[0])

config = ConfigLoader(configfile)

slab_params     = config.get_slab_config()
observables     = config.get_observables()
sampling_params = config.get_sampling_config()
io_config       = config.get_io_config()
data_config     = config.get_data_config()

output_dir = io_config.output_dir

slab_definition = [slab_params.chi_min, slab_params.chi_max, slab_params.slab_width]

map_tools = MapTools(slab_params.N_grid, slab_params.L)
F         = FourierTransforms(slab_params.N_grid)

# Initialize catalogs
catalog_init = CatalogInitializer()
z_slabs      = catalog_init.setup_boundaries(cosmo, slab_params)
catalogs     = catalog_init.create_catalogs(observables)

N_slabs = z_slabs.shape[0]

if(slab_params.transform == "gaussian"):
    transform = GaussianTransform(N_slabs, slab_params.N_grid, slab_params.L, config.pk_emu_file)
elif(slab_params.transform == "lognormal"):
    transform = LogNormalTransform(N_slabs, slab_params.N_grid, slab_params.L, config.pk_emu_file)

x_l             = np.array(onp.random.normal(size=(N_slabs, 2, slab_params.N_grid, slab_params.N_grid//2 + 1))) 
dens_slabs_true = transform.x2delta(x_l, theta_fid)
    
obs_calc = ObservableCalculator(z_slabs)

N_LENS_BINS = len(catalogs.nz_lens_list)
N_SRC_BINS  = len(catalogs.nz_src_list) 

kappa_list        = [obs_calc.get_kappa(catalogs.nz_src_list[i], cosmo, 0., dens_slabs_true) for i in range(N_SRC_BINS)]
kappa_ia_list     = [obs_calc.get_kappa_ia(catalogs.nz_src_list[i], cosmo, 0., data_config.A_ia, 0., dens_slabs_true) for i in range(N_SRC_BINS)]
proj_density_list = [obs_calc.get_proj_density(catalogs.nz_lens_list[i], 0., dens_slabs_true) for i in range(N_LENS_BINS)]

l = (slab_params.L / slab_params.N_grid)
nbar        = data_config.nbar_Mpc3 * l**2 * slab_params.slab_width 

data_gen    = DataGenerator(F, data_config.sigma_e, nbar)
shape_data  = data_gen.generate_shape_data(kappa_list, kappa_ia_list)
counts_data = data_gen.generate_galaxy_counts(proj_density_list)

write_data(data_config.datafile, dens_slabs_true, shape_data, counts_data)

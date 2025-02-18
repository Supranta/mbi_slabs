import matplotlib.pyplot as plt
import h5py as h5
from tqdm import trange
from matplotlib import cm
import sys
import os
from mbi_slabs import *
from mbi_slabs.utils import *
from mbi_slabs.observables import *
import numpy as np

def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.
            
    Args:
        directory_path (str): Path to the directory to be created
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully")
    else:
        print(f"Directory '{directory_path}' already exists")


configfile = sys.argv[1]
config = ConfigLoader(configfile)

cosmo = get_cosmo(0.3)

slab_params     = config.get_slab_config()
observables     = config.get_observables()
sampling_params = config.get_sampling_config()
output_dir      = config.get_output_dir()
data_config     = config.get_data_config()

# Initialize catalogs
catalog_init = CatalogInitializer()
z_slabs = catalog_init.setup_boundaries(cosmo, slab_params)
catalogs = catalog_init.create_catalogs(observables)

obs_calc = ObservableCalculator(z_slabs)

n_samples = (sampling_params.num_sampling_iterations + 1) * sampling_params.n_samples

def get_field(x, scalar=False):
    y_list = []
    for i in trange(n_samples):
        with h5.File(output_dir + '/mcmc_%d.h5'%(i), 'r') as f:
            if scalar:
                y = f[x][()]
            else:
                y = f[x][:]
        y_list.append(y)
    return np.array(y_list)

Dz_lens = get_field('Dz_lens')
bg      = get_field('bg')
#======================================
Dz_src  = get_field('Dz_src')
m       = get_field('m')
#======================================
A_ia    = get_field('A_ia', True)
eta_ia  = get_field('eta_ia', True)

N_SRC_BINS  = 4
N_LENS_BINS = 5

def plot_lens_trace(Dz_lens, bg, savename=None):
    fig, ax = plt.subplots(2,N_LENS_BINS,figsize=(15., 5.))

    for j in range(N_LENS_BINS):
        ax[0,j].set_xticks([])
        ax[1,j].set_xlabel('MCMC step')
        
        ax[0,j].set_ylabel('$\Delta^{(%d)}_{z,L}$'%(j+1))
        ax[0,j].plot(Dz_lens[:,j], 'k-', lw=0.5)
        ax[0,j].axhline(0., c='r', ls='--')

        ax[1,j].set_ylabel('$b^{(%d)}_{g}$'%(j+1))
        ax[1,j].plot(bg[:,j], 'k-', lw=0.5)
        ax[1,j].axhline(1., c='r', ls='--')

    plt.tight_layout()    
    if savename is not None:
        plt.savefig(savename, dpi=150.)
    plt.close()
    
def plot_src_trace(Dz_src, m, savename=None):
    fig, ax = plt.subplots(2,N_SRC_BINS,figsize=(12., 5.))

    for j in range(N_SRC_BINS):
        ax[0,j].set_xticks([])
        ax[1,j].set_xlabel('MCMC step')
        
        ax[0,j].set_ylabel('$\Delta^{(%d)}_{z,S}$'%(j+1))
        ax[0,j].plot(Dz_src[:,j], 'k-', lw=0.5)
        ax[0,j].axhline(0., c='r', ls='--')

        ax[1,j].set_ylabel('$m^{(%d)}$'%(j+1))
        ax[1,j].plot(m[:,j], 'k-', lw=0.5)
        ax[1,j].axhline(0., c='r', ls='--')

    plt.tight_layout()    
    if savename is not None:
        plt.savefig(savename, dpi=150.)
    plt.close()

def plot_ia(A_ia, eta_ia, savename=None):
    fig, ax = plt.subplots(1,2,figsize=(7., 2.5))

    for i in range(2):
        ax[i].set_xlabel('MCMC step')
    ax[0].set_ylabel('$A_{ia}$')
    ax[1].set_ylabel('$\eta_{ia}$')
    ax[0].plot(A_ia, 'k-', lw=0.5)
    ax[1].plot(eta_ia, 'k-', lw=0.5)

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=150.)
    plt.close()

create_directory(output_dir + '/figs/')
create_directory(output_dir + '/figs/trace_plots/')

savename = output_dir + '/figs/trace_plots/lens_trace.png'
plot_lens_trace(Dz_lens, bg, savename)

savename = output_dir + '/figs/trace_plots/src_trace.png'
plot_src_trace(Dz_src, m, savename)

savename = output_dir + '/figs/trace_plots/ia_trace.png'
plot_ia(A_ia, eta_ia, savename)   

#=============== Plot autocorrelation =================
def _chain_autocorr(samples, max_lag=100):
    time_arr = np.arange(max_lag)    
    auto_corr_arr = np.zeros((max_lag))
    for t in time_arr:
        auto_corr_arr[t] = np.corrcoef(np.array([samples[:len(samples)-t], samples[t:len(samples)]]))[0,1]                                                
    return auto_corr_arr

def get_sample_autocorr(samples, max_lag=100):
    autocorr_list = []
    for chain in samples:
        autocorr_i = _chain_autocorr(chain, max_lag)
        autocorr_list.append(autocorr_i)
    return np.array(autocorr_list)

autocorr_Dz_lens = get_sample_autocorr(Dz_lens.T)
autocorr_bg      = get_sample_autocorr(bg.T)

autocorr_Dz_src = get_sample_autocorr(Dz_src.T)
autocorr_m      = get_sample_autocorr(m.T)

autocorr_ia      = get_sample_autocorr(np.array([A_ia, eta_ia]))

create_directory(output_dir + '/figs/autocorr/')

lag_arr = np.arange(100)

fig, ax = plt.subplots(2,3,figsize=(10., 5.))

ax[0,2].axis('off')
ax[0,0].set_title("$\Delta^{L}_z$ autocorrelation")
ax[0,1].set_title("$b_g$ autocorrelation")
ax[1,0].set_title("$\Delta^{S}_z$ autocorrelation")
ax[1,1].set_title("$m$ autocorrelation")
ax[1,2].set_title("IA parameter autocorrelation")

for j in range(2):
    ax[j,0].set_ylabel('Correlation length')
for j in range(3):
    ax[0,j].set_xticks([])
    ax[1,j].set_xlabel('Lag')

for i in range(N_LENS_BINS):
    ax[0,0].plot(lag_arr, autocorr_Dz_lens[i], color=cm.viridis(i/N_LENS_BINS))
    ax[0,1].plot(lag_arr, autocorr_bg[i], color=cm.viridis(i/N_LENS_BINS))

for i in range(N_SRC_BINS):
    ax[1,0].plot(lag_arr, autocorr_Dz_src[i], color=cm.viridis(i/N_SRC_BINS))
    ax[1,1].plot(lag_arr, autocorr_m[i], color=cm.viridis(i/N_SRC_BINS))

ax[1,2].plot(lag_arr, autocorr_ia[0], color=cm.viridis(0/2))
ax[1,2].plot(lag_arr, autocorr_ia[1], color=cm.viridis(1/2))

plt.tight_layout()
plt.savefig(output_dir + '/figs/autocorr/autocorr_chains.png')
plt.close()

# =============================================================
with h5.File(data_config.datafile, 'r') as f:
    delta_true = f['delta_slabs'][:]

def get_adaptive_mean_var(x_n, mean, var, n):
    new_mean = ((n - 1) * mean + x_n) / n
    delta_mean = (new_mean - mean)
    if(n==1):
        new_var = 0
    else:
        new_var = ((n - 2) * (var + delta_mean**2) + (x_n - new_mean)**2) / (n - 1)
    return new_mean, new_var

def get_slab_mean_var():
    with h5.File(output_dir + '/mcmc_0.h5', 'r') as f:
        delta_slab0 = f['slab_dens'][:]
    mean_delta_slab = 0. * delta_slab0
    var_delta_slab  = 0. * delta_slab0
    for i in trange(n_samples):
        with h5.File(output_dir + '/mcmc_%d.h5'%(i), 'r') as f:
            delta_slab = f['slab_dens'][:]
            mean_delta_slab, var_delta_slab = get_adaptive_mean_var(delta_slab, mean_delta_slab, var_delta_slab, i+1)
    return mean_delta_slab, np.sqrt(var_delta_slab)

mean_delta_slab, std_delta_slab = get_slab_mean_var()

def plot_mean_var_maps(true_maps, mean_maps, std_maps, offset_index=0, N_maps=5, savename=None):
    snr_maps = np.abs(mean_maps / std_maps)

    fig, ax = plt.subplots(N_maps,4,figsize=(8., 2. * N_maps))

    ax[0,0].set_title('Truth')
    ax[0,1].set_title('Sample mean')
    ax[0,2].set_title('SNR')
    ax[0,3].set_title('Std. dev.')
    for i in range(N_maps):
        ind = i + offset_index
        DELTA_STD = true_maps[ind].std()
        ax[i,0].imshow(true_maps[ind], vmin=-1.5 * DELTA_STD, vmax=1.5 * DELTA_STD, cmap=cm.coolwarm)
        ax[i,1].imshow(mean_maps[ind], vmin=-1.5 * DELTA_STD, vmax=1.5 * DELTA_STD, cmap=cm.coolwarm)
        ax[i,2].imshow(snr_maps[ind], vmin=0., vmax=1., cmap=cm.coolwarm)
        ax[i,3].imshow(std_maps[ind], vmin=0.7 * DELTA_STD, vmax=1.4 * DELTA_STD, cmap=cm.coolwarm)
        for j in range(4):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
        ax[i,0].set_ylabel("Slab %d"%(ind + 1))
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=150.)
    plt.close()

create_directory(output_dir + '/figs/slab_mean_var/')
for i in range(5):
    savename = output_dir + '/figs/slab_mean_var/%d.png'%(i)
    plot_mean_var_maps(delta_true, mean_delta_slab, std_delta_slab, 5 * i, savename=savename)

#============== Plot observables ==================
def get_kappa(dens_slabs, Dz_src):
    kappa_list = [obs_calc.get_kappa(catalogs.nz_src_list[i], 0.3, Dz_src[i], dens_slabs)
                             for i in range(N_SRC_BINS)]
    return np.array(kappa_list)

def get_delta_g(dens_slabs, Dz_lens):
    proj_density = [obs_calc.get_proj_density(catalogs.nz_lens_list[i], Dz_lens[i], dens_slabs)
                            for i in range(N_LENS_BINS)]
    return np.array(proj_density)

kappa_true = get_kappa(delta_true, np.zeros(N_SRC_BINS))
delta_g_true = get_delta_g(delta_true, np.zeros(N_LENS_BINS))

mean_kappa = 0. * kappa_true
var_kappa  = 0. * kappa_true

mean_delta_g = 0. * delta_g_true
var_delta_g  = 0. * delta_g_true


for i in trange(n_samples):
    with h5.File(output_dir + '/mcmc_%d.h5'%(i), 'r') as f:
        dens_slabs = f['slab_dens'][:]
        Dz_src     = f['Dz_src'][:]
        Dz_lens    = f['Dz_lens'][:]
    kappa_i = get_kappa(dens_slabs, Dz_src)
    delta_g_i = get_delta_g(dens_slabs, Dz_lens)
    
    mean_kappa, var_kappa = get_adaptive_mean_var(kappa_i, mean_kappa, var_kappa, i+1)
    mean_delta_g, var_delta_g = get_adaptive_mean_var(delta_g_i, mean_delta_g, var_delta_g, i+1)

savename = output_dir + '/figs/slab_mean_var/kappa_maps.png'
plot_mean_var_maps(kappa_true, mean_kappa, np.sqrt(var_kappa), N_maps=4, savename=savename)

savename = output_dir + '/figs/slab_mean_var/delta_g_maps.png'
plot_mean_var_maps(delta_g_true, mean_delta_g, np.sqrt(var_delta_g), savename=savename)
#============== Plot for neighboring slab correlations ==================
def get_adaptive_cross_mean_var(x_n, y_n, mean_x, mean_y, var_x, var_y, cross_var, n):
    new_mean_x, new_var_x = get_adaptive_mean_var(x_n, mean_x, var_x, n) 
    new_mean_y, new_var_y = get_adaptive_mean_var(y_n, mean_y, var_y, n) 
    delta_mean_x = (new_mean_x - mean_x)
    delta_mean_y = (new_mean_y - mean_y)
    if(n==1):
        new_cross_var = 0
    else:
        new_cross_var = ((n - 2) * (cross_var + delta_mean_x * delta_mean_y) + (x_n - new_mean_x) * (y_n - new_mean_y)) / (n - 1)
    return new_mean_x, new_mean_y, new_var_x, new_var_y, new_cross_var 

def get_slabs(i, ind1, ind2):
    with h5.File(output_dir + '/mcmc_%d.h5'%(i), 'r') as f:
        dens = f['slab_dens'][:]
    return dens[ind1], dens[ind2]

def get_cross_corr_neighboring_slabs(ind1, ind2):
    mean1 = np.zeros((slab_params.N_grid, slab_params.N_grid))
    mean2 = np.zeros((slab_params.N_grid, slab_params.N_grid))
    var1  = np.zeros((slab_params.N_grid, slab_params.N_grid))
    var2  = np.zeros((slab_params.N_grid, slab_params.N_grid))
    delta_cross_variance = np.zeros((slab_params.N_grid, slab_params.N_grid))
    for i in trange(n_samples):
        delta_1, delta_2 = get_slabs(i, ind1, ind2)
        mean1, mean2, var1, var2, delta_cross_variance = get_adaptive_cross_mean_var(delta_1, delta_2, mean1, mean2, var1, var2, delta_cross_variance, i+1)
    return delta_cross_variance / np.sqrt(var1 * var2)

def plot_crosscorr_neighbors(delta_true, ind, savename=None):
    ind1, ind2 = ind, ind + 1

    rho_12 = get_cross_corr_neighboring_slabs(ind1, ind2)

    delta_true_1 = delta_true[ind1]
    delta_true_2 = delta_true[ind2]

    fig, ax = plt.subplots(1,3,figsize=(11.,2.7))

    ax[0].set_title("$\delta$ (bin %d)"%(ind1 + 1))
    ax[1].set_title("Cross-correlation map")
    ax[2].set_title("$\delta$ (bin %d)"%(ind2 + 1))

    im0 = ax[0].imshow(delta_true_1, vmin=-1.5 * np.std(delta_true_1), vmax=1.5 * np.std(delta_true_1), cmap=cm.coolwarm)
    im1 = ax[1].imshow(rho_12, vmin=-0.2, vmax=0.1, cmap=cm.coolwarm)
    im2 = ax[2].imshow(delta_true_2, vmin=-1.5 * np.std(delta_true_2), vmax=1.5 * np.std(delta_true_2), cmap=cm.coolwarm)

    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1], label='Cross-correlation coeff')
    fig.colorbar(im2, ax=ax[2])

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=150.)
    plt.close()

create_directory(output_dir + '/figs/neighboring_slab_corr/')

N_slabs = 25
for i in range(N_slabs):
    print("i: %d"%(i+1))
    savename = output_dir + '/figs/neighboring_slab_corr/%d.png'%(i+1) 
    plot_crosscorr_neighbors(delta_true, i, savename)


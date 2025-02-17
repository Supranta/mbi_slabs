import matplotlib.pyplot as plt
import h5py as h5
from tqdm import trange
from matplotlib import cm
import sys
from mbi_slabs import *
import numpy as np

configfile = sys.argv[1]
config = ConfigLoader(configfile)

sampling_params = config.get_sampling_config()
output_dir      = config.get_output_dir()

def get_field(x, scalar=False):
    y_list = []
    for i in trange(sampling_params.n_samples):
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

savename = output_dir + '/lens_trace.png'
plot_lens_trace(Dz_lens, bg, savename)

savename = output_dir + '/src_trace.png'
plot_src_trace(Dz_src, m, savename)

savename = output_dir + '/ia_trace.png'
plot_ia(A_ia, eta_ia, savename)   

#=============== Plot autocorrelation =================
print(np.__file__)
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
plt.savefig(output_dir + '/autocorr_chains.png')
plt.close()

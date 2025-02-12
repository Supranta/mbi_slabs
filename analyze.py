import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from tqdm import trange
import sys
from mbi_slabs import *

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

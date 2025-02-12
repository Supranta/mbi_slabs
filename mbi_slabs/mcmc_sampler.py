import jax
import jax.numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import h5py as h5
from tqdm import trange

class MCMCSampler:
    def __init__(self, transform, Fourier, obs_calc, N_slabs, N_grid, sigma_noise, nbar):
        self.transform = transform
        self.Fourier = Fourier
        self.obs_calc = obs_calc
        self.N_slabs = N_slabs
        self.N_grid = N_grid
        self.sigma_noise = sigma_noise
        self.nbar = nbar
        
    def setup_model(self, N_SRC_BINS, N_LENS_BINS, shape_data, counts_data, key):
        def get_kappa_from_slabs(nz_src_list, Dz_src, dens_slabs):
            kappa_list = [self.obs_calc.get_kappa(nz_src_list[i], 0.3, Dz_src[i], dens_slabs) 
                         for i in range(N_SRC_BINS)]
            return np.stack(kappa_list)

        def get_kappa_ia_from_slabs(nz_src_list, Dz_src, A_ia, eta_ia, dens_slabs):
            kappa_list = [self.obs_calc.get_kappa_ia(nz_src_list[i], 0.3, Dz_src[i], A_ia, eta_ia, dens_slabs) 
                         for i in range(N_SRC_BINS)]
            return np.stack(kappa_list)

        def numpyro_sample(prior, x, key):
            if isinstance(prior[x], dict):
                return numpyro.deterministic(x, prior[x]['value'])
            else:
                return numpyro.sample(x, prior[x], rng_key=key)

        def density_slab_model(nz_src_list, nz_lens_list, prior):
            x_l = numpyro.sample("x_l", 
                               dist.Normal(np.zeros((self.N_slabs, 2, self.N_grid, self.N_grid//2 + 1)), 
                                         np.ones((self.N_slabs, 2, self.N_grid, self.N_grid//2 + 1))), 
                               rng_key=key)
            dens_slabs = self.transform.x2G(x_l)
            
            # Sample parameters
            Dz_src  = numpyro_sample(prior, 'Dz_src', key)
            m       = numpyro_sample(prior, 'm', key)
            A_ia    = numpyro_sample(prior, 'A_ia', key)
            eta_ia  = numpyro_sample(prior, 'eta_ia', key)
            Dz_lens = numpyro_sample(prior, 'Dz_lens', key)
            bg      = numpyro_sample(prior, 'bg', key)

            # Calculate observables
            kappa = get_kappa_from_slabs(nz_src_list, Dz_src, dens_slabs)
            kappa_ia = get_kappa_ia_from_slabs(nz_src_list, Dz_src, A_ia, eta_ia, dens_slabs)
            gamma = jax.vmap(self.Fourier.kappa2gamma)(kappa + kappa_ia)

            # Sample observations
            for i in range(N_SRC_BINS):
                numpyro.sample(f'e_obs_{i+1}', 
                             dist.Normal((1. + m[i]) * gamma[i], self.sigma_noise), 
                             obs=shape_data[i])

            for i in range(N_LENS_BINS):
                proj_density = self.obs_calc.get_proj_density(nz_lens_list[i], Dz_lens[i], dens_slabs)
                mu = np.clip(self.nbar * (1. + bg[i] * proj_density), 1e-3)
                numpyro.sample(f'Ng_{i+1}', dist.Poisson(mu), obs=counts_data[i])

        return density_slab_model

    def run_mcmc(self, model, sampling_params, nz_src_list, nz_lens_list, prior, rng_key):
        kernel = NUTS(model, target_accept_prob=0.65, max_tree_depth=sampling_params.nuts_tree_depth)
        mcmc = MCMC(kernel, num_warmup=sampling_params.n_warmup, num_samples=sampling_params.n_samples)
        mcmc.run(rng_key, nz_src_list, nz_lens_list, prior)
        return mcmc.get_samples()

    def save_samples(self, samples, output_dir, n_samples):
        for i in trange(n_samples):
            dens_slabs_sample = self.transform.x2G(samples['x_l'][i])
            with h5.File(f'{output_dir}/mcmc_{i}.h5', 'w') as f:
                f['slab_dens'] = dens_slabs_sample
                f['bg'] = samples['bg'][i]
                f['m'] = samples['m'][i]
                f['Dz_src'] = samples['Dz_src'][i]
                f['Dz_lens'] = samples['Dz_lens'][i]
                f['A_ia'] = samples['A_ia'][i]
                f['eta_ia'] = samples['eta_ia'][i]

# config_utils.py
import yaml
import os
import jax
import jax.numpy as np
import h5py as h5
import numpyro.distributions as dist
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class SlabConfig:
    chi_min: float
    chi_max: float
    slab_width: float
    N_grid: int
    theta_max: float
    transform: str
    cl_emu_file: str
    polydist_file: Optional[str] = None

    def __post_init__(self):
        valid_transforms = ["gaussian", "lognormal"]
        if self.transform not in valid_transforms:
            raise ValueError(f"transform must be one of {valid_transforms}, got {self.transform}")
        theta_pix_arcmin = (self.theta_max * 60.) / self.N_grid
        self.pixel_area_arcmin2 = theta_pix_arcmin**2 

@dataclass
class SamplingConfig:
    n_warmup: int
    n_samples: int
    burnin_warmup: int
    num_sampling_iterations: int
    nuts_tree_depth: Optional[int] = 11
    
@dataclass
class IOConfig:
    output_dir: str
    save_maps: Optional[bool] = False

@dataclass
class DataConfig:
    A_ia: float
    sigma_e: float
    nbar_src: List[float]
    nbar_lens: List[float]
    datafile: str
    Om_fid: float = 0.27
    sigma8_fid: float = 0.82

    def __post_init__(self):
        self.cosmo_fid = np.array([self.Om_fid, self.sigma8_fid])
        
        self.nbar_src = np.array(self.nbar_src)
        self.nbar_lens = np.array(self.nbar_lens)

class EnvironmentSetup:
    @staticmethod
    def setup_jax_env():
        jax.config.update("jax_enable_x64", True)
        os.environ['iTF_XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True '
            '--xla_gpu_enable_async_collectives=true '
            '--xla_gpu_enable_latency_hiding_scheduler=true '
            '--xla_gpu_enable_highest_priority_async_stream=true '
        )

class PriorConfig:
    def __init__(self, prior_config):
        self.prior = {}
        for param in prior_config:
            self.prior[param] = self.get_prior_dist(prior_config, param)

    def convert_str_to_float(self, number_str):
        split_number_str = number_str.split(',')
        number_list = []
        for x in split_number_str:
            number_list.append(float(x))
        return np.array(number_list)

    def check_type_and_convert(self, var):
        if type(var) is str:
            var = self.convert_str_to_float(var)
        return var

    def get_prior_dist(self, prior_config, param):
        prior_dist = prior_config[param]['dist'].lower() 
        assert prior_dist in ['normal', 'uniform', 'deterministic'], "Provided distribution not currently supported"
        if(prior_dist == 'normal'):
            mean = self.check_type_and_convert(prior_config[param]['mean'])
            std  = self.check_type_and_convert(prior_config[param]['std'])
            return dist.Normal(mean, std)
        if(prior_dist == 'uniform'):
            low  = self.check_type_and_convert(prior_config[param]['low'])
            high = self.check_type_and_convert(prior_config[param]['high'])
            return dist.Uniform(low, high)
        if(prior_dist == 'deterministic'):
            value = self.check_type_and_convert(prior_config[param]['value'])
            return {'value': value}

class ConfigLoader:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config: Dict[str, Any] = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        with open(self.config_file, "r") as stream:
            return yaml.safe_load(stream)

    def get_prior_config(self):
        return PriorConfig(self.config['prior'])

    def get_slab_config(self) -> SlabConfig:
        self.cl_emu_file = self.config['slabs']['cl_emu_file']
        return SlabConfig(**self.config['slabs'])

    def get_sampling_config(self) -> SamplingConfig:
        mcmc_config = self.config['sampling']['mcmc']
        return SamplingConfig(n_warmup=mcmc_config['n_warmup'],
                            n_samples=mcmc_config['n_samples'],
                            burnin_warmup=500,
                            nuts_tree_depth=mcmc_config['nuts_tree_depth'],
                            num_sampling_iterations=mcmc_config['num_sampling_iterations'])
                                                                                                                                            
    def get_data_config(self) -> DataConfig:
        data_config = self.config['data']
        return DataConfig(**self.config['data'])
    
    def get_io_config(self) -> IOConfig:
        return IOConfig(**self.config['io'])
                                                                                                                                                            
    def get_observables(self) -> Dict[str, Any]:
        return self.config['observables']

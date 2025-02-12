# config_utils.py
import yaml
import os
import jax
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class SlabConfig:
    chi_min: float
    chi_max: float
    slab_width: float
    N_grid: int
    L: float

@dataclass
class SamplingConfig:
    n_warmup: int
    n_samples: int
    nuts_tree_depth: int
    sample_cosmo: bool

@dataclass
class DataConfig:
    A_ia: float
    sigma_e: float
    nbar_Mpc3: float
    datafile: str

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

class ConfigLoader:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config: Dict[str, Any] = self._load_yaml()
                            
    def _load_yaml(self) -> Dict[str, Any]:
        with open(self.config_file, "r") as stream:
            return yaml.safe_load(stream)
                                                        
    def get_slab_config(self) -> SlabConfig:
        return SlabConfig(**self.config['slabs'])
                                                                        
    def get_sampling_config(self) -> SamplingConfig:
        mcmc_config = self.config['sampling']['mcmc']
        return SamplingConfig(n_warmup=mcmc_config['n_warmup'],
                            n_samples=mcmc_config['n_samples'],
                            nuts_tree_depth=mcmc_config['nuts_tree_depth'],
                            sample_cosmo=self.config['sampling']['sample_cosmo'])
                                                                                                                                            
    def get_data_config(self) -> DataConfig:
        data_config = self.config['data']
        return DataConfig(datafile=data_config['datafile'], 
                            sigma_e=data_config['sigma_e'],
                            nbar_Mpc3=data_config['nbar_Mpc3'],
                            A_ia=data_config['A_ia'])

    def get_output_dir(self) -> str:
        return self.config['io']['output_dir']
                                                                                                                                                            
    def get_observables(self) -> Dict[str, Any]:
        return self.config['observables']

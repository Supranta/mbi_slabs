import jax
import jax.numpy as np
from jax import grad, jit
import jaxopt
import h5py as h5
from tqdm import trange
from sklearn.decomposition import PCA

jax.config.update("jax_enable_x64", True)

@jit
def rbf_kernel(X1, X2, length_scale, signal_var):
    X1 = X1 / length_scale
    X2 = X2 / length_scale
    
    X1_sq = np.sum(X1**2, axis=1)[:, None]
    X2_sq = np.sum(X2**2, axis=1)[None, :]
    X1X2 = np.matmul(X1, X2.T)
    sq_dist = X1_sq + X2_sq - 2 * X1X2
        
    return signal_var * np.exp(-0.5 * sq_dist)
    
@jit
def compute_kernel_matrix(X1, X2, params):
    length_scale, signal_var = params[:-1], params[-1]
    return rbf_kernel(X1, X2, length_scale, signal_var)

class JAXGP:
    def __init__(self, n_dim):
        self.n_dim = n_dim
        
    def negative_log_likelihood(self, params, X, y):
        length_scale = params[:-1]
        signal_var = params[-1]
        
        K_y = compute_kernel_matrix(X, X, params)
        
        L = np.linalg.cholesky(K_y + 1e-6 * np.eye(len(X)))

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
       
        nll = 0.5 * np.dot(y, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * len(X) * np.log(2 * np.pi)
        
        return nll
    
    def nll_wrapper(self, unconstrained_params, X, y):
        params = np.exp(unconstrained_params)
        return self.negative_log_likelihood(params, X, y)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1)

        initial_params = np.array([0. for _ in range(self.n_dim)] + [6.])  
        
        def objective(params):
            return self.nll_wrapper(params, X, y)
       
        optimizer = jaxopt.ScipyMinimize(fun=objective) 
        result    = optimizer.run(initial_params) 
        opt_params = result.params 

        self.params = np.exp(opt_params)
        self.X_train = X
        
        K_y = compute_kernel_matrix(X, X, self.params)
        L = np.linalg.cholesky(K_y + 1e-6 * np.eye(len(X)))
        self.alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
        print(f"Optimized hyperparameters: length_scale={opt_params[:-1]}, "
              f"signal_var={opt_params[-1]:.4f}") 

def normalize_cl(cl_scale):
    l0_scaling = cl_scale[:,0]
    cl_fit = cl_scale / l0_scaling[:,np.newaxis]
    return cl_fit, np.log(l0_scaling)

def fit_pca(cl_scale, N_PCA=6):
    cl_fit, l0_scaling = normalize_cl(cl_scale)
    pca = PCA(N_PCA)
    pca.fit(cl_fit)
    pca_coeff      = pca.transform(cl_fit)
    pca_mean       = pca.mean_
    pca_components = pca.components_
    return np.array(pca_coeff), l0_scaling, np.array(pca_mean), np.array(pca_components)

class ClEmulator:
    def __init__(self, emu_file, lognormal=False):
        self.emu_file = emu_file
        self.emu_trained = False
        with h5.File(self.emu_file, 'r') as f:
            self.ell       = f['ell'][:]
            fid            = f['fiducial']
            self.cl_fid    = fid['cl'][:][0]
            self.theta_fid = fid['theta'][:]
            theta          = f['emulator']['theta'][:]
            cl_scale       = f['emulator']['cl_scale'][:]
            if(lognormal):
                self.y_mean_fid = fid['mu_y'][:]
                y_scale = f['emulator']['y_scale'][:]
            else:
                y_scale = None
            self.emu_trained = 'trained_emu' in f
        self.theta = np.array(theta)
        self.N_slabs = cl_scale.shape[1]
        self.ndim = 2
        self.N_PCA = 6
        self.N_ell = cl_scale.shape[-1]
        self.lognormal = lognormal
        if self.emu_trained:
            self.load_emu()
        else:
            self.train_emu(cl_scale, y_scale)
            self.save_emu()

    def train_emu(self, cl_scale, y_scale=None):
        self.gp_list = []
        pca_means       = []
        pca_components = []
        for i in trange(self.N_slabs):
            pca_coeff, l0_scaling, pca_mean, pca_component = fit_pca(cl_scale[:,i], self.N_PCA)
            pca_means.append(pca_mean)              
            pca_components.append(pca_component)
            l0_scaling_gp = self.train_gp(l0_scaling)
            pca_coeff_gps = []
            pca_coeffs    = [] 
            for j in range(self.N_PCA):
                pca_coeff_data = [self.theta, np.array(pca_coeff[:,j])]
                pca_coeffs.append(np.array(pca_coeff[:,j]))
                pca_coeff_gp = self.train_gp(np.array(pca_coeff[:,j]))
                pca_coeff_gps.append(pca_coeff_gp)
            gp_i = [l0_scaling_gp, pca_coeff_gps]
            if(self.lognormal):
                y_gp = self.train_gp(y_scale[:,i])
                gp_i.append(y_gp)
            self.gp_list.append(gp_i)
        self.pca_mean       = np.array(pca_means)
        self.pca_components = np.array(pca_components)
        self.extract_l0_gp_params()
        self.extract_pca_gp_params()
        if(self.lognormal):
            self.extract_y_gp_params()
            y_mean = self.get_y_mean(self.theta_fid[np.newaxis])

    def train_gp(self, y):
        gp_emu = JAXGP(self.ndim)
        gp_emu.fit(self.theta, y)
        return gp_emu
    
    def _extract_gp_params(self, gp):
        return gp.params, gp.alpha 

    def extract_l0_gp_params(self):
        params_list  = []
        alpha_list   = []
        for i in range(self.N_slabs):
            l0_scaling_gp = self.gp_list[i][0]
            params, alpha = self._extract_gp_params(l0_scaling_gp)
            params_list.append(params)
            alpha_list.append(alpha)
        self.l0_params = np.array(params_list)
        self.l0_alpha  = np.array(alpha_list)

    def extract_y_gp_params(self):
        params_list  = []
        alpha_list   = []
        for i in range(self.N_slabs):
            y_gp = self.gp_list[i][2]
            params, alpha = self._extract_gp_params(y_gp)
            params_list.append(params)
            alpha_list.append(alpha)
        self.y_params = np.array(params_list)
        self.y_alpha  = np.array(alpha_list)

    def extract_pca_gp_params(self):
        params_list  = []
        alpha_list   = []
        for i in range(self.N_slabs):
            pca_gp = self.gp_list[i][1]
            for j in range(self.N_PCA):
                params, alpha = self._extract_gp_params(pca_gp[j])
                params_list.append(params)
                alpha_list.append(alpha)
        self.pca_params = np.array(params_list)
        self.pca_alpha  = np.array(alpha_list)

    def get_gp_kernel(self, params_batch, theta_pred):
        batch_kernel_fn = jax.vmap(lambda p, tp, t: compute_kernel_matrix(tp, t, p),
                                in_axes=(0, None, None))
        return batch_kernel_fn(params_batch, theta_pred, self.theta)

    def get_l0(self, theta_pred):
        Ks = self.get_gp_kernel(self.l0_params, theta_pred)[:,0]
        return np.sum(Ks * self.l0_alpha, axis=1) 

    def get_y_mean(self, theta_pred):
        Ks = self.get_gp_kernel(self.y_params, theta_pred)[:,0]
        y_scale = np.sum(Ks * self.y_alpha, axis=1) 
        return y_scale * self.y_mean_fid

    def get_pca_coeff(self, theta_pred):
        Ks = self.get_gp_kernel(self.pca_params, theta_pred)[:,0]
        pca_coeff = np.sum(Ks * self.pca_alpha, axis=1)
        return pca_coeff.reshape((self.N_slabs, self.N_PCA))

    def predict_cl(self, theta_pred):
        l0 = self.get_l0(theta_pred)[np.newaxis,:,np.newaxis]
        pca_coeffs = self.get_pca_coeff(theta_pred)[:,:,np.newaxis]
        cl_pred = self.pca_mean + np.sum(pca_coeffs * self.pca_components, axis=1)
        return np.exp(l0) * cl_pred * self.cl_fid
 
    def load_emu(self):
        print("Loading emulator...")
        with h5.File(self.emu_file, 'r') as f:
            trained_emu = f['trained_emu']
            self.l0_params = trained_emu['l0_params'][:]
            self.l0_alpha  = trained_emu['l0_alpha'][:]
            self.pca_params = trained_emu['pca_params'][:] 
            self.pca_alpha = trained_emu['pca_alpha'][:]  
            self.pca_mean = trained_emu['pca_mean'][:]
            self.pca_components = trained_emu['pca_components'][:]
            if(self.lognormal):
                self.y_params = trained_emu['y_params'][:]
                self.y_alpha  = trained_emu['y_alpha'][:]
            
    def save_emu(self):
        print("Saving emulator...")
        with h5.File(self.emu_file, 'r+') as f:
            trained_emu = f.create_group('trained_emu')
            trained_emu['l0_params'] = self.l0_params
            trained_emu['l0_alpha']  = self.l0_alpha
            trained_emu['pca_params'] = self.pca_params
            trained_emu['pca_alpha']  = self.pca_alpha
            trained_emu['pca_mean']   = self.pca_mean
            trained_emu['pca_components'] = self.pca_components
            if(self.lognormal):
                trained_emu['y_params'] = self.y_params
                trained_emu['y_alpha']  = self.y_alpha

   

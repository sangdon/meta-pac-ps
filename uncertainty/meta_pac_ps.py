import os, sys
import numpy as np
import pickle
import types
import itertools
import scipy
import math
import warnings
import copy

import torch as tc
from torch.utils.data import DataLoader, TensorDataset
from learning import *
from uncertainty import *
import model
from .util import *
    

class PredSetConstructor_meta(PredSetConstructor):
    def __init__(self, model, params, name_postfix='metaps'):
        super().__init__(model=model, params=params, name_postfix=name_postfix)


    def _save_model(self, best=True, shallow=True):
        if best:
            model_fn = self.mdl_fn_best%('_'+self.name_postfix if self.name_postfix else '')
        else:
            model_fn = self.mdl_fn_final%('_'+self.name_postfix if self.name_postfix else '')
        os.makedirs(os.path.dirname(model_fn), exist_ok=True)
        state_dict = {k: v for k, v in self.mdl.state_dict().items() if 'mdl.' not in k} # do not save a base model        
        tc.save(state_dict, model_fn)
        return model_fn

    
    def _load_model(self, best=True):
        if best:
            model_fn = self.mdl_fn_best%('_'+self.name_postfix if self.name_postfix else '')
        else:
            model_fn = self.mdl_fn_final%('_'+self.name_postfix if self.name_postfix else '')
        print(f'[{"best" if best else "final" } model is loaded] {model_fn}')
        self.mdl.load_state_dict(tc.load(model_fn), strict=False) # it is okay not to have a base model
        return model_fn

    
    def train(self, ld_cal, params):
        n_datasets = params.n_datasets
        n_ways = params.n_ways
        n_shots_cal = params.n_shots_cal
        eps = params.eps
        delta = params.delta
        alpha = params.alpha
        if self.name_postfix is None:
            self.name_postfix = ''    
        self.name_postfix += f'_n_datasets_{n_datasets}_n_ways_{n_ways}_n_shots_cal_{n_shots_cal}_eps_{eps:e}_delta_{delta:e}_alpha_{alpha:e}'
        
        print(f'## construct a meta prediction set: '\
              f'n_datasets = {n_datasets}, n_shots_cal = {n_shots_cal}, '
              f'eps = {eps:.2e}, delta = {delta:.2e}, alpha = {alpha:.2e}')

        #TODO: beter way?
        self.mdl.n_datasets = nn.Parameter(tc.tensor(n_datasets), requires_grad=False)
        self.mdl.n_ways = nn.Parameter(tc.tensor(n_ways), requires_grad=False)
        self.mdl.n_shots = nn.Parameter(tc.tensor(n_shots_cal), requires_grad=False)
        self.mdl.alpha = nn.Parameter(tc.tensor(alpha), requires_grad=False)


        # load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True

        # construct tau for each dataset
        tau_list = []
        for i in range(1, n_datasets+1):
            mdl_i = copy.deepcopy(self.mdl)
            l_i = PredSetConstructor_BC(mdl_i, params=self.params, name_postfix=f"{self.name_postfix}_dataset_{i}")
            l_i.train(ld_cal, types.SimpleNamespace(n=n_shots_cal*n_ways, eps=eps, delta=delta/2.0, verbose=False, save=False))
            tau_list.append((-mdl_i.T.data).exp().item())
        tau_list = tc.tensor(tau_list)
        
        # construct a prediction set for taus
        warnings.warn('write more general code')
        def find_tau(v_list, n, eps, delta):
            T, T_step, T_end, T_opt = 0.0, self.params.T_step, self.params.T_end, 0.0
            while T <= T_end:
                ## CP bound
                error_U = (v_list < T).sum().float()
                k_U, n_U, delta_U = error_U.int().item(), n, delta
                U = bci_clopper_pearson_worst(k_U, n_U, delta_U)

                print(f'[n = {n}, eps = {eps:.2e}, delta = {delta:.2e}, T = {T:.4f}] '
                      f'T_opt = {T_opt:.4f}, #error = {k_U}, error_emp = {k_U/n_U:.4f}, U = {U:.6f}')

                if U <= eps:
                    T_opt = T
                elif U >= self.params.eps_tol*eps: ## no more search if the upper bound is too large
                    break
                T += T_step        
            return T_opt
        T_opt = find_tau(tau_list, n_datasets, delta/2.0, alpha)        
        T_opt_nll = -np.log(T_opt).astype(np.float32)

        # finalize a prediction set model
        self.mdl.T.data = tc.tensor(T_opt_nll)
        self.mdl.n_datasets = nn.Parameter(tc.tensor(n_datasets), requires_grad=False)
        self.mdl.n_ways = nn.Parameter(tc.tensor(n_ways), requires_grad=False)
        self.mdl.n_shots = nn.Parameter(tc.tensor(n_shots_cal), requires_grad=False)
        self.mdl.eps.data = tc.tensor(eps)
        self.mdl.delta.data = tc.tensor(delta)
        self.mdl.alpha = nn.Parameter(tc.tensor(alpha), requires_grad=False)

        self.mdl.to(self.params.device)

        # save
        self._save_model(best=True, shallow=True)
        self._save_model(best=False, shallow=True)
        print()

        return True
        
    
    def test(self, ld, ld_name, verbose=False, save=True):
        ## compute set size and error
        fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'stats_pred_set.pk')
        if False: #os.path.exists(fn) and not self.params.rerun:
            res = pickle.load(open(fn, 'rb'))
            size, error = res['size_test'], res['error_test']
        else:
            print('!! currently always recompute!')
            size, error = [], []
            for x, y in ld:
                # size and error for each test dataset
                size_i = loss_set_size(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
                error_i = loss_set_error(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
                size.append(size_i.mean().unsqueeze(0))
                error.append(error_i.mean().unsqueeze(0))
            size, error = tc.cat(size), tc.cat(error)
            
            if save:
                pickle.dump({'error_test': error, 'size_test': size, 'n': self.mdl.n, 'eps': self.mdl.eps, 'delta': self.mdl.delta}, open(fn, 'wb'))

        if verbose:
            mn = size.min()
            Q1 = size.kthvalue(int(round(size.size(0)*0.25)))[0]
            Q2 = size.median()
            Q3 = size.kthvalue(int(round(size.size(0)*0.75)))[0]
            mx = size.max()
            av = size.mean()
            print(
                f'[test: {ld_name}, n = {self.mdl.n}, eps = {self.mdl.eps:.2e}, delta = {self.mdl.delta:.2e}, T = {(-self.mdl.T.data).exp():.5f}] '
                f'error = {error.mean():.4f}, min = {mn}, 1st-Q = {Q1}, median = {Q2}, 3rd-Q = {Q3}, max = {mx}, mean = {av:.2f}'
            )
            
        return size, error


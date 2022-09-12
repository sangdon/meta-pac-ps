import os, sys
from learning import *
import numpy as np
import pickle
import types
import itertools
import scipy

import torch as tc
from .util import *
from uncertainty import *

def geb_VC(delta, n, d=1.0):
    n = float(n)
    g = np.sqrt(((np.log((2*n)/d) + 1.0) * d + np.log(4/delta))/n)
    return g

def geb_iw_finite(delta, m, n_C, M, d2_max):
    m, n_C, M, d2_max = float(m), float(n_C), float(M), float(d2_max)
    g = 2.0*M*(np.log(n_C) + np.log(1.0/delta)) / 3.0 / m + np.sqrt( 2.0*d2_max*(np.log(n_C) + np.log(1.0/delta))/m )
    return g


def log_factorial(n):

    #log_f = tc.arange(n, 0, -1).float().log().sum()
    log_f = np.sum(np.log(np.arange(n, 0, -1.0)))
    
    return log_f


def log_n_choose_k(n, k):
    if k == 0:
        return np.log(1.0)
    else:
        #res = log_factorial(n) - log_factorial(k) - log_factorial(n-k)
        #res = tc.arange(n, n-k, -1).float().log().sum() - log_factorial(k)
        res = np.sum(np.log(np.arange(n, n-k, -1.0))) - log_factorial(k)
        return res

    
def half_line_bound_upto_k(n, k, eps):
    assert(eps > 0.0)
    ubs = []
    #eps = tc.tensor(eps)
    for i in np.arange(0, k+1):
        bc_log = log_n_choose_k(n, i)
        #log_ub = bc_log + eps.log()*i + (1.0-eps).log()*(n-i)
        #ubs.append(log_ub.exp().unsqueeze(0))
        log_ub = bc_log + np.log(eps)*i + np.log(1.0-eps)*(n-i)
        ubs.append([np.exp(log_ub)])
    ubs = np.concatenate(ubs)
    ub = np.sum(ubs)
    return ub


def binedges_equalmass(x, n_bins):
    n = len(x)
    return np.interp(np.linspace(0, n, n_bins + 1),
                     np.arange(n),
                     np.sort(x))



class PredSetConstructor_BC(PredSetConstructor):
    def __init__(self, model, params=None, name_postfix=None):
        super().__init__(model, params, name_postfix)
        
        
    def _compute_error_permissive_VC(self, eps, delta, n):
        g = geb_VC(delta, n)    
        error_per = eps - g
        # if error_per >= 0.0:
        #     error_per = round(error_per*n)
        #     error_per = min(n, error_per)
        # else:
        #     error_per = None
        # return error_per   
        return round(error_per*n) if error_per >= 0.0 else None

    
    def _compute_error_permissive_direct(self, eps, delta, n):
        k_min = 0
        k_max = n
        bnd_min = half_line_bound_upto_k(n, k_min, eps)
        if bnd_min > delta:
            return None
        assert(bnd_min <= delta)
        k = n
        while True:
            # choose new k
            k_prev = k
            #k = (T(k_min + k_max).float()/2.0).round().long().item()
            k = round(float(k_min + k_max)/2.0)
        
            # terminate condition
            if k == k_prev:
                break
        
            # check whether the current k satisfies the condition
            bnd = half_line_bound_upto_k(n, k, eps)
            if bnd <= delta:
                k_min = k
            else:
                k_max = k

        # confirm that the solution satisfies the condition
        k_best = k_min
        assert(half_line_bound_upto_k(n, k_best, eps) <= delta)
        assert(k_best >= 0 and k_best <= n)
        #error_allow = float(k_best) / float(n)
        return k_best

    
    def _find_opt_T(self, ld, n, error_perm):
        nlogp = []
        for x, y in ld:
            x = to_device(x, self.params.device)
            y = to_device(y, self.params.device)
            nlogp_i = self.mdl(x, y)
            nlogp.append(nlogp_i)
            if n <= sum([len(v) for v in nlogp]):
                break
        nlogp = tc.cat(nlogp)
        assert len(nlogp) >= n, f'len(nlogp) = {len(nlogp)} >= n = {n} should hold'
        nlogp = nlogp[:n]
        nlogp_sorted = nlogp.sort(descending=True)[0]
        T_opt = nlogp_sorted[error_perm]

        return T_opt

                
    def train(self, ld, params):
        n = params.n
        eps = params.eps
        delta = params.delta
        if self.name_postfix is None:
            self.name_postfix = ''    
        self.name_postfix = self.name_postfix + f'_n_{n}_eps_{eps:e}_delta_{delta:e}'
        verbose = params.verbose
        save = params.save
        
        #n, eps, delta = self.mdl.n.item(), self.mdl.eps.item(), self.mdl.delta.item()
        print(f"## construct a prediction set: n = {n}, eps = {eps:.2e}, delta = {delta:.2e}")

        ## load a model
        if not self.params.rerun and self._check_model(best=False):
            if self.params.load_final:
                self._load_model(best=False)
            else:
                self._load_model(best=True)
            return True
        
        ## compute permissive error
        if self.params.bnd_type == 'VC':
            error_permissive = self._compute_error_permissive_VC(eps, delta, n)
        elif self.params.bnd_type == 'direct':
            error_permissive = self._compute_error_permissive_direct(eps, delta, n)
        else:
            raise NotImplementedError
        
        if error_permissive is None:
            print('error_permissive is None')
            T_opt = tc.tensor(np.inf)
        else:
            T_opt = self._find_opt_T(ld, n, error_permissive)
        self.mdl.T.data = T_opt
            
        self.mdl.to(self.params.device)
        print(f"error_permissive = {error_permissive}, T_opt = {T_opt}")

        ## save
        if save:
            self._save_model(best=True)
            self._save_model(best=False)
            print()

        return True
        
        

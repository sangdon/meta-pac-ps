import os, sys
from learning import *
import numpy as np
import pickle
import types
import itertools
import scipy

import torch as tc
from .util import *


class PredSetConstructor(BaseLearner):
    def __init__(self, model, params=None, name_postfix=None):
        super().__init__(model, params, name_postfix)

        
    def train(self, ld):
        raise NotImplementedError
            
        
    def test(self, ld, ld_name, verbose=False, save=True):

        ## compute set size and error
        fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'stats_pred_set.pk')
        if save and os.path.exists(fn) and not self.params.rerun:
            res = pickle.load(open(fn, 'rb'))
            error = res['error_test']
            size = res['size_test']
        else:
            size, error = [], []
            for x, y in ld:
                size_i = loss_set_size(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
                error_i = loss_set_error(x, y, self.mdl, reduction='none', device=self.params.device)['loss']
                size.append(size_i)
                error.append(error_i)
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

            ## plot results

            
        return size, error


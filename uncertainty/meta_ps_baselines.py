import os, sys
import numpy as np
import pickle
import types
import itertools
import scipy
import math
import warnings

import torch as tc

from learning import *
from uncertainty import *
import model
from .util import *
    
class PredSetConstructor_meta_naive(PredSetConstructor_meta):

    def __init__(self, model, params=None, name_postfix=None):
        super().__init__(model=model, params=params, name_postfix=name_postfix)
        self.pac_ps = PredSetConstructor_BC(model=model, params=params, name_postfix=name_postfix)

        
    def train(self, ld, params):
        return self.pac_ps.train(ld, params)



class PredSetConstructor_meta_ideal(PredSetConstructor_meta):
    
    def __init__(self, model, params=None, name_postfix=None):
        super().__init__(model=model, params=params, name_postfix=name_postfix)

        
    def train(self, ld, params):
        self.params_ps = params
        
        return True
        
    
    def test(self, ld, ld_name, verbose=False):

        ## compute set size and error
        fn = os.path.join(self.params.snapshot_root, self.params.exp_name, 'stats_pred_set.pk')
        if False: #os.path.exists(fn) and not self.params.rerun:
            res = pickle.load(open(fn, 'rb'))
            error = res['error_test']
            size = res['size_test']
        else:
            print("!! rerun test always")
            size, error = [], []
            for x, y in ld:

                ##start{update}
                n_cal = self.params_ps.n
                n_adapt = self.params_ps.n_adapt

                # split cal and test
                if 'x_adapt' in x:
                    x_cal = {'x_adapt': x['x_adapt'], 'y_adapt': x['y_adapt'], 'x_eval': x['x_eval'][:n_cal]}
                    y_cal = y[:n_cal]

                    x_test = {'x_adapt': x['x_adapt'], 'y_adapt': x['y_adapt'], 'x_eval': x['x_eval'][n_cal:]}
                    y_test = y[n_cal:]

                    ld_split_cal = [(x_cal, y_cal)]
                    ld_split_test = [(x_test, y_test)]
                elif 'word' in x[0]:
                    # the special case for the FewRel dataset
                    n_ways = self.params_ps.n_ways
                    n_shots_cal = self.params_ps.n_shots_cal
                    n_shots_test = self.params_ps.n_shots_test                    
                    n_shots_adapt = self.params_ps.n_shots_adapt
                    print(f'n_shots_test = {n_shots_test}')
                    
                    support, query = x
                    dim = support['word'].shape[-1]
                    assert(all([v.shape[-1] == dim for k, v in support.items()]))
                    assert(all([v.shape[-1] == dim for k, v in query.items()]))
                    
                    
                    #print(support['word'].shape, query['word'].shape, y.shape)

                    support = {k: v.view(-1, n_ways, n_shots_adapt, dim) for k, v in support.items()}
                    query = {k: v.view(-1, n_ways*n_shots_test, dim) for k, v in query.items()}
                    labels = y.view(-1, n_ways*n_shots_test)
                    
                    #print(support['word'].shape, query['word'].shape, labels.shape)

                    # unroll batch
                    assert(support['word'].shape[0] == query['word'].shape[0] == labels.shape[0])

                    batch_size = labels.shape[0]
                    for i_batch in range(batch_size):
                        support_i_batch = {k: v[i_batch] for k, v in support.items()}                        
                        query_i_batch = {k: v[i_batch] for k, v in query.items()}
                        labels_i_batch = labels[i_batch]

                        # split cal and test
                        support_cal = {k: v.view(-1, dim) for k, v in support_i_batch.items()}
                        query_cal = {k: v[:n_shots_cal*n_ways].view(-1, dim) for k, v in query_i_batch.items()}
                        labels_cal = labels_i_batch[:n_shots_cal*n_ways].unsqueeze(0)
                        query_cal['n_ways'] = n_ways
                        query_cal['n_shots'] = n_shots_cal

                        support_test = {k: v.view(-1, dim) for k, v in support_i_batch.items()}
                        query_test = {k: v[n_shots_cal*n_ways:].view(-1, dim) for k, v in query_i_batch.items()}
                        labels_test = labels_i_batch[n_shots_cal*n_ways:].unsqueeze(0)
                        query_test['n_ways'] = n_ways
                        query_test['n_shots'] = n_shots_test - n_shots_cal


                        # loaders for prediction set construction
                        ld_split_cal = [((support_cal, query_cal), labels_cal)]
                        ld_split_test = [((support_test, query_test), labels_test)]

                        # calibrate
                        l = PredSetConstructor_BC(self.mdl, self.params)
                        l.train(ld_split_cal, self.params_ps)
                        size_i, error_i = l.test(ld_split_test, ld_name=f'none', verbose=False, save=False)
                
                        size.append(size_i.mean().unsqueeze(0))
                        error.append(error_i.mean().unsqueeze(0))

                    continue # don't execute the following code; special treatment for FewRel

                    
                    # x_cal = ({k: v[:n_adapt+n_cal].view(-1, dim) for k, v in support.items()}, {k: v[:n_adapt+n_cal].view(-1, dim) for k, v in query.items()})
                    # y_cal = labels[:n_cal]

                    # x_test = ({k: tc.cat((v[:n_adapt].view(-1, dim), v[n_adapt+n_cal:].view(-1, dim)), 0) for k, v in support.items()},
                    #           {k: tc.cat((v[:n_adapt].view(-1, dim), v[n_adapt+n_cal:].view(-1, dim)), 0) for k, v in query.items()})
                    # y_test = labels[n_cal:]

                    print(len(support_test), len(query_test), len(labels_test))

                    ld_split_cal = zip(zip(support_cal, query_cal), labels_cal)
                    ld_split_test = zip(zip(support_test, query_test), labels_test)
                    
                else:
                    x_cal, y_cal = x[:n_adapt+n_cal], y[:n_cal]
                    x_test, y_test = tc.cat((x[:n_adapt], x[n_adapt+n_cal:]), 0), y[n_cal:]

                    ld_split_cal = zip(x_cal.unsqueeze(0), y_cal.unsqueeze(0))
                    ld_split_test = zip(x_test.unsqueeze(0), y_test.unsqueeze(0))

                # calibrate
                l = PredSetConstructor_BC(self.mdl, self.params)
                l.train(ld_split_cal, self.params_ps)
                size_i, error_i = l.test(ld_split_test, ld_name=f'none', verbose=False, save=False)
                ##end{update}
                
                size.append(size_i.mean().unsqueeze(0))
                error.append(error_i.mean().unsqueeze(0))
            size, error = tc.cat(size), tc.cat(error)
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

            
        return size.mean(), error.mean()


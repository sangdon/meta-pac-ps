import os, sys
import time
import numpy as np
from sklearn import metrics

import torch as tc
from learning import *


class FewshotClsLearner(BaseLearner):
    def __init__(self, mdl, params=None, name_postfix=None):
        super().__init__(mdl, params, name_postfix)
        self.loss_fn_train = loss_xe
        self.loss_fn_val = loss_01
        self.loss_fn_test = loss_01

        
    def test(self, ld, mdl=None, loss_fn=None, ld_name=None, verbose=False):
        t_start = time.time()
        error, *_ = super().test(ld, mdl, loss_fn)
        
        if verbose:
            print('[test%s, %f secs.] classificaiton error = %.2f%%'%(
                ': %s'%(ld_name if ld_name else ''), time.time()-t_start, error*100.0))

        return error,

    
class FewshotDetLearner(BaseLearner):
    def __init__(self, mdl, params=None, name_postfix=None):
        super().__init__(mdl, params, name_postfix)
        self.loss_fn_train = lambda x, y, model, reduction='mean', device=tc.device('cpu'): loss_xe(x, y, model, reduction, device, weight=tc.tensor([1.0, 10.0]))
        self.loss_fn_val = loss_01 #TODO: dummy
        self.loss_fn_test = loss_01 #TODO: dummy

        
    def test(self, ld, mdl=None, loss_fn=None, ld_name=None, verbose=False):
        t_start = time.time()

        mdl = mdl if mdl else self.mdl
        device = self.params.device
        
        y_list = []
        fh_list = []
        for x, y in ld:
            x = to_device(x, device)
            with tc.no_grad():
                fh = mdl(x)['fh']
            y_list.append(y.numpy())
            fh_list.append(fh[:, 1].cpu().numpy())
        y_list = np.concatenate(y_list)
        fh_list = np.concatenate(fh_list)
        
        fpr, tpr, thresholds = metrics.roc_curve(y_list, fh_list, pos_label=1)
        auroc = metrics.auc(fpr, tpr)
    
        if verbose:
            print('[test%s, %f secs.] auroc = %.2f'%(
                ': %s'%(ld_name if ld_name else ''), time.time()-t_start, auroc))
        error = 1.0 - auroc
        return error,


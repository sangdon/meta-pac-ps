import os, sys
import time
from copy import deepcopy

from learning import *
#from uncertainty import compute_ece

class ReptileMetaClsLearner(ClsLearner):
    def __init__(self, mdl, params=None, name_postfix=None):
        super().__init__(mdl, params, name_postfix)

    def train(self, ld_train_list, ld_val_list=None, ld_test=None):

        print(f'## meta iteration = 0/{len(ld_train_list)}')
        if ld_test is not None:
            self.test(ld_test, ld_name='target dataset', verbose=True)
        
        for i_epoch_meta, ld_train in enumerate(ld_train_list):
            print(f'## meta iteration = {i_epoch_meta+1}/{len(ld_train_list)}')
            
            ## get original parameters
            weights_before = deepcopy(self.mdl.state_dict())
            
            ## adapt with small optimization steps
            l = ClsLearner(self.mdl, self.params, f'metaiter_{i_epoch_meta+1}')
            l.train(ld_train)
            
            ## update model parameters
            weights_after = l.mdl.state_dict()
            outerstepsize = self.params.lr_meta * (1 - i_epoch_meta / len(ld_train_list)) # linear schedule
            self.mdl.load_state_dict(
                {name : weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize for name in weights_before})

            ## eval current model
            if ld_test is not None:
                self.test(ld_test, ld_name='target dataset', verbose=True)
            

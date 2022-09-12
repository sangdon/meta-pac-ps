import os, sys
import glob
import time
import random
import pickle
import numpy as np
import os.path as osp
import types
import pandas
import warnings

import torch as tc
from torchvision import transforms
from torch.utils.data import DataLoader
#from PIL import Image
#from data.miniimagenet import CategoriesSampler
import data

DATAFILES = [
    'LLCP2011.XPT',
    'LLCP2012.XPT',
    'LLCP2013.XPT',
    'LLCP2014.XPT',
    'LLCP2015.XPT',
    'LLCP2016.XPT',
    'LLCP2017.XPT',
    'LLCP2018.XPT',
    'LLCP2019.XPT',
    'LLCP2020.XPT',
]

STATEKEY = '_STATE'
LABELKEY = 'CVDINFR4' # Ever Diagnosed with Heart Attack

def load_data(root, fn='data.pk'):
    if os.path.exists(os.path.join(root, fn)):
        return pickle.load(open(os.path.join(root, fn), 'rb'))
    
    years = [fn[4:8] for fn in DATAFILES]

    data = [pandas.read_sas(os.path.join(root, fn)) for fn in DATAFILES]
    keys = [set(d.keys()) for d in data]

    # find common keys
    common_keys = set.intersection(*keys)

    # remove unnecessary keys: IDATE, IYEAR, IMONTH, IDAY, SEQNO
    common_keys = common_keys.difference({'IDATE', 'IYEAR', 'IMONTH', 'IDAY', 'SEQNO'})
    
    common_keys = list(common_keys)
    # print(f'n_common_keys = {len(common_keys)}')
    # print('common_keys =', common_keys)
    
    # remove keys with many nan
    common_val_keys = []
    for k in common_keys:
        rate_inval = np.mean([np.isnan(d[k].to_numpy()).mean() for d in data])
        if rate_inval <= 0.05:
            common_val_keys.append(k)
    print(f'n_common_val_keys = {len(common_val_keys)}')
    print('common_val_keys =', common_val_keys)
    assert(STATEKEY in common_val_keys)
    assert(LABELKEY in common_val_keys)

    # remove data with nan
    data_common_val = [d[common_val_keys].to_numpy() for d in data]
    data_common_val = [d[~np.isnan(d.sum(1))] for d in data_common_val]
    
    # remove idk labels
    ind_label = np.argmax(np.array(common_val_keys) == LABELKEY)
    data_common_val = [d[(d[:, ind_label] == 1) | (d[:, ind_label] == 2)] for d in data_common_val]
    for i, d in enumerate(data_common_val):
        d[d[:, ind_label]==2, ind_label] = 0 # 1: yes, 0: no
        data_common_val[i] = d
    
    # return
    ind_state = np.argmax(np.array(common_val_keys) == STATEKEY)
    assert(ind_state != ind_label)
    states_per_year = [d[:, ind_state].astype(int) for d in data_common_val]
    labels_per_year = [d[:, ind_label].astype(int) for d in data_common_val]
    examples_per_year = [np.delete(d, [ind_state, ind_label], axis=1) for d in data_common_val]

    def get_task_ids(states_per_year):
        strid2id = {}
        cnt = 0
        task_ids = []
        for i, states in enumerate(states_per_year):
            task_ids_i = []
            for state in states:
                strid = f'{i}_{state}'
                if strid in strid2id:
                    pass
                else:
                    strid2id[strid] = cnt
                    cnt = cnt + 1
                task_ids_i.append(strid2id[strid])
            task_ids.append(task_ids_i)
        return task_ids
    
    #task_ids_per_year = get_task_ids(states_per_year)
    
    assert(all([x.shape[0] ==  y.shape[0] for x, y in zip(examples_per_year, labels_per_year)]))
    assert(all([~np.isnan(e.sum()) for e in examples_per_year]))

    for yr, y, s in zip(years, labels_per_year, states_per_year):
        print(f'[year = {yr}] n_states = {len(set(s))}, n_examples = {len(y)}, '
              f'n_pos = {(y==1).sum()} ({float((y==1).sum())/float(len(y))*100.0:.2f}%), n_neg = {(y==0).sum()} ({float((y==0).sum())/float(len(y))*100.0:.2f}%)')

    pickle.dump((years, states_per_year, examples_per_year, labels_per_year), open(os.path.join(root, fn), 'wb'))

    return years, states_per_year, examples_per_year, labels_per_year

    

class HeartDataset:

    def __init__(self, x, y, mean, std):
        self.x = np.concatenate(x, 0)
        self.y = np.concatenate(y, 0)
        self.mean = mean
        self.std = std

        
    def __len__(self):
        return len(self.x)


    def __getitem__(self, i):
        x, y = self.x[i], self.y[i]
        x = (x - self.mean) / self.std
        x = tc.tensor(x).float()
        y = tc.tensor(y).long()
        return x, y


class YearStateSampler:

    def __init__(self, state, n_ways, n_datasets, n_shots):
        self.n_datasets = n_datasets
        self.n_ways = n_ways
        self.n_shots = n_shots # could be n_shots_adapt + n_shots_{train,val,test}
        
        self.ind2yearstate = []
        for ind_year, s_year in enumerate(state):
            for s in s_year:
                self.ind2yearstate.append(f'{ind_year}_{s}')
        self.ind2yearstate = np.array(self.ind2yearstate)
                
        self.yearstate2ind = []
        for yearstate in set(self.ind2yearstate):
            ind = np.argwhere(self.ind2yearstate == yearstate).reshape(-1)
            ind = tc.from_numpy(ind)
            if len(ind) >= self.n_shots*self.n_ways:
                self.yearstate2ind.append(ind)

        assert len(self.yearstate2ind) >= self.n_datasets, f'should hold {len(self.yearstate2ind)} >= {self.n_datasets}'

        
    def __len__(self):
        return self.n_datasets

    
    def __iter__(self):
        for _ in range(self.n_datasets):
            ind_yearstate = tc.randint(len(self.yearstate2ind), (1,)).item()
            ind = self.yearstate2ind[ind_yearstate]            
            ind_batch = ind[tc.randperm(len(ind))[:self.n_shots*self.n_ways]]
            
            yield ind_batch
            
            
    
class Heart:
    def __init__(self, root, args, split_ratio={'train': 0.4, 'val': 0.5, 'test': 0.1}, task_label=False):
        
        seed = args.seed
        n_datasets_train = args.n_datasets_train
        n_datasets_val = args.n_datasets_val
        n_datasets_cal = args.n_datasets_cal
        n_datasets_test = args.n_datasets_test
        num_workers = args.n_workers
        n_ways = args.n_ways
        assert(n_ways == 2)
        
        n_samples_train = args.n_shots_adapt + args.n_shots_train
        n_samples_val = args.n_shots_adapt + args.n_shots_val
        n_samples_cal = args.n_shots_adapt + args.n_shots_cal
        n_samples_test = args.n_shots_adapt + args.n_shots_test

        # load and split data
        years, states_per_year, examples_per_year, labels_per_year = load_data(root)
        
        n_total = len(years)
        n_train = int(n_total * split_ratio['train'])
        n_val = int(n_total * split_ratio['val'])
        n_test = n_total - n_train - n_val
        states_train, states_val, states_test = np.split(states_per_year, [n_train, n_train+n_val])
        examples_train, examples_val, examples_test = np.split(examples_per_year, [n_train, n_train+n_val])
        labels_train, labels_val, labels_test = np.split(labels_per_year, [n_train, n_train+n_val])

        def split(batch, n_samples_adapt):
            x = tc.stack([x for x, y in batch], 0)
            y = tc.stack([y for x, y in batch], 0)

            x = {'x_adapt': x[:n_samples_adapt], 'y_adapt': y[:n_samples_adapt], 'x_eval': x[n_samples_adapt:]}
            y = y[n_samples_adapt:]
            return x, y

        def get_mean_std(examples):
            d = np.concatenate(examples_train, 0)
            m = np.mean(d, axis=0).astype(float)
            s = np.std(d, axis=0).astype(float)
            return m, s
            
        mean_train, std_train = get_mean_std(examples_train)
        
        ds = HeartDataset(examples_train, labels_train, mean_train, std_train)
        sampler = YearStateSampler(states_train, n_ways, n_datasets_train, n_samples_train)
        self.train = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: split(batch, args.n_shots_adapt*n_ways), num_workers=num_workers, pin_memory=True)

        ds = HeartDataset(examples_val, labels_val, mean_train, std_train)
        sampler = YearStateSampler(states_val, n_ways, n_datasets_val, n_samples_val)
        self.val = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: split(batch, args.n_shots_adapt*n_ways), num_workers=num_workers, pin_memory=True)

        ds = HeartDataset(examples_val, labels_val, mean_train, std_train)
        sampler = YearStateSampler(states_val, n_ways, n_datasets_cal, n_samples_cal)
        self.cal = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: split(batch, args.n_shots_adapt*n_ways), num_workers=num_workers, pin_memory=True)

        ds = HeartDataset(examples_test, labels_test, mean_train, std_train)
        sampler = YearStateSampler(states_test, n_ways, n_datasets_test, n_samples_test)
        self.test = DataLoader(dataset=ds, batch_sampler=sampler, collate_fn=lambda batch: split(batch, args.n_shots_adapt*n_ways), num_workers=num_workers, pin_memory=True)

        print(f'#train dataset = {len(self.train.dataset)}, '\
              f'#val datasets = {len(self.val.dataset)}, '\
              f'#cal datasets = {len(self.cal.dataset)}, '\
              f'#test datasets = {len(self.test.dataset)}')


        
if __name__ == '__main__':

    
    dsld = Heart(root='data/heart',
                 args=types.SimpleNamespace(n_datasets_train=100, n_datasets_val=50, n_datasets_cal=100, n_datasets_test=50,
                                            n_ways=2,
                                            n_shots_adapt=5, n_shots_train=5, n_shots_val=5, n_shots_cal=10, n_shots_test=20,
                                            seed=0, n_workers=8))
    

    for i, (x, y) in enumerate(dsld.train):
        print(i, x['x_adapt'].shape, x['y_adapt'].shape, x['x_eval'].shape, y.shape)
        print(y)
    



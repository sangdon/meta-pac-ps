import os, sys
import time
import random
import numpy as np
import types
import pickle

import torch as tc
from torch.utils.data import DataLoader
import data

import chemprop
import collections
import csv


def load_data(fn):

    # Read dataset.
    fn_cache = fn + '.pk'
    if os.path.exists(fn_cache):
        mols = pickle.load(open(fn_cache, 'rb'))
    else:
        mols = []
        with open(fn, "r") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            for row in reader:
                smiles = row[columns[0]]
                task = row[columns[1]]
                value = float(row[columns[2]])

                # Slight misuse of the MoleculeDatapoint to include single target + task id.
                mol = chemprop.data.MoleculeDatapoint(
                    smiles=[smiles],
                    targets=[value],
                    features_generator=["rdkit_2d_normalized"]
                )
                mol.task = task
                mols.append(mol)
        pickle.dump(mols, open(fn_cache, 'wb'))
            
    # Transform to dataset.
    dataset = chemprop.data.MoleculeDataset(mols)

    # Gather task examples.
    tasks = {}
    indices = collections.defaultdict(list)
    for idx, mol in enumerate(dataset):
        if mol.task not in tasks:
            tasks[mol.task] = len(tasks)
        indices[tasks[mol.task]].append(idx)

    return dataset, indices, tasks


class FewShotSampler(tc.utils.data.Sampler):
    """Data sampler for few-shot regression on ChEMBL.
    Randomly samples batches of support examples + query examples.
    """

    def __init__(self, indices, tasks_per_batch, num_support, num_query, iters=None):
        # Indices is a list of lists.
        self.indices = indices
        self.tasks_per_batch = tasks_per_batch
        self.num_support = num_support
        self.num_query = num_query
        self.iters = iters or float("inf")

    def create_batch(self):
        batch_indices = []
        for _ in range(self.tasks_per_batch):
            task_idx = np.random.randint(0, len(self.indices))
            task_indices = self.indices[task_idx]
            if self.num_support + self.num_query <= len(task_indices):
                samples = np.random.permutation(len(task_indices))[:self.num_support + self.num_query]
            else:
                samples = np.random.choice(len(task_indices), self.num_support + self.num_query)
            example_indices = [task_indices[i] for i in samples]
            batch_indices.extend(example_indices)
        return batch_indices

    
    def __iter__(self):
        i = 0
        while i < self.iters:
            yield self.create_batch()
            i += 1

    def __len__(self):
        return self.iters

   
def construct_molecule_batch(batch, n_shots_adapt):
    # smiles = [x.smiles for x in batch]
    # tasks = [x.task for x in batch]

    batch_adapt, batch_eval = batch[:n_shots_adapt], batch[n_shots_adapt:]

    def get(_batch):        
        inputs = chemprop.data.data.construct_molecule_batch(_batch)

        mol_graph = inputs.batch_graph()[0]
        if mol_graph:
            mol_graph = mol_graph.get_components()

        mol_features = inputs.features()
        if mol_features:
            mol_features = tc.from_numpy(np.stack(mol_features)).float()

        targets = inputs.targets()
        targets = [t[0] for t in targets]
        targets = tc.Tensor(targets).float()
        return mol_graph, mol_features, targets

    mol_graph_adapt, mol_features_adapt, targets_adapt = get(batch_adapt)
    mol_graph_eval, mol_features_eval, targets_eval = get(batch_eval)
    
    x = {'x_adapt': [mol_graph_adapt, mol_features_adapt],
         'y_adapt': targets_adapt,
         'x_eval': [mol_graph_eval, mol_features_eval]}
    y = targets_eval.unsqueeze(1)
    
    return x, y

        
    # mol_graph_batch = []
    # mol_feature_batch = []
    # target_batch = []
    # for b in batch:
    #     inputs = chemprop.data.data.construct_molecule_batch([b])
    #     mol_graph = inputs.batch_graph()[0]
    #     if mol_graph:
    #         mol_graph = mol_graph.get_components()

    #     mol_features = inputs.features()
    #     if mol_features:
    #         mol_features = tc.from_numpy(np.stack(mol_features)).float()

    #     targets = inputs.targets()
    #     targets = [t[0] for t in targets]
    #     targets = tc.Tensor(targets).float()

    #     mol_graph_batch.append(mol_graph)
    #     mol_feature_batch.append(mol_features)
    #     target_batch.append(targets)
        
    # mol_feature_batch = tc.cat(mol_feature_batch)
    # target_batch = tc.cat(target_batch)

    # inputs = chemprop.data.data.construct_molecule_batch(batch)

    # mol_graph = inputs.batch_graph()[0]
    # if mol_graph:
    #     mol_graph = mol_graph.get_components()

    # mol_features = inputs.features()
    # print(mol_features)
    # if mol_features:
    #     mol_features = tc.from_numpy(np.stack(mol_features)).float()

    # targets = inputs.targets()
    # targets = [t[0] for t in targets]
    # targets = tc.Tensor(targets).float()

    # inputs = [mol_graph, mol_features, targets]
    # print(f'len(mol_graph) = {len(mol_graph)}')
    # print(mol_graph[0].shape)
    # print(f'len(mol_features) = {len(mol_features)}')
    
    # x = {'x_adapt': [mol_graph_batch[:n_shots_adapt], mol_feature_batch[:n_shots_adapt]],
    #      'y_adapt': target_batch[:n_shots_adapt],
    #      'x_eval': [mol_graph_batch[n_shots_adapt:], mol_feature_batch[n_shots_adapt:]]}
    # y = target_batch[n_shots_adapt:]
    # return x, y

    #return inputs, (smiles, tasks)
    


class Chembl:
    def __init__(self, root, args):
        
        # seed = args.seed
        # n_datasets_train = args.n_datasets_train
        # n_datasets_val = args.n_datasets_val
        # n_datasets_cal = args.n_datasets_cal
        # n_datasets_test = args.n_datasets_test
        # num_workers = args.n_workers
        
        # n_samples_train = args.n_shots_adapt + args.n_shots_train
        # n_samples_val = args.n_shots_adapt + args.n_shots_val
        # n_samples_cal = args.n_shots_adapt + args.n_shots_cal
        # n_samples_test = args.n_shots_adapt + args.n_shots_test

        ds, indices, tasks_train = load_data(os.path.join(root, 'train_molecules.csv'))
        sampler = FewShotSampler(indices=indices,
                                 tasks_per_batch=1,
                                 num_support=args.n_shots_adapt,
                                 num_query=args.n_shots_train,
                                 iters=args.n_datasets_train)
        self.train = tc.utils.data.DataLoader(dataset=ds, batch_sampler=sampler, num_workers=args.n_workers, collate_fn=lambda batch: construct_molecule_batch(batch, args.n_shots_adapt))

        ds, indices, tasks_val = load_data(os.path.join(root, 'val_molecules.csv'))
        sampler = FewShotSampler(indices=indices,
                                 tasks_per_batch=1,
                                 num_support=args.n_shots_adapt,
                                 num_query=args.n_shots_val,
                                 iters=args.n_datasets_val)
        self.val = tc.utils.data.DataLoader(dataset=ds, batch_sampler=sampler, num_workers=args.n_workers, collate_fn=lambda batch: construct_molecule_batch(batch, args.n_shots_adapt))

        ds, indices, tasks_cal = load_data(os.path.join(root, 'val_molecules.csv'))
        sampler = FewShotSampler(indices=indices,
                                 tasks_per_batch=1,
                                 num_support=args.n_shots_adapt,
                                 num_query=args.n_shots_cal,
                                 iters=args.n_datasets_cal)
        self.cal = tc.utils.data.DataLoader(dataset=ds, batch_sampler=sampler, num_workers=args.n_workers, collate_fn=lambda batch: construct_molecule_batch(batch, args.n_shots_adapt))

        ds, indices, tasks_test = load_data(os.path.join(root, 'test_molecules.csv'))
        sampler = FewShotSampler(indices=indices,
                                 tasks_per_batch=1,
                                 num_support=args.n_shots_adapt,
                                 num_query=args.n_shots_test,
                                 iters=args.n_datasets_test)
        self.test = tc.utils.data.DataLoader(dataset=ds, batch_sampler=sampler, num_workers=args.n_workers, collate_fn=lambda batch: construct_molecule_batch(batch, args.n_shots_adapt))

        print(f'#train dataset = {len(self.train.dataset)}, '\
              f'#val datasets = {len(self.val.dataset)}, '\
              f'#cal datasets = {len(self.cal.dataset)}, '\
              f'#test datasets = {len(self.test.dataset)}')
        print(f'#train tasks = {len(tasks_train)}, '\
              f'#val tasks = {len(tasks_val)}, '\
              f'#cal tasks = {len(tasks_cal)}, '\
              f'#test tasks = {len(tasks_test)}')


        
if __name__ == '__main__':

    
    dsld = Chembl(root='data/chembl',
                  args=types.SimpleNamespace(n_datasets_train=100, n_datasets_val=50, n_datasets_cal=100, n_datasets_test=100,
                                             n_ways=2,
                                             n_shots_adapt=5, n_shots_train=5, n_shots_val=5, n_shots_cal=10, n_shots_test=20,
                                             seed=0, n_workers=8))
    
    sys.exit()

    for i, (x, y) in enumerate(dsld.train):
        print(x)
        print(y)
        print()
        
        



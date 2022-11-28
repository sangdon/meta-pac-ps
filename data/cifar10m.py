import os, sys
import glob
import time
import random
import pickle
import numpy as np

import torch as tc
import data

SEVERITY=[0, 1, 2, 3, 4, 5]
SYNPERT_NAME = [
    'GaussianNoise',
    'ShotNoise',
    'ImpulseNoise',
    #'GlassBlur', # slow
    'DefocusBlur',
    'MotionBlur',
    'ZoomBlur',
    'Fog',
    'Frost',
    #'Snow', # slow
    'Contrast',
    'Brightness',
    'JPEGCompression',
    'Pixelate',
    'Elastic',
]

NOISE_BLUR_SYNPERT_NAME = [
    'GaussianNoise',
    'ShotNoise',
    'ImpulseNoise',
    #'GlassBlur', # slow
    'DefocusBlur',
    'MotionBlur',
    'ZoomBlur',
]


class CIFAR10M:
    def __init__(
            self, root, batch_size,
            dataset_sample_size,
            sample_size={'train': None, 'val': None, 'test': None},
            seed=None,
            num_workers=4,
            #ext='JPEG',
            n_max_aug=np.inf,
            perturbation_type=None,
    ):
        n_datasets = sum(dataset_sample_size.values())
        ## get datasets
        singleton_aug_list = []
        if perturbation_type is None:
            pert_type_name = SYNPERT_NAME
        elif perturbation_type == "noiseblur":
            pert_type_name = NOISE_BLUR_SYNPERT_NAME 
        else:
            raise NotImplementedError
        
        for s in SEVERITY:
            if s == 0:
                singleton_aug_list.append('NoAug')
                continue
            for p in pert_type_name:
                singleton_aug_list.append(f'CIFAR10:{p}:{s}')
                
        aug_list = []
        random.seed(seed)
        for _ in range(n_datasets):
            ## draw the number of augmentations
            n_aug = random.randint(1, min(n_max_aug, len(singleton_aug_list)))
            ## draw the sublist from ang_list
            aug_sublist_rnd = random.choices(singleton_aug_list, k=n_aug)
            aug_list.append(aug_sublist_rnd)
        random.seed(int(time.time()))

        ld_list = []
        for aug_types in aug_list:
            ld = data.CIFAR10(root=root[:-1], batch_size=batch_size,
                               aug_types=aug_types, sample_size=sample_size, seed=seed, num_workers=num_workers)
            print()
            ld_list.append(ld)
        print(f'#augmentations = {len(singleton_aug_list)}, #datasets = {len(ld_list)}')

        n_train = dataset_sample_size['train']
        n_val = dataset_sample_size['val']
        n_cal = dataset_sample_size['cal']
        n_test = dataset_sample_size['test']

        self.train = ld_list[:n_train]
        self.val = ld_list[n_train:n_train+n_val]
        self.cal = ld_list[n_train+n_val:n_train+n_val+n_cal]
        self.test = ld_list[n_train+n_val+n_cal:]
        print(f'#train dataset = {len(self.train)}, '
              f'#val datasets = {len(self.val)}, '
              f'#cal datasets = {len(self.cal)}, '
              f'#test datasets = {len(self.test)}')

        
if __name__ == '__main__':
    dsld = CIFAR10M(root='data/CIFAR10', batch_size=100, dataset_sample_size={'source': 10, 'target': 1})





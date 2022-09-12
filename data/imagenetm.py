import os, sys
import glob
import time
import random
import pickle
import numpy as np

import torch as tc
import data


#SEVERITY=[0, 1, 2, 3, 4, 5]
# SYNPERT_NAME = [
#     'GaussianNoise',
#     'ShotNoise',
#     'ImpulseNoise',
#     #'GlassBlur', # slow
#     'DefocusBlur',
#     'MotionBlur',
#     'ZoomBlur',
#     'Fog',
#     'Frost',
#     #'Snow', # slow
#     'Contrast',
#     'Brightness',
#     'JPEGCompression',
#     'Pixelate',
#     'Elastic',
#     'SpeckleNoise', # extra 1
#     'GaussianBlur', # extra 2
# ]

SEVERITY=[0, 1]
#SEVERITY=[0]
SYNPERT_NAME = [
    'GaussianNoise',
    #'ShotNoise',
    'ImpulseNoise',
    #'GlassBlur', # slow
    'DefocusBlur',
    #'MotionBlur',
    'ZoomBlur',
    'Fog',
    'Frost',
    #'Snow', # slow
    'Contrast',
    'Brightness',
    #'JPEGCompression',
    'Pixelate',
    'Elastic',
    #'SpeckleNoise', # extra 1
    #'GaussianBlur', # extra 2
]


class ImageNetMDataset:
    pass

class ImageNetM:
    def __init__(
            self, root, batch_size,
            dataset_sample_size,
            sample_size={'train': None, 'val': None, 'test': None},
            seed=None,
            num_workers=4,
            ext='JPEG',
            n_max_aug=np.inf,
    ):
        n_datasets = sum(dataset_sample_size.values())
        ## get datasets
        singleton_aug_list = []
        for s in SEVERITY:
            if s == 0:
                singleton_aug_list.append('NoAug')
                continue
            for p in SYNPERT_NAME:
                singleton_aug_list.append(f'ImageNet:{p}:{s}')
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
            ld = data.ImageNet(root=root[:-1], batch_size=batch_size,
                               aug_types=aug_types, sample_size=sample_size, seed=seed, num_workers=num_workers, ext=ext)
            print()
            ld_list.append(ld)
        print(f'#augmentations = {len(singleton_aug_list)}, #datasets = {len(ld_list)}')

        self.train = [l for l in ld_list[:dataset_sample_size['train']]]
        self.val = [l for l in ld_list[dataset_sample_size['train']:dataset_sample_size['train']+dataset_sample_size['val']]]
        self.test = [l for l in ld_list[-dataset_sample_size['test']:]]
        print(f'#train dataset = {len(self.train)}, #val datasets = {len(self.val)}, #test datasets = {len(self.test)}')

        
if __name__ == '__main__':
    dsld = ImageNetM(root='imagenet', batch_size=100, dataset_sample_size={'train': 10, 'val': 10, 'test': 1})

## ImageNet
#train =  1281167
#val =  25000
#test =  25000




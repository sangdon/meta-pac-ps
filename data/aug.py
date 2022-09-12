import sys, os
import torch as tc
from . import aug_imagenetc, aug_cifar10c

SYNPERT = [
    'GaussianNoise',
    'ShotNoise',
    'ImpulseNoise',
    'GlassBlur', # slow
    'DefocusBlur',
    'MotionBlur',
    'ZoomBlur',
    'Fog',
    'Frost',
    'Snow', # slow
    'Contrast',
    'Brightness',
    'JPEGCompression',
    'Pixelate',
    'Elastic',
    'SpeckleNoise', # extra 1
    'GaussianBlur', # extra 2
]

def get_aug_tforms(aug_names):
    if aug_names is None:
        return []
    aug_tforms = []
    for aug_name in aug_names:
        ## ImageNet-C/CIFAR10-C synthetic perturbations
        if aug_name.split(":")[0] in ['CIFAR10', 'ImageNet']:
            if aug_name.split(":")[1] in SYNPERT:
                if aug_name.split(":")[0] == 'CIFAR10':
                    synpert = getattr(aug_cifar10c, aug_name.split(":")[1])
                elif aug_name.split(":")[0] == 'ImageNet':
                    synpert = getattr(aug_imagenetc, aug_name.split(":")[1])
                else:
                    raise NotImplementedError
                severity = int(aug_name.split(":")[2])
                aug_tforms += [synpert(severity)]
            else:
                raise NotImplementedError
        elif aug_name == 'NoAug':
            continue
        else:
            raise NotImplementedError

    return aug_tforms



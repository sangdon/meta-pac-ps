import os, sys
import numpy as np
import math

from scipy import stats

import torch as tc


def bci_clopper_pearson(k, n, alpha, two_side=True, use_R=False):
    if two_side:
        if use_R:
            lo = stats.qbeta(alpha/2, int(k), int(n-k+1))[0]
            hi = stats.qbeta(1 - alpha/2, int(k+1), int(n-k))[0]
        else:
            lo = stats.beta.ppf(alpha/2, k, n-k+1)
            hi = stats.beta.ppf(1 - alpha/2, k+1, n-k)
        
            lo = 0.0 if math.isnan(lo) else lo
            hi = 1.0 if math.isnan(hi) else hi
    
        return lo, hi
    else:
        if use_R:
            hi = stats.qbeta(1 - alpha, int(k+1), int(n-k))[0]
        else:    
            hi = stats.beta.ppf(1 - alpha, k+1, n-k)
            hi = 1.0 if math.isnan(hi) else hi
    
        return hi


def bci_clopper_pearson_worst(k, n, alpha):
    return bci_clopper_pearson(k, n, alpha, two_side=False)
    
    

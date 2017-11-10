import numpy as np

def kld(p, q):
    '''Compute Kullback-Leibler divergence between p and q.'''
    return (p * np.log(p / (q + 1e-10))).sum()
import numpy as np 
from numpy import linalg as LA

# from scipy.spatial import distance ## another choice for calculating distances 

## va and vb definately share at least one common item. 
## which is guarrented by the data structure `invert file`.

def cosine(va, vb):
    return np.dot(va, vb) / (LA.norm(va) * LA.norm(vb))
    # return 1 - distance.cosine(va, vb)

def pearson(va, vb):
    va_centered = va - np.mean(va) 
    vb_centered = vb - np.mean(vb) 
    return np.dot(va_centered, vb_centered) / (LA.norm(va_centered) * LA.norm(vb_centered)) ** 0.5
    # return 1 -  distance.correlation(va, vb)


def adjust_cosine(va, vb):
    pass 

def msd(va, vb):
    pass 

def src(va, vb):
    pass
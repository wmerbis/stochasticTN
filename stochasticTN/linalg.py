# Copyright 2023 stochasticTN Developers, GNU GPLv3

import numpy as np
from typing import Any, Optional, List



def svd(tensor: np.ndarray, 
        axis: int = -1,
        Dmax: Optional[int] = None, 
        cutoff: Optional[float] = 0,
        normalizeSVs: Optional[bool] = True):
    ''' Singular value decomposition of MPS tensor
    
    Arg:
        tensor: tensor to SVD
        axis: edge where to split the tensor before flattening to a matrix
        Dmax: maximum number of sungular values to keep
        cutoff: maximum truncation error
        normalizeSVs: if True, normalize singular values to sum to one
        
    Returns:
        u: left tensor factor with shape `tensor.shape[:axis]`+(D)
        s: vector of singular values from large to small
        v: right tensor factor with shape (D)+`tensor.shape[axis:]`
        truncation_error: integer with size of discarded singular values
    '''
    shape = tensor.shape
    left = np.prod(shape[:axis])
    right = np.prod(shape[axis:])
    matrix = tensor.reshape((left, right))
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    if normalizeSVs:
        s /= s.sum()
    truncation_error = 0
    curD = len(s)
    while s[-1] < cutoff:
        truncation_error += s[-1]
        curD -=1
        s = s[:curD]
    u = u[:,:curD]
    v = v[:curD,:]
    
    if Dmax is not None and curD > Dmax:
        truncation_error += sum(s[Dmax:])
        u = u[:,:Dmax]
        s = s[:Dmax]
        v = v[:Dmax,:]
        curD =Dmax
        
    return u.reshape(shape[:axis]+(curD,)), s, v.reshape((curD,)+shape[axis:]), truncation_error



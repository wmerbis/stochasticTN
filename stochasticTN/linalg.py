#
#
#
#
#
#
""" Implementations of common TN operations for stochastic MPS """

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
    matrix = np.reshape(tensor, (left, right))
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    if normalizeSVs:
        s /= s.sum()
    truncation_error = 0
    curD = len(s)
    
    if Dmax is not None and curD > Dmax:
        truncation_error += sum(s[Dmax:])
        u = u[:,:Dmax]
        s = s[:Dmax]
        v = v[:Dmax,:]
        curD =Dmax
    
    if s[-1] <= cutoff:
        truncation_error = s[s<=cutoff].sum()
        s = s[s>cutoff]
        curD = len(s)
        u = u[:,:curD]
        v = v[:curD,:]
            
    return np.reshape(u, shape[:axis]+(curD,)), s, np.reshape(v, (curD,)+shape[axis:]), truncation_error



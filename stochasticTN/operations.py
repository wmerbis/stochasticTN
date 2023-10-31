#
#
#
#
#
""" Implementations of common TN operations for stochastic MPS """

import numpy as np
from stochasticTN.mps import MPS
from stochasticTN.mpo import MPO
from typing import Any, Optional, List

def ContractBraKet(ket, bra):
    ''' contracts a ket with a bra
    
    Args:
        ket: a single mps tensor 
        bra: a single mps tensor
        
    Returns:
        contracted ket - bra with index structure:
    #  0--ket--2
    #     |
    #  1--bra--3
    '''
    
    Tens = np.tensordot(ket, bra, axes=[1,1])
    return np.transpose(Tens, (0,2,1,3))

def ContractBraOKet(ket, O, bra):
    ''' contracts a ket with an operator O and a bra
    
    Args:
        ket: a single mps tensor 
        O: a single mpo tensor
        bra: a single mps tensor
        
    Returns:
        contracted ket - O - bra with index structure:
      0--ket--3
          |
      1-- O --4
          |
      2--bra--5
    '''
    
    Tens = np.tensordot(O, bra, axes=[1,1])
    Tens = np.tensordot(ket, Tens, axes = [1,1])
    
    return np.transpose(Tens, (0,2,4,1,3,5))

def ContractBraO2Ket(ket, O, bra):
    ''' Contracts a ket with an operator O^2 and a bra
    
    Args:
        ket: a single mps tensor 
        O: a single mpo tensor
        bra: a single mps tensor
        
    Returns:
        contracted ket - O - O - bra with index structure:
      0--ket--4
          |
      1-- O --5
          |
      2-- O --6
          |
      3--bra--7
    '''
        
    Tens = np.tensordot(O, bra, axes = [1,1])
    Tens = np.tensordot(O, Tens, axes = [1,1])
    Tens = np.tensordot(ket, Tens, axes = [1,1])
    
    return np.transpose(Tens, (0,2,4,6,1,3,5,7))

def getbra(mps: MPS, cx: Optional[str] ='stoch'):
    N=len(mps)
    ket = mps.tensors
    if cx == 'complex':
        bra = {}
        for i in range(N):
            bra[i] = np.conj(ket[i])
    elif cx == 'stoch':
        bra = {}
        for i in range(N):
            bra[i] = np.reshape(np.ones([2]), (1,2,1))
    else:
        bra = ket
    return bra

def MPOexpectation(mps: MPS, mpo: MPO, cx: Optional[str] ='stoch'):
    ''' Compute the expectation value of the MPO on a MPS
    
    Args:
        mps: the MPS 
        mpo: the MPO
        cx: protocol for the bra's: 
            * 'complex' computes the expectation value in the L^2 norm for complex or real mps
            * 'stoch' computes the expectation value in the L^1 norm for real valued mps
            
    Returns:
        MPO expectation value
    '''
    N=len(mps)
    ket = mps.tensors
    MPO = mpo.tensors
    bra=getbra(mps,cx)
    
    Tens = ContractBraOKet(ket[0], MPO[0], bra[0])
#     Tens = np.reshape(Tens, (Tens.shape[3],Tens.shape[4],Tens.shape[5]))
    
    for i in range(1,N):
        Tens = np.tensordot(Tens, bra[i], axes = [5,0])
        Tens = np.tensordot(Tens, MPO[i], axes = ([4,5],[0,1]))
        Tens = np.tensordot(Tens, ket[i], axes = ([3,5],[0,1])).transpose(0,1,2,5,4,3)
    
    sh = Tens.shape
    Tens = np.reshape(Tens, (sh[0]*sh[1]*sh[2],sh[3]*sh[4]*sh[5]))
    
    return np.trace(Tens)


def MPOvariance(mps: MPS, mpo: MPO, cx: Optional[str] ='stoch'):
    ''' Compute the variance of an MPO on a MPS
    
    Args:
        mps: the MPS 
        mpo: the MPO
        cx: protocol for the bra's: 
            * 'complex' computes the expectation value in the L^2 norm for complex or real mps
            * 'stoch' computes the expectation value in the L^1 norm for real valued mps
            
    Returns:
        MPO variance
    '''
    
    N=len(mps)
    ket = mps.tensors
    MPO = mpo.tensors
    bra=getbra(mps,cx)

    Tens = ContractBraO2Ket(ket[0], MPO[0], bra[0])
#     Tens = np.reshape(Tens, (Tens.shape[4],Tens.shape[5],Tens.shape[6],Tens.shape[7]))
    
    for i in range(1,N):
        Tens = np.tensordot(Tens, bra[i], axes = [7,0])
        Tens = np.tensordot(Tens, MPO[i], axes = ([6,7],[0,1]))
        Tens = np.tensordot(Tens, MPO[i], axes = ([5,7],[0,1]))
        Tens = np.tensordot(Tens, ket[i], axes = ([4,7],[0,1])).transpose(0,1,2,3,7,6,5,4)
        
    sh = Tens.shape
    Tens = np.reshape(Tens, (sh[0]*sh[1]*sh[2]*sh[3],sh[4]*sh[5]*sh[6]*sh[7]))    

    return np.trace(Tens)

def single_site_occupancy(mps: MPS, site: int, norm: Optional[bool] = 0):
    ''' Computes the expectation value for 'site' to be occupied
    
    Args:
        mps: Input MPS     
        site: The site whose expectation value is desired
        norm: if True, the norm of the MPS is computed and result is normalize.
              else, an unnormalized result is returned
        
    Returns:
        PI: Expectation value for 'site' to be occupied in the MPS 'mps'
        
    '''
    N=len(mps)
    if site >= N:
        raise ValueError("Site is larger than chain length!") 
    PI=[0,1]
    normV=[1,1]
    if norm == 0:
        norm = mps.norm()
    
    if site ==0 :
        PIcont= np.tensordot(mps.tensors[0], PI, axes = [1,0])
    else:
        PIcont= np.tensordot(mps.tensors[0], normV, axes=[1,0])
    
    for i in range(1,N):
        if i == site:
            PIconti = np.tensordot(mps.tensors[i], PI, axes=[1,0])
        else:
            PIconti = np.tensordot(mps.tensors[i], normV, axes=[1,0])
        
        PIcont=np.tensordot(PIcont,PIconti, axes=[1,0])
    
    return np.trace(PIcont)/norm
    
def single_site_vacancy(mps: MPS, site: int, norm: Optional[bool] = 0):
    ''' Computes the expectation value for 'site' to be empty
    
    Args:
        mps: Input MPS     
        site: The site whose expectation value is desired
        norm: if True, the norm of the MPS is computed and result is normalize.
              else, an unnormalized result is returned
        
    Returns:
        PI: Expectation value for 'site' to be empty in the MPS 'mps'
        
    '''
    N=len(mps)
    if site >= N:
        raise ValueError("Site is larger than chain length!") 
    
    PI=[1,0]
    normV=[1,1]
    
    if norm == 0:
        norm = mps.norm()
    
    if site ==0 :
        PIcont= np.tensordot(mps.tensors[0], PI, axes = [1,0])
    else:
        PIcont= np.tensordot(mps.tensors[0], normV, axes=[1,0])
    
    for i in range(1,N):
        if i == site:
            PIconti = np.tensordot(mps.tensors[i], PI, axes=[1,0])
        else:
            PIconti = np.tensordot(mps.tensors[i], normV, axes=[1,0])
        
        PIcont=np.tensordot(PIcont,PIconti, axes=[1,0])
    
    return np.trace(PIcont)/norm
    
    
def single_site_magnetization(mps: MPS, site: int, norm: Optional[bool] = True):
    ''' Computes the expectation value for the magnetization of 'site'
    
    Args:
        mps: Input MPS     
        site: The site whose expectation value is desired
        norm: if True, the norm of the MPS is computed and result is normalize.
              else, an unnormalized result is returned
        
    Returns:
        PI: Expectation value for magnetization of 'site' in the MPS 'mps'
        
    '''
    N=len(mps)
    PI=[1,-1]
    normV=[1,1]
    
    if norm == 0:
        norm = mps.norm()
        
    if site ==0 :
        PIcont= np.tensordot(mps.tensors[0], PI, axes = [1,0])
    else:
        PIcont= np.tensordot(mps.tensors[0], normV, axes=[1,0])
    
    for i in range(1,N):
        if i == site:
            PIconti = np.tensordot(mps.tensors[i], PI, axes=[1,0])
        else:
            PIconti = np.tensordot(mps.tensors[i], normV, axes=[1,0])
        
        PIcont=np.tensordot(PIcont,PIconti, axes=[1,0])
    
    return np.trace(PIcont)/norm


    
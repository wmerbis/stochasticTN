#
#
#
#
#
""" Implementations of common TN operations for stochastic MPS """

import numpy as np
from stochasticTN.mps import MPS
from stochasticTN.mpo import MPO
from stochasticTN.linalg import svd
from scipy.stats import entropy
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

def MPSMPOcontraction(mps: MPS, mpo: MPO, 
                      renormalize: Optional[bool] = False,
                      Dmax: Optional[int] = None,
                      cutoff: Optional[float] = 0):
    '''
    Contracts mps with an mpo site by site and optionaly performs trunctation on the singular value
    spectrum. Overwrites input mps with the result of the contraction
    
    Args:
        mps: input stochasticTN MPS object
        mpo: input stochasticTN MPO object
        renormalize: if True the output mps will be renormalized with specified cutoff and/or maximal bond dimension
        Dmax: optional maximal bond dimension
        cutoff: optional cutoff on the singular value spectrum
        
    Returns:
        err: truncation error after renormalization
    '''
    if len(mps) != len(mpo):
        raise ValueError("MPS and MPO of different lengths")
    N = len(mps)
    mps_bonds = mps.bond_dimensions
    mpo_bonds = mpo.bond_dimensions
    d = mps.physical_dimensions
    err = 0
    
    for i, t in enumerate(mps):
            
        ttemp = np.tensordot(t, mpo.tensors[i], axes = [1,2]).transpose(0,2,3,1,4)
        ttemp = np.reshape(ttemp, (mps_bonds[i]*mpo_bonds[i], d[i], mps_bonds[i+1]*mpo_bonds[i+1]))
        
        if renormalize and i>0:
            ttemp = np.tensordot(sv, ttemp, axes = [1,0]) 
                
        if renormalize and i<N-1:
            u, s, v, e = svd(ttemp, -1, Dmax, cutoff, True)
            err += e
            mps.tensors[i] = u
            sv = np.tensordot(np.diag(s), v, axes =[1,0])
            mps.center = i+1
        else:
            mps.tensors[i] = ttemp
    
    if renormalize:
        e = mps.position(0, True, Dmax, cutoff) 
        err += e
        
    return err

def MPOMPOcontraction(mpo1: MPO, mpo2: MPO, 
                      renormalize: Optional[bool] = False,
                      Dmax: Optional[int] = None,
                      cutoff: Optional[float] = 0,
                      breaker: Optional[bool] = False,
                      computeEE: Optional[bool] = False):
    '''
    Contracts mpo with another mpo site by site and optionaly performs trunctation on the singular value
    spectrum. 
    
    Args:
        mpo1: input stochasticTN MPO object
        mpo2: input second stochasticTN MPO object
        renormalize: if True the output mps will be renormalized with specified cutoff and/or maximal bond dimension
        Dmax: optional maximal bond dimension
        cutoff: optional cutoff on the singular value spectrum
        breaker: optional boolian; if True then the algorithm raises an error when Dmax is reached
        
    Returns:
        mpo: Output MPO object
        err: truncation error after renormalization
    '''
    if len(mpo1) != len(mpo2):
        raise ValueError("MPOs are of different lengths")
    N = len(mpo1)
    mpo1_bonds = mpo1.bond_dimensions
    mpo2_bonds = mpo2.bond_dimensions
    d = mpo1.physical_dimensions
    err = 0
    tensors = N*[None]
    
    for i, t in enumerate(mpo1):
            
        ttemp = np.tensordot(t, mpo2.tensors[i], axes = [1,2]).transpose(0,3,4,1,2,5)
        ttemp = np.reshape(ttemp, (mpo1_bonds[i]*mpo2_bonds[i], d[i], d[i], mpo1_bonds[i+1]*mpo2_bonds[i+1]))
        
        if renormalize and i>0:
            ttemp = np.tensordot(sv, ttemp, axes = [1,0]) 
                
        if renormalize and i<N-1:
            u, s, v, e = svd(ttemp, -1, Dmax, cutoff, True)
            if breaker and len(s) == Dmax:
                raise ValueError("Maximal bond dimension reached")
            err += e
            tensors[i] = u
            sv = np.tensordot(np.diag(s), v, axes =[1,0])
            
            
        else:
            tensors[i] = ttemp
    
    mpo = MPO(tensors, canonicalize=False)
    mpo.r = mpo1.r
    mpo.s = mpo1.s
    
    if renormalize and computeEE:
        mpo.center = N-1
        e = mpo.position(N//2-1,True, Dmax, cutoff)
        err+=e
        
        Tens = np.tensordot(mpo.tensors[N//2-1], mpo.tensors[N//2] , axes=[3,0] )
        u, s, v, e = svd(Tens, 3, normalizeSVs = False, cutoff = cutoff)
        p = s**2
        EE = entropy(p, base = 2)
        e = mpo.position(0, True, Dmax, cutoff)
        err+=e
        return mpo, err, EE
    
    if renormalize:
        mpo.center = N-1
        e = mpo.position(0, True, Dmax, cutoff) 
        err += e
        

        
    return mpo, err
    

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


def full_matrix(mpo: MPO):
    '''
    Contract the mpo to build the full matrix. Only possible for len(mpo) < 28
    '''
    n = len(mpo)
    if n>=28:
        raise ValueError("MPO too large to contract exactly...")
    
    Q = mpo.tensors[0]
    for i in range(1, n):
        Q = np.tensordot(Q, mpo.tensors[i], axes = [-1,0])
    tp = tuple([2*i+1 for i in range(n+1)])+tuple([2*i for i in range(n+1)])
    return Q.transpose(tp).reshape((2**n, 2**n))
    
    
def full_distribution(mps, norm = 0):
    '''
    Contract mps to get the full distribution over all configurations. Only possible for len(mps)<28
    '''
    n = len(mps)
    if n>=28:
        raise ValueError("MPS too large to contract exactly...")
    
    if norm ==0:
        norm = mps.norm()
        
    P = mps.tensors[0]
    for i in range(1, n):
        P = np.tensordot(P, mps.tensors[i], axes = [-1,0])
    
    return P.reshape(2**n)/norm
    
    
    
def marginal(mps, sites, norm = 0):
    '''
    Computes the marginal distribution over the sites specified in `sites`
    '''
    n = len(mps)
    if norm == 0:
        norm = mps.norm()
    
    flat = np.ones(2)
    tens = np.ones(1)
    for i in range(len(mps)):
        if i in sites:
            tens = np.tensordot(tens, mps.tensors[i], axes = [-1,0])
        else:
            marg = np.tensordot(mps.tensors[i], flat, axes = [1,0])
            tens = np.tensordot(tens, marg, axes = [-1,0])
    
    return tens.flatten()/norm  

def compute_pk_infected(mps, canonicalize=False):
    ''' 
    Compute the probability distribution of having exactly k infected, as a function of k
    '''
    n = len(mps)
    pn = np.zeros(n+1)
    norm = mps.norm()
    for k in range(n+1):
        mpo_k = project_on_k_infected(n,k,canonicalize)
        pn[k] = stn.MPOexpectation(mps,mpo_k)/norm
    return pn
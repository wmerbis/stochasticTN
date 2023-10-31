#
#
#
#
#
#
""" Implementations of various information theory measures for the stochastic MPS class
"""

import numpy as np
from stochasticTN.mps import MPS
from stochasticTN.linalg import svd
from scipy.stats import entropy
import matplotlib.pyplot as plt
from typing import Optional


def ShannonE(mps: MPS, base: Optional[float] = 2) -> float:
    ''' Computes the Shannon entropy of the MPS. 
    
    NOTE: This is exponentially costly as it constructs the full 2^N dimensional probability distribution.
    
    ONLY USE FOR SMALL MPS
        
    Args:
        mps: Input MPS     
        base: Optional base for the logarithm (default = 2)
       
    Returns:
        Hx: Shannon entropy of MPS.
    
    '''
    
    d=mps.physical_dimensions[0]
    N=len(mps)
    px = mps.tensors[0]
    for i in range(1,N):
        px = np.tensordot(px, mps.tensors[i], axes=[-1,0])
    
    px = np.trace(px, axis1 = 0, axis2 = -1 )
    px = np.reshape(px, d**N)
    if mps.norm() < 0:
        px = -px
    negprob = px[px<0].sum()
    px = px[px>=0]
    Hx = entropy(px, base=base)
 
    if negprob != 0:
        print("Amount of negative probability encountered in ShannonE: ", negprob)
    return Hx

def mutual_information(mps: MPS, site: int, SE: Optional[float] = 0, base: Optional[float] = 2) -> float:
    ''' Computes the Mutual information (log base 2) between two sides of the MPS, separated to the right of 'site'.
   
    NOTE: This is exponentially costly as it constructs the full 2^N dimensional probability distribution
        
    Args:
        mps: Input MPS as dictionary of N elements with np.array values
        site: Node to the left of the bond which is cut. 
        SE: If Schannon entropy is known, input it here to speed up the calculation.
        base: Optional base for the logarithm (default = 2)        
           
    Returns:
        MI: Mutual information between the two sides of the MPS.
        '''
    d=mps.physical_dimensions[0]
    N=len(mps)
    normV=[1,1]
    negprob = 0
    norm = mps.norm()
    
    ketR = np.tensordot( mps.tensors[N-1], normV, axes=[1,0])
    for i in range(N-2,site, -1):
        ketRi = np.tensordot(mps.tensors[i], normV, axes=[1,0])
        ketR = np.tensordot(ketRi, ketR, axes=[1,0])
    for i in range(site, -1,-1):
        ketR = np.tensordot(mps.tensors[i], ketR, axes=[2,0])
    ketR = np.trace(ketR, axis1 = 0, axis2 = -1)
    ketR= np.reshape(ketR, d**(site+1))
    if norm < 0:
        ketR = -ketR
    negprob += ketR[ketR<0].sum()
    ketR = ketR[ketR>=0]
    HR = entropy(ketR, base=base)

    ketL = np.tensordot(mps.tensors[0],normV, axes = [1,0])
    for i in range(1,site+1):
        ketLi = np.tensordot(mps.tensors[i], normV, axes=[1,0])
        ketL = np.tensordot(ketL, ketLi, axes=[1,0])
    for i in range(site+1,N):
        ketL = np.tensordot(ketL, mps.tensors[i], axes = [-1,0])
    ketL = np.trace(ketL, axis1=0, axis2=-1)
    ketL = np.reshape(ketL, d**(N-site-1))

    if norm < 0:
        ketL = -ketL
        
    negprob += ketL[ketL<0].sum()
    ketL = ketL[ketL>=0]
    HL = entropy(ketL, base=base)
            
    if SE == 0:
        HLR = ShannonE(mps)
    else:
        HLR = SE
     
    if negprob != 0:
        print("Amount of negative probability encountered in Mutual Information: ", negprob)

    return HL + HR - HLR

def singular_value_entropy(mps: MPS, site: int, 
                           base: Optional[float] = 2,
                           method: Optional[str] = "square", 
                           plotSVs: Optional[bool] = False) -> float:
    """ Computes the entropy of the singular value spectrum across the bond right of 'site'.
    
    Args:
        mps: an MPS whose singular value entropy across the bond is required
        site: integer specifying the site left of the bond whose singular value entropy is sought
        base: base for the logarithm, default is 2
        method: two options:
            * 'square' computes the entropy of the spectrum of squared SVs
            * 'linear' compute the entropy of the spectrum of SVs
        plotSVs: plots the SV spectrum for inspection 
            
    Returns
        HSV: Shannon entropy of singular value spectrum across the bond to the right of 'site' 
    """
    
    N=len(mps)
    d=mps.physical_dimensions[0]
    center_site = mps.center
#     MPSmc=MPS.copy()
    mps.position(site)

    Tens = np.tensordot(mps.tensors[site], mps.tensors[site+1] , axes=[2,0] )
    u, s, v, _ = svd(Tens, 2, normalizeSVs = False)
    #     print("sum of sing val:", sum(s))
    
    if plotSVs:
        plt.semilogy(s)
        plt.title('Singular value spectrum')
        plt.show()
    
    if method == 'square':
        p = s**2
        Hx = entropy(p, base = base)
    elif method == 'linear':
        Hx = entropy(s, base = base)
    else:
        raise ValueError("Unknown method specified, please choose 'linear' or 'square' ")
    mps.position(center_site)
    
    return Hx

def second_Renyi_entropy(mps: MPS, norm: Optional[float] = 0, base: Optional[float]  = 2)->float:
    ''' Computes the second Renyi entropy of the stochastc MPS, defined as - log2( sum(p^2))
    
    Args:
        mps: the MPS under investigation
        norm: optionally one can provide the l^1 norm of `mps` here to avoid computing the 
              norm various times. If left 0 (default) the l^1 norm is computed
        base: the base of the logarithm. Default is base 2
        
    Returns:
        H2: the second Renyi entorpy of the MPS
    '''
    
    if norm == 0:
        norm = mps.norm(1)
    p2 = mps.norm(2)**2/norm**2
    return - np.log(p2)/np.log(base)


def second_Renyi_EE(mps: MPS, site: int, 
             norm: Optional[float] = 0, 
             base: Optional[float]  = 2, 
             side: Optional[str] = 'L')-> float:
    ''' Computes the second Renyi entanglement entropy across the bond to the right of `site`
    
    Args:
        mps: the MPS class object
        site: integer specifying the bond across which to compute the second Renyi entanglement entropy
        norm: optionally one can provide the l^1 norm of `mps` here to avoid computing the 
              norm various times. If left 0 (default) the l^1 norm is computed
        base: the base of the logarithm. Default is base 2
        side: specify which side of the cut to compute the Second Renyi Entanglement Entropy of, either:
            * 'L' for the left side (default)
            * 'R' for the right side
    
    Returns:
        SREE: the Second Renyi Entanglement Entropy
    '''
    N=len(mps)
    d=mps.physical_dimensions[0]
    if norm == 0:
        norm = mps.norm()
        
    ell1=np.ones(d)
    tens=np.ones(1)
    
    if side == 'L':
        for i in range(N-1,site,-1):
            ketn = np.tensordot(mps.tensors[i], ell1, axes=[1,0])
            tens = np.tensordot(ketn,tens,axes=[1,0])
        end = np.tensordot(tens,tens,axes=[(),()])
        for i in range(site,-1,-1):
            blocki = np.tensordot(mps.tensors[i], end, axes = [2,0])
            end = np.tensordot(blocki, mps.tensors[i], axes = ([1,2],[1,2]) )
        SREE = - np.log(end[0,0]/norm**2)/np.log(base)
    elif side == 'R':
        for i in range(site+1):
            ketn = np.tensordot(mps.tensors[i], ell1, axes=[1,0])
            tens = np.tensordot(tens, ketn, axes=[0,0])
        end = np.tensordot(tens,tens,axes=[(),()])
        for i in range(site+1,N):
            blocki = np.tensordot(mps.tensors[i], end, axes = [0,0])
            end = np.tensordot(blocki, mps.tensors[i], axes = ([0,2],[1,0]) )
        SREE = - np.log(end[0,0]/norm**2)/np.log(base)
    else:
        raise ValueError("Unknown value for 'side', please specify 'L' or  'R'. ")
    
    return SREE

def second_Renyi_MI(mps: MPS, site: int,
                    SRE: Optional[float] = 0,
                    SREEleft: Optional[float] = 0,
                    SREEright: Optional[float] = 0,
                    norm: Optional[float] = 0, 
                    base: Optional[float] = 2)->float:
    ''' Computes the second Renyi mutual information in the MPS across the bond right of `site`
    
    Args:
        mps: MPS class object
        site: integer specifying the bond across which to compute the second Renyi mutual information
        SRE: Second Renyi entropy of 'mps'
        SREEleft: Second Renyi entropy of left side of 'mps' (marginalized over right side) 
        SREEright: Second Renyi entropy of right side of 'mps' (marginalized over left side) 
        norm: optionally one can provide the l^1 norm of `mps` here to avoid computing the 
              norm various times. If left 0 (default) the l^1 norm is computed
        base: the base of the logarithm. Default is base 2
        
    Returns:
        SRMI: the Second Renyi Mutual Information in the MPS 'mps'
        
    '''
    if norm == 0:
        norm = mps.norm()
        
    if SRE == 0:
        SRE = second_Renyi_entropy(mps, norm, base)
    
    if SREEleft == 0:
        SREEleft = second_Renyi_EE(mps, site, norm, base, 'L')
    
    if SREEright == 0:
        SREEright = second_Renyi_EE(mps, site, norm, base, 'R')
        
    return SREEleft + SREEright - SRE 
    
    
# Copyright 2023 stochasticTN Developers, GNU GPLv3

import numpy as np
from typing import Any, Optional, List
from stochasticTN.linalg import svd
import csv
import os

class MPS:
    """ Defines the MPS class for the stochastic MPS package:
    
    Attributes:
    
        * 'MPS.tensors': A list of MPS tensors with length N and index structure:
            0 -- x -- 2
                 |
                 1
        * 'MPS.center': Location of orthogonality site of the MPS. Can be moved with MPS.position(site)
        * 'MPS.bond_dimensions: list of local bond dimensions with length N + 1
        * 'MPS.norm(order): gives the L^order norm of MPS. Only implemented for L^1 and L^2 norms
    
    """
    
    
    def __init__(self, 
                 tensors: List[Any],
                 center: Optional[int] = None,
                 canonicalize: Optional[bool] = True, 
                 name: Optional[str] = None) -> None:
        """Initialize an MPS from a list of tensors
        
        Args:
            tensors: A list of numpy arrays constituting the tensors of the MPS
            center: the initial position of the center node. 
            canonicalize: A Boolean if 'True' MPS is canonicalized at initiation 
            name: a name for the mps
        """
                    
        self.tensors = tensors
        self.center = center
        self.name = name
        
        if canonicalize:
            self.canonicalize()
        
    
    def __len__(self) -> int:
        return len(self.tensors)
    
    def __iter__(self):
        return iter(self.tensors)
    
    @property
    def bond_dimensions(self) -> np.ndarray:
        return np.array([self.tensors[0].shape[0]] + [t.shape[2] for t in self])
    
    @property
    def physical_dimensions(self) -> List:
        return [t.shape[1] for t in self]
    
    def position(self, site: int, normalize_SVs: Optional[bool] = True,
                     Dmax: Optional[int] = None, 
                     cutoff: Optional[float] = 0) -> float:
        ''' Puts MPS in canonical form with respect to `site` such that all 
        tensors to the left and to the right of `site` are unitary
        
        Arg:
            site: the site of the center position
            normalize_SVs: Boolean, if True the vector of singular values s is divided by np.sum(s) 
            Dmax: maximal number of singular values to keep
            cutoff: maximal absolute value for the singular values, SVs below this value are dropped
            
        Returns:
            truncation_error: size of the singular values that are dropped.
        '''
        
        if self.center is None:
            raise ValueError('MPS.center is None, cannot shift its position')
        
        if (site<0 or site>= len(self)):
            raise ValueError(f'Cannot move MPS.center to position {site}, it lies outside the chain.')
            
        truncation_error = 0
        if site == self.center:
            return truncation_error
        
        elif self.center < site:
            for i in range(self.center, site):
                u, s, v, err = svd(self.tensors[i], 2, Dmax, cutoff, normalize_SVs)
                self.tensors[i] = u
                sv = np.tensordot(np.diag(s),v, axes= [1,0]) 
                self.tensors[i+1] = np.tensordot(sv, self.tensors[i+1], axes = [1,0])
                truncation_error += err
        else:
            for i in range(self.center,site, -1):
                u, s, v, err = svd(self.tensors[i], 1, Dmax, cutoff, normalize_SVs)
                self.tensors[i] = v
                us = np.tensordot(u,np.diag(s), axes= [1,0]) 
                self.tensors[i-1] = np.tensordot(self.tensors[i-1], us, axes = [2,0])
                truncation_error += err
                
        self.center = site
            
        return truncation_error
    
    def canonicalize(self, normalize_SVs: Optional[bool] = True,
                     Dmax: Optional[int] = None, 
                     cutoff: Optional[float] = 0) -> float:
        ''' Puts MPS in canonical form with respect to the left-most site (0) such that all 
        tensors on the right are unitary
        
        Arg:
            normalize_SVs: Boolean, if True the vector of singular values s is divided by np.sum(s) 
            Dmax: maximal number of singular values to keep
            cutoff: maximal absolute value for the singular values, SVs below this value are dropped
            
        Returns:
            truncation_error: size of the singular values that are dropped.
        '''
        truncation_error = 0
        n = len(self)
        
        if self.center is None:
            self.center = 0
            
        truncation_error += self.position(n-1, normalize_SVs, Dmax, cutoff)
        truncation_error += self.position(0, normalize_SVs, Dmax, cutoff)
        
        return truncation_error
    
    def norm(self, order: int = 1) -> float:
        """ Returns the norm of the mps, where `order` specifies the L^order norm

        Args:
            order: the order is can be either 1 or 2 for the L^1 or L^2 norm respectively

        Returns:
            norm: the L^order norm of the mps 

        """

        if order == 2:
            if self.center is None:
                self.canonicalize()
                return np.linalg.norm(self.tensors[0])
            else:
                return np.linalg.norm(self.tensors[self.center])
        elif order == 1:
            flat = np.ones(2)
            norm = np.ones(1)
            for i, t in enumerate(self.tensors):
                ttemp = np.tensordot(t,flat, axes = [1,0]) # ncon([flat,t],[[1],[-1,1,-2]])
                norm = np.tensordot(norm, ttemp, axes = [0,0]) #ncon([norm,ttemp],[[1],[1,-1]])
            return norm[0]       
        else:
            raise ValueError("'order' of norm should be either '1' or '2'")
            
    def save(self, name: Optional['str'] = None):
        ''' Routine to save mps tensors to store for later use
        '''
        if name is None and self.name is None:
            raise ValueError('Please provide a name for the MPS')
        elif name is None and self.name is not None:
            name = self.name
        else:
            pass
        
        PATH = 'MPSs/'+name+'/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        
        mps_metadata = {'N' : len(self), 
                        'center': self.center, 
                        'bds': list(self.bond_dimensions), 
                        'pds': self.physical_dimensions}
        
        with open(PATH+name+".csv", "w", newline='') as csvfile:
            fieldnames = list(mps_metadata.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(mps_metadata)
#         w = csv.writer(open(PATH+name+".csv", "w"))
#         for key, val in mps_metadata.items():
#             w.writerow([key, val])
        
        for i, tensor in enumerate(self):
            np.save(PATH+'tensor'+str(i),tensor)
        
            
def randomMPS(N,D,d=2):
    """ Construct a random MPS object of length N with maximal bond dimension D, 
        where each tensor is filled with random numbers
        
    Args:
        N: number of sites
        D: maximal bond dimension of the MPS
        d: physical dimension of each site
        
    Returns:
        MPS: a MPS object of length N and maximal bond dimension D and physical dimension d
             filled with random numbers and canonicalized at the left-most site. 
    """
    tensors = [None]*N
    tensors[0] = np.random.rand(1,d,D)
    for i in range(1,N-1):
        tensors[i] = np.random.rand(D,d,D) # Convention: first index is physical index
    tensors[N-1] = np.random.rand(D,d,1)
    return MPS(tensors,canonicalize=True)

def loadMPS(name: str)->MPS:
    ''' Loads a previously saved MPS and converts it to MPS class object
    '''
    PATH = 'MPSs/'+name+'/'
    mps_metadata = {}
    with open(PATH+name+'.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in row.items():
                mps_metadata[k] = eval(v)
    N = mps_metadata['N']
    center = mps_metadata['center']
    tensors = [None]*N
    for i in range(N):
        tensors[i] = np.load(PATH+'tensor'+str(i)+'.npy')
    
    mps = MPS(tensors, center = center , name = name)
    if list(mps.bond_dimensions) != mps_metadata['bds']:
        raise ValueError('Bond dimensions do not match up!')
    if mps.physical_dimensions != mps_metadata['pds']:
        raise ValueError('Physical dimensions do not match up!')
    
    return mps

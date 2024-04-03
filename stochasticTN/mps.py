# 
#
#
#
#

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
        self.Rs = None
        
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
    
    def norm(self, cx = 'stoch') -> float:
        """ Returns the norm of the mps, where `order` specifies the L^order norm

        Args:
            cx: protocol for the norm: 
            * 'complex' computes the L^2 norm for complex or real mps
            * 'stoch' computes the L^1 norm for real valued mps

        Returns:
            norm: the desired norm of the mps 

        """

        if cx == 'complex':
            if self.center is None:
                self.canonicalize()
                return np.linalg.norm(self.tensors[0])
            else:
                return np.linalg.norm(self.tensors[self.center])
        elif cx == 'stoch':
            flat = np.ones(2)
            norm = np.eye(self.bond_dimensions[0])
            for i, t in enumerate(self.tensors):
                ttemp = np.tensordot(t,flat, axes = [1,0]) # ncon([flat,t],[[1],[-1,1,-2]])
                norm = np.tensordot(norm, ttemp, axes = [1,0]) #ncon([norm,ttemp],[[1],[1,-1]])
            return np.trace(norm)       
        else:
            raise ValueError("'cx' should be either 'stoch' or 'complex'")
            
    def probabilities(self, norm = 0):
        '''
        Explicitly compute all probabilities from the MPS representation. 
        Exponentially costly, so only use for small MPSs to compare with exact results

        Args:
            mps: an MPS representation of the probability distribution

        Returns:
            p: a 2**n dimensional vector with the probabilities of all configurations 
            
        Raises:
            ValueError if the length of the MPS is larger than 28 to prevent the distribution 
                from becoming too large
        '''    
        n = len(self)
        if n > 28:
            raise ValueError("MPS is too large to compute all probabilities!")
            
        if norm == 0:
            norm = self.norm()
            
        p = self.tensors[0]
        for i in range(1,n):
            p = np.tensordot(p, self.tensors[i], axes = (-1,0))
        if self.bond_dimensions[0] != 1:
            p = np.trace(p, axis1 = 0, axis2 = -1)
        
        return p.reshape(2**n)/norm
            
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
            
    def compute_Rs(self):
        ''' Computes all right environments needed for trace and sampling algo
        '''
        n = len(self)
        flat = np.ones(2)
        self.Rs = {n-1: np.ones(1)}
#         self.Rs = {n-2: np.tensordot(self.tensors[n-1], flat, axes = [1,0])}
        for i in range(n-2,-1,-1):
            keti = np.tensordot(self.tensors[i+1], flat, axes = [1,0])
            self.Rs[i] = np.tensordot(keti, self.Rs[i+1], axes = [1,0])
            
    def int_to_sample(self,num, width=None):
        ''' Takes number `num` and returns np.ndarray of the binary representation of this number'''
        return  np.array(list(np.binary_repr(num, width=width)),dtype=int)
    
    def sample_array(self,batch_size=6, method = 'random'):
        '''
        Method to sample array of bits from the proability distribution represented by the mps
        '''
        n = len(self)

        num_batches = n//batch_size
        remainder = n%batch_size

        flat = np.ones(2)
        up=np.array([1,0])
        down=np.array([0,1])

        if self.Rs == None:
            self.compute_Rs()
        else:
            pass

        sample = []
        Ls = {0: np.ones(1)}

        for x in range(num_batches):
            i = x*batch_size
            p = np.tensordot(Ls[x*batch_size], self.tensors[x*batch_size], axes = [-1,0])
            for y in range(1,batch_size):
                i += 1
                p = np.tensordot(p, self.tensors[i], axes = [-1,0])
            if i < n-1:
                p = np.tensordot(p, self.Rs[i], axes = [-1,0])
            p[p<0]=0
            p = p.flatten()/np.sum(p)
            
            if method == 'random':
                choice = np.random.choice([i for i in range(2**batch_size)],p = p)
            elif method == 'greedy':
                choice = np.argmax(p)
            else:
                raise ValueError("Please specify method for sampling")
            sample.extend(self.int_to_sample(choice,width=batch_size))

            for y in range(batch_size):
                i = x*batch_size + y
                temp = np.tensordot(Ls[i], self.tensors[i], axes = [-1,0])

                if sample[i] == 0:
                    Ls[i+1] = np.tensordot(temp, up, axes = [0,0])
                else:
                    Ls[i+1] = np.tensordot(temp,down, axes = [0,0])

        if remainder != 0:
            i = num_batches*batch_size
            p = np.tensordot(Ls[i], self.tensors[i], axes = [-1,0])
            for y in range(1,remainder):
                i +=  1
                p = np.tensordot(p, self.tensors[i], axes = [-1,0])
            if i < n-1:    
                p = np.tensordot(p, self.Rs[i], axes = [-1,0])
            p[p<0]=0
            p = p.flatten()/np.sum(p)
            
            if method == 'random':
                choice = np.random.choice([i for i in range(2**remainder)],p = p)
            elif method == 'greedy':    
                choice = np.argmax(p)
            else:
                raise ValueError("Please specify method for sampling")
                
            sample.extend(self.int_to_sample(choice,width=remainder))


        return np.array(sample)
        
            
def randomMPS(N,D,d=2, bc = 'open'):
    """ Construct a random MPS object of length N with maximal bond dimension D, 
        where each tensor is filled with random numbers
        
    Args:
        N: number of sites
        D: maximal bond dimension of the MPS
        d: physical dimension of each site
        bc: 'open' or 'periodic' boundary conditions
        
    Returns:
        MPS: a MPS object of length N and maximal bond dimension D and physical dimension d
             filled with random numbers and canonicalized at the left-most site. 
    """
    tensors = [None]*N
    if bc == 'open':
        tensors[0] = np.random.rand(1,d,D)
    elif bc == 'periodic':
        tensors[0] = np.random.rand(D,d,D)
    else:
        raise ValueError("Please specify bc as 'open' or 'periodic'.") 
        
    for i in range(1,N-1):
        tensors[i] = np.random.rand(D,d,D) # Convention: middle index is physical index
    
    if bc == 'open':
        tensors[N-1] = np.random.rand(D,d,1)
    elif bc == 'periodic':
        tensors[N-1] = np.random.rand(D,d,D)
        
    return MPS(tensors,canonicalize=True)

def occupied_mps(N,D, bc = 'open'):
    '''
    Creates an MPS with bond dimension D where all sites are in the 'occupied' (down) state
    
    Args:
        N: number of sites
        D: maximal bond dimension
        bc: boundary conditions
            - 'open' for open boundary conditions
            - 'periodic' for periodic boundary conditions 
    '''
    tensors = N*[None]
    bare = np.zeros((D,2,D))
    bare[:,1,:] = np.eye(D,D)
    
    for i in range(N):
        tensors[i] = bare
    
    if bc == 'open':
        L = np.zeros((1,2,D))
        R = np.zeros((D,2,1))
        L[:,1,:] = np.ones((1,D))/D
        R[:,1,:] = np.ones((D,1))
        tensors[0] = L
        tensors[N-1] = R
        
    return MPS(tensors, canonicalize = False)

def uniform_mps(N,D, bc = 'open'):
    '''
    Creates an MPS with bond dimension D where all configurations appear with equal probability
    
    Args:
        N: number of sites
        D: maximal bond dimension
        bc: boundary conditions
            - 'open' for open boundary conditions
            - 'periodic' for periodic boundary conditions 
    Returns:
        mps: the mps
    '''
    tensors = N*[None]
    bare = np.zeros((D,2,D))
    bare[:,0,:] = np.eye(D,D)
    bare[:,1,:] = np.eye(D,D)
    
    for i in range(N):
        tensors[i] = bare
    
    if bc == 'open':
        L = np.zeros((1,2,D))
        R = np.zeros((D,2,1))
        L[:,0,:] = np.ones((1,D))/(D*N)
        L[:,1,:] = np.ones((1,D))/(D*N)
        R[:,0,:] = np.ones((D,1))
        R[:,1,:] = np.ones((D,1))
        tensors[0] = L
        tensors[N-1] = R
    
    return MPS(tensors, canonicalize=False)

def mps_from_array(array):
    '''
    Creates an direct product MPS with bond dimension 1 from the bits specified in 'array'
    
    Args:
        N: number of sites
        D: maximal bond dimension
        bc: boundary conditions
            - 'open' for open boundary conditions
            - 'periodic' for periodic boundary conditions 
    
    Returns:
        mps: the mps    
    '''
    N = len(array)
    tensors = N*[None]
    zero = np.array([1,0]).reshape(1,2,1)
    one = np.array([0,1]).reshape(1,2,1)
    
    for i in range(N):
        if array[i] == 0:
            tensors[i] = zero
        elif array[i] == 1:
            tensors[i] = one
        else:
            raise ValueError("Input array should contain only bits (0 or 1)") 

    return MPS(tensors, canonicalize=False)


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
    